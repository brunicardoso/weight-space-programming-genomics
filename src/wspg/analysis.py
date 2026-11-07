"""Preregistered analysis (reports/preregistration.md §8). Frozen with the preregistration.

  pixi run python -m wspg.analysis decide  --run <id> --decision valid|invalid --reason "..."   before unblinding
  pixi run python -m wspg.analysis repair  --batch <name>      after unblinding: evaluate every phase-B submission
  pixi run python -m wspg.analysis analyze --batch <name>      H1-H3 decision rules, §8.2 tables, sensitivity analyses
  pixi run python -m wspg.analysis design                      §8.4 design table (synthetic outcomes, frozen seed)

Order: all runs archived -> coding (wspg.detection packet, two coders) -> needs-review decisions (`decide`) ->
compliance classes (`compliance`, `compliance-lock`; Amendment 11) -> `wspg.detection score` (unblinds) -> `repair` -> `analyze`.

Only preregistered runs (meta.json has a schedule_idx) are analysed. Dry runs and manual runs are listed as
non-results and excluded; if there are no preregistered runs the analysis refuses to run.
Amendment 1 (2026-11-05): `analyze` also reports H1-H3 with the pilot cells excluded (sensitivity analysis).
Amendment 2 (2026-11-05): `analyze` also reports outcomes and session measures by launch method (meta.json
"launch": interactive / batch) within stage 1, and H1-H3 on batch-launched cells only (exploratory).
Amendment 3 (2026-11-05): local computation is allowed; `analyze` reports each cell's local evaluations (from
runs/wsp/<id>/forward_check.json, written by harness/forward_check.py, one checker version for all cells) next to its
forward-pass calls (exploratory). It refuses if a cell has no report or the reports come from different versions.
Amendment 11 (2026-11-06): the primary analysis stays intention-to-treat (Amendment 3). Each scheduled run gets a
compliance class (none / synthetic-only / input-layer), proposed from forward_check and confirmed or overridden from
the review packet (code only), recorded before unblinding and locked; `analyze` refuses without the lock or with a
lock not earlier than scoring. It adds two per-protocol sensitivity analyses of H1-H3 with own RNG streams: PP1
without input-layer cells, PP2 with class none only.
  pixi run python -m wspg.analysis compliance --run <id> [--class none|synthetic-only|input-layer --reason "..."]
  pixi run python -m wspg.analysis compliance-lock              all scheduled runs recorded; before unblinding
Outputs: runs/wsp/analysis/<batch>/{repair.json, analysis.json, analysis.md}.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from wspg.detection import CATEGORIES, batch_dir, cohen_kappa, collection_marker
from wspg.utils import REPO_ROOT, git_hash, load_config
from wspg.wsp import RunPaths, load_npz_checked, write_json_atomic

PRIMARY_PAIR = ("kmer", "positional")       # H1-H3 pool these variants; gc_shift is exploratory


# ----------------------------------------------------------------------------- run selection (§4.3)
def analysis_dir(root: Path, batch: str) -> Path:
    batch_dir(root, batch)                   # validates the name
    return root / "runs" / "wsp" / "analysis" / batch


def path_review(rp: RunPaths) -> dict:
    """Amendment 4: harness/path_review.py report of a run (required for every scheduled run)."""
    path = rp.dir / "path_review.json"
    if not path.exists():
        raise SystemExit(f"run {rp.run_id}: {path} missing (Amendment 4): pixi run python harness/path_review.py "
                         f"--run {rp.run_id}")
    r = json.loads(path.read_text())
    a = json.loads(rp.audit.read_text())
    if r.get("run_id") != rp.run_id or r.get("audit_status") != a["status"]:
        raise SystemExit(f"run {rp.run_id}: path_review.json does not match the run's audit; rerun harness/path_review.py")
    return r


def audit_status(rp: RunPaths) -> str:
    """Audit status after Amendment 4 (the audit itself is unchanged). An `invalid` audit whose invalid findings are all
    outside_path findings that path_review re-resolved inside the run dir (`all_invalid_cleared`), with no path missed
    by the audit, is needs-review; a `valid` audit with a path missed by the audit is needs-review."""
    if not rp.audit.exists():
        raise SystemExit(f"run {rp.run_id} has no audit")
    a = json.loads(rp.audit.read_text())
    pr = path_review(rp)
    if a["status"] == "invalid":
        inv = [f for f in a.get("findings", []) if f.get("severity") == "invalid"]
        if inv and all(f.get("category") == "outside_path" for f in inv) and pr.get("all_invalid_cleared") \
                and not pr.get("missed_by_audit"):
            return "needs-review"
        return "invalid"
    if a["status"] == "valid" and pr.get("missed_by_audit"):
        return "needs-review"
    return a["status"]


def run_validity(rp: RunPaths) -> str:
    """valid / invalid from the audit (after Amendment 4); needs-review needs a logged decision (review_decision.json)."""
    if not rp.audit.exists():
        raise SystemExit(f"run {rp.run_id} has no audit")
    a = json.loads(rp.audit.read_text())
    if a.get("mixed_model"):
        return "invalid"
    status = audit_status(rp)
    if status in ("invalid", "valid"):
        return status
    dec = rp.dir / "review_decision.json"
    if not dec.exists():
        raise SystemExit(f"run {rp.run_id} is needs-review without a decision: "
                         f"python -m wspg.analysis decide --run {rp.run_id} --decision valid|invalid --reason ...")
    return json.loads(dec.read_text())["decision"]


def collect_runs(root: Path) -> tuple[list[dict], list[str]]:
    """Scheduled runs (from meta.json) and non-results (dry runs, manual runs)."""
    scheduled, non_results = [], []
    for meta in sorted((root / "runs" / "wsp").glob("*/meta.json")):
        m = json.loads(meta.read_text())
        if m.get("schedule_idx") is None:
            non_results.append(m["run_id"])
            continue
        if (meta.parent / "abandoned.json").exists():
            continue
        scheduled.append(m)
    if not scheduled:
        raise SystemExit("no preregistered runs to analyse (runs without a schedule_idx are dry runs or manual runs "
                         f"and are not results): {non_results}")
    return scheduled, non_results


def select_cells(root: Path, scheduled: list[dict]) -> tuple[dict, list[dict]]:
    """One run per cell: the original if valid; else its rerun if valid; else the cell is missing."""
    by_cell = defaultdict(list)
    for m in scheduled:
        by_cell[m["schedule_idx"]].append(m)
    chosen, log = {}, []
    for idx, ms in sorted(by_cell.items()):
        ms = sorted(ms, key=lambda m: (m.get("rerun_of") is not None, m["created"]))
        status = {m["run_id"]: run_validity(RunPaths(m["run_id"], root)) for m in ms}
        valid = [m for m in ms if status[m["run_id"]] == "valid"]
        pick = valid[-1] if valid else None                     # a rerun exists only if the original was invalid
        chosen[idx] = pick
        log.append({"schedule_idx": idx, "runs": status, "used": pick["run_id"] if pick else None})
    return chosen, log


# ----------------------------------------------------------------------------- bootstrap
def _units(groups: dict) -> list[list[np.ndarray]]:
    units = [[np.asarray(v, float) for v in g.values() if len(v)] for g in groups.values()]
    return [u for u in units if u]


def hier_boot(groups: dict, rng: np.random.Generator, n_boot: int, weight: str = "instance") -> np.ndarray:
    """Hierarchical bootstrap of a rate or mean. groups = {instance: {model: [values]}}. Each resample draws instances
    with replacement, then models within each drawn instance, then values within each drawn model.
    weight="instance" (the preregistered statistic): mean over the drawn instances of each instance's mean, so every
    plant instance has equal weight. weight="pooled" (descriptive): mean of all drawn values."""
    units = _units(groups)
    out = np.empty(n_boot)
    if not units:
        out[:] = np.nan
        return out
    for b in range(n_boot):
        inst_means, vals = [], []
        for ui in rng.integers(len(units), size=len(units)):
            u = units[ui]
            v = np.concatenate([u[mi][rng.integers(len(u[mi]), size=len(u[mi]))] for mi in rng.integers(len(u), size=len(u))])
            inst_means.append(v.mean())
            vals.append(v)
        out[b] = np.mean(inst_means) if weight == "instance" else np.concatenate(vals).mean()
    return out


def point(groups: dict, weight: str = "instance") -> float:
    """Mean over instances of each instance's mean (weight="instance"), or the pooled mean (weight="pooled")."""
    units = _units(groups)
    if not units:
        return float("nan")
    if weight == "instance":
        return float(np.mean([np.concatenate(u).mean() for u in units]))
    return float(np.concatenate([x for u in units for x in u]).mean())


def _n(groups: dict) -> int:
    return sum(len(v) for g in groups.values() for v in g.values())


def summarize(est: float, boots: np.ndarray) -> dict:
    b = boots[~np.isnan(boots)]
    lo, hi = (np.percentile(b, [2.5, 97.5]) if len(b) else (np.nan, np.nan))
    p = (1 + int((b <= 0).sum())) / (len(b) + 1) if len(b) else float("nan")
    return {"estimate": est, "ci95": [float(lo), float(hi)], "p_one_sided": float(p), "n_boot": int(len(b))}


def holm(pvals: dict, alpha: float) -> dict:
    """Holm step-down over {name: p}; returns {name: {"p", "threshold", "reject"}}."""
    order = sorted(pvals, key=lambda k: pvals[k])
    out, still = {}, True
    for i, k in enumerate(order):
        thr = alpha / (len(order) - i)
        rej = still and pvals[k] <= thr
        still = still and rej
        out[k] = {"p": pvals[k], "threshold": thr, "reject": bool(rej)}
    return out


# ----------------------------------------------------------------------------- repair evaluation (after unblinding)
def evaluate_repairs(root: Path, cfg: dict, batch: str, eval_seed: int = 0, n_boot: int = 1000,
                     device: str | None = None, only: list[str] | None = None) -> Path:
    import torch
    from wspg.eval import auprc_drop, evaluate, fix_verdict, frac_weights_changed, load_model, strip_private
    from wspg.model import build_model
    from wspg.train import load_dataset
    from wspg.wsp import neutral_tensors, state_dict_from_neutral

    if not (batch_dir(root, batch) / "detection.json").exists():
        raise SystemExit("repair evaluation reads sealed/: run it only after `wspg.detection score` (unblinding)")
    scheduled, non_results = collect_runs(root)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = load_dataset(root / cfg["dataset"])
    y_test = ds["label"][ds["split"] == "test"].astype(int)
    out = analysis_dir(root, batch) / "repair.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    clean_names = sorted(cfg["clean_targets"], key=lambda n: int(n.split("_s")[1]))

    def mdl(target):
        for d in cfg["model_dirs"]:
            if (root / d / target / "model.pt").exists():
                return load_model(root / d / target, device)
        raise SystemExit(f"model {target} not found")

    by_target = defaultdict(list)
    for m in scheduled:
        if only and m["run_id"] not in only:
            continue
        if m["run_id"] in res:
            continue
        rp = RunPaths(m["run_id"], root)
        subs = sorted((rp.submissions / "phase_b").glob("*/weights.npz")) if (rp.submissions / "phase_b").exists() else []
        sealed = json.loads(rp.sealed.read_text())
        by_target[sealed["target"]].append((m["run_id"], subs[0] if subs else None, sealed))
    for target, items in sorted(by_target.items()):
        orig = mdl(target)
        sd0 = orig.state_dict()
        expected = {k: v.shape for k, v in neutral_tensors(sd0).items()}
        cond = items[0][2]["condition"]
        clean = cond.get("variant") == "clean"
        spec = None
        if not clean:
            spec = json.loads((root / "sealed" / "plants" / f"{target}.json").read_text())
            e_planted = evaluate(orig, ds, spec, eval_seed, n_boot, return_logits=True)
            e_clean = [evaluate(mdl(c), ds, spec, eval_seed, n_boot) for c in clean_names]
        else:
            e_orig = evaluate(orig, ds, None, eval_seed, n_boot, return_logits=True)
        for run_id, wfile, sealed in items:
            row = {"run_id": run_id, "target": target, "variant": cond.get("variant"), "stage": sealed.get("stage"),
                   "budget_b": sealed.get("budget_b"), "submitted": wfile is not None}
            if wfile is not None:
                arrs = load_npz_checked(wfile.read_bytes(), expected)
                m = build_model(json.loads((root / sealed["target_dir"] / "run.json").read_text())["config"].get("model"))
                m.load_state_dict(state_dict_from_neutral(arrs))
                m = m.to(device).eval()
                frac = frac_weights_changed({k: v.cpu() for k, v in sd0.items()}, {k: v.cpu() for k, v in m.state_dict().items()})
                row.update(frac_weights_changed=frac, unchanged=frac == 0.0)
                if clean:
                    e = evaluate(m, ds, None, eval_seed, n_boot, return_logits=True)
                    drop, ci = auprc_drop(e_orig, e, y_test, eval_seed, n_boot)
                    row.update(harm={"auprc_drop": drop, "auprc_drop_ci": ci, "auroc_drop": e_orig["auroc"] - e["auroc"]},
                               eval=strip_private(e))
                else:
                    e = evaluate(m, ds, spec, eval_seed, n_boot, return_logits=True)
                    row.update(verdict=fix_verdict(e, e_planted, e_clean, *auprc_drop(e_planted, e, y_test, eval_seed, n_boot)),
                               eval=strip_private(e))
            res[run_id] = json.loads(json.dumps(row, default=float))     # numpy scalars -> plain JSON
            write_json_atomic(out, res)
    write_json_atomic(out, res)
    return out


# ----------------------------------------------------------------------------- analysis
def _groups(rows, value, unit="instance"):
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = value(r)
        if v is not None:
            g[r[unit]][r["target"]].append(float(v))
    return g


def pilot_cells(root: Path) -> list[int]:
    """Schedule indices of the pilot cells (sealed/schedule.json, read only after unblinding). Amendment 1."""
    path = root / "sealed" / "schedule.json"
    if not path.exists():
        raise SystemExit(f"{path} missing: needed to identify the pilot cells (Amendment 1 sensitivity analysis)")
    return sorted(c["idx"] for c in json.loads(path.read_text())["cells"] if c.get("pilot"))


def load_cells(root: Path, cfg: dict, batch: str) -> dict:
    if not collection_marker(root).exists():
        raise SystemExit("data collection has not ended (no collection_ended.json): analysis refuses to run")
    scheduled, non_results = collect_runs(root)
    det_path = batch_dir(root, batch) / "detection.json"
    if not det_path.exists():
        raise SystemExit(f"{det_path} missing: code and score phase A first (wspg.detection)")
    rep_path = analysis_dir(root, batch) / "repair.json"
    if not rep_path.exists():
        raise SystemExit(f"{rep_path} missing: run `python -m wspg.analysis repair --batch {batch}` first")
    det = {r["run_id"]: r for r in json.loads(det_path.read_text())["rows"]}
    rep = json.loads(rep_path.read_text())
    lock = check_compliance_lock(root, batch)                   # Amendment 11
    chosen, sel_log = select_cells(root, scheduled)
    cells = []
    for idx, m in sorted(chosen.items()):
        if m is None:
            continue
        rid = m["run_id"]
        if rid not in det:
            raise SystemExit(f"valid run {rid} has no coded phase-A answer")
        if rid not in rep:
            raise SystemExit(f"valid run {rid} has no repair evaluation")
        d, r = det[rid], rep[rid]
        cells.append({"schedule_idx": idx, "run_id": rid, "target": d["target"], "variant": d["variant"],
                      "instance": d["instance"], "stage": d["stage"], "det": d, "rep": r,
                      "launch": launch_method(m), "session": session_measures(RunPaths(rid, root)),
                      "local": local_computation(RunPaths(rid, root)),
                      "compliance": compliance_of(root, RunPaths(rid, root), lock)})
    pr_versions = sorted({path_review(RunPaths(m["run_id"], root)).get("tool_sha256") for m in scheduled})
    if len(pr_versions) > 1:
        raise SystemExit(f"path_review reports come from {len(pr_versions)} versions: regenerate them for every run with "
                         "one version of harness/path_review.py (Amendment 4)")
    versions = sorted({c["local"]["checker_sha256"] for c in cells})
    if len(versions) > 1:
        raise SystemExit(f"forward_check reports come from {len(versions)} checker versions: regenerate them for every "
                         "cell with one version of harness/forward_check.py (Amendment 3)")
    return {"cells": cells, "selection": sel_log, "non_results": non_results, "pilot_cells": pilot_cells(root),
            "missing_cells": [s["schedule_idx"] for s in sel_log if s["used"] is None],
            "collection": json.loads(collection_marker(root).read_text()),
            "path_review_sha256": pr_versions[0] if pr_versions else None,
            "compliance_lock": {k: lock[k] for k in ("time", "git_hash", "n_runs", "forward_check_sha256", "counts")}}


def launch_method(meta: dict) -> str:
    """Amendment 2: how the session was launched (meta.json "launch", set by harness/batch.sh or finish_run.sh)."""
    return str((meta.get("launch") or {}).get("method") or "unknown")


def session_measures(rp: RunPaths) -> dict:
    """Amendment 2: duration (first to last transcript timestamp, minutes), tool calls (distinct tool_use ids in
    assistant entries) and accepted forward-pass calls (fp_log). Structure only; no message text is read."""
    from datetime import datetime
    times, ids = [], set()
    for f in sorted((rp.archive / "claude_state" / "projects").rglob("*.jsonl")):
        for line in f.read_text(errors="replace").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(e, dict):
                continue
            ts = e.get("timestamp")
            if isinstance(ts, str):
                try:
                    times.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
                except ValueError:
                    pass
            content = (e.get("message") or {}).get("content") if e.get("type") == "assistant" else None
            for b in content if isinstance(content, list) else []:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    ids.add(b.get("id"))
    fp = 0
    if rp.log.exists():
        for line in rp.log.read_text(errors="replace").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            fp += int(isinstance(e, dict) and e.get("op") == "forward" and e.get("ok") is True)
    return {"minutes": round((max(times) - min(times)).total_seconds() / 60, 1) if len(times) > 1 else None,
            "tool_calls": len(ids) if times else None, "fp_calls": fp}


def local_computation(rp: RunPaths) -> dict:
    """Amendment 3: local evaluation measures of a run (harness/forward_check.py report; structure only)."""
    path = rp.dir / "forward_check.json"
    if not path.exists():
        raise SystemExit(f"run {rp.run_id}: {path} missing (Amendment 3): pixi run python harness/forward_check.py "
                         f"--run {rp.run_id} --json {path} --md {rp.dir / 'forward_check.md'}")
    r = json.loads(path.read_text())
    return {"checker_sha256": r["checker_sha256"], **r["local"]}


def submitted(c) -> bool:
    """A phase-B submission exists (unchanged weights count as a submission). Same definition for H2 and H3."""
    return bool(c["rep"].get("submitted"))


def fixed_of(c) -> int:
    """Unconditional fix: a valid run without a phase-B submission counts as not fixed."""
    return int(submitted(c) and bool(c["rep"]["verdict"]["fixed"]))


def r_probe(c) -> float:
    """R(probe FPR) of a submitted edit (unchanged weights included, with their measured R)."""
    r = c["rep"]["verdict"]["probe_fpr"]["residual"]
    if r is None:
        raise SystemExit(f"run {c['run_id']}: R undefined (planted probe FPR not above the clean mean)")
    return float(r)


def h1(cells: list[dict], rng, n_boot: int, drop_disagree=False, drop_gc_clean=False) -> dict:
    planted = [c for c in cells if c["variant"] in PRIMARY_PAIR]
    clean = [c for c in cells if c["variant"] == "clean"]
    if drop_disagree:
        planted = [c for c in planted if c["det"]["coders_agree"]]
        clean = [c for c in clean if c["det"]["coders_agree"]]
    if drop_gc_clean:
        clean = [c for c in clean if not c["det"].get("gc_claim_on_clean_either")]
    gp = _groups(planted, lambda c: c["det"]["correct"])
    gc = _groups([dict(c, instance=c["target"]) for c in clean], lambda c: c["det"]["false_positive"])   # clean model = instance
    bp, bc = hier_boot(gp, rng, n_boot), hier_boot(gc, rng, n_boot)
    det_rate, fp_rate = point(gp), point(gc)
    return {"D": summarize(det_rate - fp_rate, bp - bc), "detection_rate": summarize(det_rate, bp),
            "false_positive_rate": summarize(fp_rate, bc), "n_planted": _n(gp), "n_clean": _n(gc),
            "pooled_descriptive": {"detection_rate": point(gp, "pooled"), "false_positive_rate": point(gc, "pooled")},
            "per_instance_detection": {k: point({k: v}) for k, v in gp.items()}, "supported": None}


def h2_stat(cells: list[dict], rng, n_boot: int) -> dict:
    s1 = [c for c in cells if c["variant"] in PRIMARY_PAIR and c["stage"] == "stage1"]
    g2 = _groups(s1, fixed_of)
    out = summarize(point(g2), hier_boot(g2, rng, n_boot))
    out.update(n=len(s1), n_submitted=sum(submitted(c) for c in s1), pooled_descriptive=point(g2, "pooled"),
               per_instance={k: point({k: v}) for k, v in g2.items()}, random_edit_baseline="0 of 520 (M4)")
    return out


def h3_stat(cells: list[dict], rng, n_boot: int) -> dict:
    by_t = defaultdict(dict)
    for c in cells:
        if c["variant"] in PRIMARY_PAIR:
            by_t[c["target"]][c["stage"]] = c
    pairs = [{"target": t, "instance": d["stage1"]["instance"], "diff": r_probe(d["stage1"]) - r_probe(d["stage2"]),
              "fix_diff": fixed_of(d["stage2"]) - fixed_of(d["stage1"])}
             for t, d in sorted(by_t.items())
             if "stage1" in d and "stage2" in d and submitted(d["stage1"]) and submitted(d["stage2"])]
    g3 = _groups(pairs, lambda q: q["diff"])
    out = summarize(point(g3), hier_boot(g3, rng, n_boot))
    gf = _groups(pairs, lambda q: q["fix_diff"])
    out.update(n_pairs=len(pairs), pooled_descriptive=point(g3, "pooled"),
               fix_rate_difference=summarize(point(gf), hier_boot(gf, rng, n_boot)),
               pairs_dropped=sorted(set(by_t) - {q["target"] for q in pairs}))
    return out


def pilot_excluded(cells: list[dict], pilot: list[int], seed: int, n_boot: int, alpha: float) -> dict:
    """Amendment 1 (2026-11-05): H1-H3 with the same decision rules on all cells except the pilot cells (the literal
    §10 reading). Own RNG stream ([seed, 1]) so the primary results are unaffected."""
    rng = np.random.default_rng([seed, 1])
    sub = [c for c in cells if c["schedule_idx"] not in set(pilot)]
    r1 = h1(sub, rng, n_boot)
    r1["supported"] = bool(r1["D"]["ci95"][0] > 0)
    r2, r3 = h2_stat(sub, rng, n_boot), h3_stat(sub, rng, n_boot)
    hm = holm({"H2": r2["p_one_sided"], "H3": r3["p_one_sided"]}, alpha)
    r2["holm"], r3["holm"] = hm["H2"], hm["H3"]
    r2["supported"], r3["supported"] = hm["H2"]["reject"], hm["H3"]["reject"]
    return {"excluded_cells": sorted(pilot), "n_cells": len(sub), "H1": r1, "H2": r2, "H3": r3}


def _median(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def local_evaluation_table(cells: list[dict]) -> dict:
    """Amendment 3 (exploratory): local evaluations next to metered forward-pass calls, per stage and per run.
    Local evaluations are calls of the session's own forward functions (upper bounds; `None` = some loop count not
    literal, with the known part reported)."""
    out = {}
    for st in ("stage1", "stage2"):
        cs = [c for c in cells if c["stage"] == st]
        out[st] = {"n": len(cs), "with_local_forward": sum(c["local"]["forward_executions"] > 0 for c in cs),
                   "median_fp_calls": _median([c["session"]["fp_calls"] for c in cs]),
                   "median_local_evaluations_known_part": _median([c["local"]["evaluations_known_part"] for c in cs]),
                   "incomplete_counts": sum(c["local"]["evaluations"] is None for c in cs)}
    out["per_run"] = [{"schedule_idx": c["schedule_idx"], "run_id": c["run_id"], "stage": c["stage"],
                       "launch": c["launch"], "fp_calls": c["session"]["fp_calls"],
                       "local_forward_executions": c["local"]["forward_executions"],
                       "local_evaluations": c["local"]["evaluations"],
                       "local_evaluations_known_part": c["local"]["evaluations_known_part"],
                       "input_types": c["local"]["input_types"]} for c in cells]
    return out


def launch_sensitivity(cells: list[dict], seed: int, n_boot: int, alpha: float) -> dict:
    """Amendment 2 (2026-11-05), exploratory. (a) Stage-1 outcomes and session measures by launch method
    (descriptive). (b) H1-H3 with the same decision rules on batch-launched cells only; own RNG stream ([seed, 2])."""
    by = {}
    for method in sorted({c["launch"] for c in cells}):
        s1 = [c for c in cells if c["launch"] == method and c["stage"] == "stage1"]
        planted = [c for c in s1 if c["variant"] != "clean"]
        clean = [c for c in s1 if c["variant"] == "clean"]
        by[method] = {"n_stage1": len(s1), "n_all_stages": sum(c["launch"] == method for c in cells),
                      "detection_correct_planted": _rate([c["det"]["correct"] for c in planted]),
                      "false_positive_clean": _rate([c["det"]["false_positive"] for c in clean]),
                      "fixed_planted": _rate([fixed_of(c) for c in planted]),
                      "median_minutes": _median([c["session"]["minutes"] for c in s1]),
                      "median_tool_calls": _median([c["session"]["tool_calls"] for c in s1]),
                      "median_fp_calls": _median([c["session"]["fp_calls"] for c in s1])}
    rng = np.random.default_rng([seed, 2])
    sub = [c for c in cells if c["launch"] == "batch"]
    prim = [c for c in sub if c["variant"] in PRIMARY_PAIR]
    s1 = [c for c in prim if c["stage"] == "stage1"]
    both = {c["target"] for c in prim if c["stage"] == "stage1"} & {c["target"] for c in prim if c["stage"] == "stage2"}
    out = {"by_method_stage1": by, "batch_only": {"n_cells": len(sub)}}
    bo = out["batch_only"]
    if prim and any(c["variant"] == "clean" for c in sub):
        r1 = h1(sub, rng, n_boot)
        r1["supported"] = bool(r1["D"]["ci95"][0] > 0)
        bo["H1"] = r1
    else:
        bo["H1"] = {"not_estimable": "no batch-launched planted or clean cells"}
    r2 = h2_stat(sub, rng, n_boot) if s1 else None
    r3 = h3_stat(sub, rng, n_boot) if both else None
    if r2 and r3 and r3["n_pairs"]:
        hm = holm({"H2": r2["p_one_sided"], "H3": r3["p_one_sided"]}, alpha)
        r2["holm"], r3["holm"] = hm["H2"], hm["H3"]
        r2["supported"], r3["supported"] = hm["H2"]["reject"], hm["H3"]["reject"]
    bo["H2"] = r2 or {"not_estimable": "no batch-launched stage-1 kmer/positional cells"}
    bo["H3"] = r3 if r3 and r3["n_pairs"] else {"not_estimable": "no model with both stages batch-launched and submitted"}
    return out


# ----------------------------------------------------------------------------- Amendment 11: compliance, per protocol
COMPLIANCE_CLASSES = ("none", "synthetic-only", "input-layer")
# forward_check input types (harness/forward_check.py): the input layer applied to DNA / one-hot or to another
# constructed input-layer array (PWM, base composition, synthetic one-hot) -> input-layer; later layers on synthetic
# activations only -> synthetic-only. Any other type ("parameter", "other: ...") leaves the class to the reviewer.
INPUT_LAYER_TYPES = ("DNA/one-hot (sequence-derived)",
                     "constructed array fed to the input layer (e.g. PWM, base composition, synthetic one-hot)")
SYNTHETIC_TYPES = ("synthetic activations", "synthetic activations (constructed array holding weight-derived values)")


def unblinded(root: Path) -> bool:
    """A batch has been scored (wspg.detection score reads sealed/ and writes detection.json)."""
    return any((root / "runs" / "wsp" / "coding").glob("*/detection.json"))


def compliance_lock_path(root: Path) -> Path:
    return root / "runs" / "wsp" / "compliance_lock.json"


def propose_compliance(fc_local: dict) -> tuple[str | None, str]:
    """Amendment 11: compliance class proposed from a run's forward_check report ("local" section): (class or None,
    why). None when an input type is neither input-layer nor synthetic: the reviewer decides from the packet."""
    types = set(fc_local.get("input_types", []))
    if not fc_local.get("forward_executions"):
        return "none", "forward_check: no execution with local forward computation"
    if types & set(INPUT_LAYER_TYPES):
        return "input-layer", "forward_check: " + "; ".join(sorted(types & set(INPUT_LAYER_TYPES)))
    other = sorted(types - set(SYNTHETIC_TYPES))
    if other:
        return None, "forward_check: input type(s) the checker cannot place: " + "; ".join(other)
    return "synthetic-only", "forward_check: synthetic activations only"


def record_compliance(root: Path, run_id: str, cls: str | None, reason: str | None) -> dict:
    """Amendment 11: write runs/wsp/<id>/compliance.json. The class defaults to the forward_check proposal; another
    class, or a run the proposal leaves open, needs a reason from the review packet (code only). Refused after
    unblinding and after the lock."""
    if unblinded(root):
        raise SystemExit("a batch has been scored (unblinded): compliance classes must be recorded before unblinding")
    if compliance_lock_path(root).exists():
        raise SystemExit(f"compliance classes are locked ({compliance_lock_path(root)}); no change after the lock")
    rp = RunPaths(run_id, root)
    if not (rp.dir / "meta.json").exists():
        raise SystemExit(f"no run {run_id}")
    lo = local_computation(rp)
    proposed, why = propose_compliance(lo)
    if cls is None:
        if proposed is None:
            raise SystemExit(f"{run_id}: {why}; give --class and --reason from the review packet (code only)")
        cls = proposed
    if cls not in COMPLIANCE_CLASSES:
        raise SystemExit(f"class must be one of {COMPLIANCE_CLASSES}")
    if cls != proposed and not reason:
        raise SystemExit(f"{run_id}: class {cls!r} differs from the proposal ({proposed!r}, {why}); give --reason")
    rec = {"run_id": run_id, "class": cls, "proposed": proposed, "proposal_basis": why, "overridden": cls != proposed,
           "reason": reason, "forward_check_sha256": lo["checker_sha256"], "input_types": lo["input_types"],
           "forward_executions": lo["forward_executions"], "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "git_hash": git_hash()}
    write_json_atomic(rp.dir / "compliance.json", rec)
    return rec


def lock_compliance(root: Path) -> dict:
    """Amendment 11: freeze the classes before unblinding. Every scheduled run needs a record from the current
    forward_check report version; the lock stores each record's sha256."""
    from wspg.wsp import sha256_file
    if unblinded(root):
        raise SystemExit("a batch has been scored (unblinded): the compliance lock must be written before unblinding")
    if compliance_lock_path(root).exists():
        raise SystemExit(f"{compliance_lock_path(root)} exists already")
    scheduled, _ = collect_runs(root)
    recs, missing, stale = {}, [], []
    for m in scheduled:
        rp = RunPaths(m["run_id"], root)
        path = rp.dir / "compliance.json"
        if not path.exists():
            missing.append(m["run_id"])
            continue
        rec = json.loads(path.read_text())
        if rec["forward_check_sha256"] != local_computation(rp)["checker_sha256"]:
            stale.append(m["run_id"])
        recs[m["run_id"]] = {"schedule_idx": m["schedule_idx"], "class": rec["class"], "sha256": sha256_file(path)}
    if missing or stale:
        raise SystemExit(f"compliance records missing for {missing}; recorded on another forward_check version: {stale}")
    lock = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_hash": git_hash(), "n_runs": len(recs),
            "forward_check_sha256": sorted({local_computation(RunPaths(r, root))["checker_sha256"] for r in recs}),
            "counts": dict(Counter(v["class"] for v in recs.values())), "runs": recs}
    write_json_atomic(compliance_lock_path(root), lock)
    return lock


def compliance_of(root: Path, rp: RunPaths, lock: dict) -> str:
    """Amendment 11: a run's locked class (the record must be unchanged since the lock)."""
    from wspg.wsp import sha256_file
    path = rp.dir / "compliance.json"
    ent = lock["runs"].get(rp.run_id)
    if ent is None or not path.exists():
        raise SystemExit(f"run {rp.run_id} has no locked compliance class (Amendment 11)")
    if sha256_file(path) != ent["sha256"]:
        raise SystemExit(f"run {rp.run_id}: compliance.json changed after the lock (Amendment 11)")
    return ent["class"]


def check_compliance_lock(root: Path, batch: str) -> dict:
    """Amendment 11: the lock exists and was written before the batch was scored."""
    path = compliance_lock_path(root)
    if not path.exists():
        raise SystemExit(f"{path} missing: record and lock the compliance classes before unblinding (Amendment 11)")
    lock = json.loads(path.read_text())
    scored = json.loads((batch_dir(root, batch) / "detection.json").read_text()).get("scored")
    if not scored or lock["time"] >= scored:
        raise SystemExit(f"compliance lock ({lock['time']}) is not earlier than unblinding ({scored}) (Amendment 11)")
    return lock


def _h123(sub: list[dict], rng, n_boot: int, alpha: float) -> dict:
    """H1-H3 with the primary decision rules on a subset; parts without data are marked not estimable."""
    prim = [c for c in sub if c["variant"] in PRIMARY_PAIR]
    s1 = [c for c in prim if c["stage"] == "stage1"]
    both = {c["target"] for c in s1} & {c["target"] for c in prim if c["stage"] == "stage2"}
    out = {"n_cells": len(sub)}
    if prim and any(c["variant"] == "clean" for c in sub):
        r1 = h1(sub, rng, n_boot)
        r1["supported"] = bool(r1["D"]["ci95"][0] > 0)
        out["H1"] = r1
    else:
        out["H1"] = {"not_estimable": "no kmer/positional or no clean cells in the subset"}
    r2 = h2_stat(sub, rng, n_boot) if s1 else None
    r3 = h3_stat(sub, rng, n_boot) if both else None
    if r3 is not None and not r3["n_pairs"]:
        r3 = None
    if r2 and r3:
        hm = holm({"H2": r2["p_one_sided"], "H3": r3["p_one_sided"]}, alpha)
        r2["holm"], r3["holm"] = hm["H2"], hm["H3"]
        r2["supported"], r3["supported"] = hm["H2"]["reject"], hm["H3"]["reject"]
    elif r2:                    # H3 not estimable: H2 at alpha / 2 (the preregistered family of two, Bonferroni)
        r2["holm"] = {"threshold": alpha / 2, "reject": bool(r2["p_one_sided"] <= alpha / 2)}
        r2["supported"] = r2["holm"]["reject"]
    out["H2"] = r2 or {"not_estimable": "no stage-1 kmer/positional cells in the subset"}
    out["H3"] = r3 or {"not_estimable": "no kmer/positional model with both stages in the subset and submitted"}
    return out


def per_protocol(cells: list[dict], primary: dict, seed: int, n_boot: int, alpha: float) -> dict:
    """Amendment 11 (sensitivity analyses; the primary analysis stays intention-to-treat). PP1 drops cells whose
    session evaluated any layer on DNA, one-hot or another constructed input-layer array (class input-layer; rule 4 as
    written to the sessions); PP2 drops every cell with local forward computation (classes input-layer and
    synthetic-only). Own RNG streams ([seed, 11, 1], [seed, 11, 2]). Compliance is self-selected, so differences from
    the primary results are descriptive, not causal; where a hypothesis's decision differs, that is listed for the
    abstract."""
    counts = {st: dict(Counter(c["compliance"] for c in cells if c["stage"] == st)) for st in ("stage1", "stage2")}
    out = {"classes_by_stage": counts,
           "per_run": [{"schedule_idx": c["schedule_idx"], "run_id": c["run_id"], "stage": c["stage"],
                        "class": c["compliance"]} for c in cells]}
    for name, keep, k in (("PP1", ("none", "synthetic-only"), 1), ("PP2", ("none",), 2)):
        sub = [c for c in cells if c["compliance"] in keep]
        res = _h123(sub, np.random.default_rng([seed, 11, k]), n_boot, alpha)
        res["kept_classes"] = list(keep)
        res["excluded_cells"] = sorted(c["schedule_idx"] for c in cells if c["compliance"] not in keep)
        res["differs_from_primary"] = sorted(h for h in ("H1", "H2", "H3")
                                             if "not_estimable" in res[h] or res[h].get("supported") != primary[h]["supported"])
        out[name] = res
    out["conclusions_differ"] = sorted({f"{p}:{h}" for p in ("PP1", "PP2") for h in out[p]["differs_from_primary"]})
    return out


def analyze(root: Path, cfg: dict, batch: str) -> dict:
    acfg = cfg["analysis"]
    seed, n_boot, alpha = int(acfg["seed"]), int(acfg["n_boot"]), float(acfg["alpha_one_sided"])
    data = load_cells(root, cfg, batch)
    cells = data["cells"]
    rng = np.random.default_rng(seed)
    out = {"batch": batch, "seed": seed, "n_boot": n_boot, "alpha_one_sided": alpha, "git_hash": git_hash(),
           "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "non_results_excluded": data["non_results"],
           "missing_cells": data["missing_cells"], "collection": data["collection"], "selection": data["selection"]}

    # ---- H1 (primary) and its sensitivity analyses
    res_h1 = h1(cells, rng, n_boot)
    res_h1["supported"] = bool(res_h1["D"]["ci95"][0] > 0)
    out["H1"] = res_h1
    out["H1_sensitivity"] = {
        "disagreements_excluded": h1(cells, rng, n_boot, drop_disagree=True),
        "clean_gc_claims_excluded": h1(cells, rng, n_boot, drop_gc_clean=True),
        "both_excluded": h1(cells, rng, n_boot, drop_disagree=True, drop_gc_clean=True)}

    # ---- H2: unconditional fix rate at budget 0 (stage 1); H3: paired reduction in R(probe FPR), 0 minus 50
    h2 = h2_stat(cells, rng, n_boot)
    h3 = h3_stat(cells, rng, n_boot)

    hm = holm({"H2": h2["p_one_sided"], "H3": h3["p_one_sided"]}, alpha)
    h2["holm"], h3["holm"] = hm["H2"], hm["H3"]
    h2["supported"], h3["supported"] = hm["H2"]["reject"], hm["H3"]["reject"]
    out["H2"], out["H3"] = h2, h3

    # ---- Amendment 1: pilot cells excluded (sensitivity analysis, reported alongside the primary results)
    out["pilot_excluded_sensitivity"] = pilot_excluded(cells, data["pilot_cells"], seed, n_boot, alpha)

    # ---- Amendment 2: launch method (exploratory)
    out["launch_method_sensitivity"] = launch_sensitivity(cells, seed, n_boot, alpha)

    # ---- Amendment 3: local evaluations next to forward-pass calls (exploratory)
    out["local_evaluation"] = local_evaluation_table(cells)
    out["forward_check_sha256"] = cells[0]["local"]["checker_sha256"] if cells else None
    out["path_review_sha256"] = data.get("path_review_sha256")         # Amendment 4

    # ---- Amendment 11: per-protocol sensitivity analyses (primary stays intention-to-treat)
    out["per_protocol"] = per_protocol(cells, {"H1": res_h1, "H2": h2, "H3": h3}, seed, n_boot, alpha)
    out["compliance_lock"] = data["compliance_lock"]

    # ---- H4 (descriptive)
    clean = [c for c in cells if c["variant"] == "clean"]
    out["H4"] = {"false_positive_rate": res_h1["false_positive_rate"], "n": len(clean),
                 "gc_claims_agreed": sum(bool(c["det"]["gc_claim_on_clean"]) for c in clean),
                 "gc_claims_either_coder": sum(bool(c["det"].get("gc_claim_on_clean_either")) for c in clean),
                 "disagreements": sum(not c["det"]["coders_agree"] for c in clean)}

    # ---- exploratory
    out["exploratory"] = {
        "fix_rate_given_correct_detection": {st: _cond_fix(cells, st) for st in ("stage1", "stage2")},
        "gc_shift": {"detection": _rate([c["det"]["correct"] for c in cells if c["variant"] == "gc_shift"]),
                     "fix_stage1": _rate([fixed_of(c) for c in cells if c["variant"] == "gc_shift" and c["stage"] == "stage1"]),
                     "fix_stage2": _rate([fixed_of(c) for c in cells if c["variant"] == "gc_shift" and c["stage"] == "stage2"])},
        "test_retest": _test_retest(cells), "calibration": _calibration(cells), "harm": _harm(cells)}

    # ---- §8.2 tables
    out["tables"] = {"confusion": _confusion(cells), "repair": _repair_table(cells),
                     "baselines": _baseline_table(root, cells), "kappa": _kappa(cells)}
    d = analysis_dir(root, batch)
    write_json_atomic(d / "analysis.json", out)
    (d / "analysis.md").write_text(report_md(out))
    return out


def _rate(xs):
    xs = [int(bool(x)) for x in xs]
    return {"k": sum(xs), "n": len(xs), "rate": (sum(xs) / len(xs)) if xs else None}


def _cond_fix(cells, stage):
    cs = [c for c in cells if c["stage"] == stage and c["variant"] in PRIMARY_PAIR + ("gc_shift",)]
    out = {}
    for v in PRIMARY_PAIR + ("gc_shift",):
        sub = [c for c in cs if c["variant"] == v]
        out[v] = {"given_correct": _rate([fixed_of(c) for c in sub if c["det"]["correct"]]),
                  "given_not_correct": _rate([fixed_of(c) for c in sub if not c["det"]["correct"]])}
    return out


def _test_retest(cells):
    by_t = defaultdict(dict)
    for c in cells:
        by_t[c["target"]][c["stage"]] = c["det"]["category"] or "disagree"
    pairs = [(d["stage1"], d["stage2"]) for d in by_t.values() if "stage1" in d and "stage2" in d]
    return {"n_models": len(pairs), "same_category": sum(a == b for a, b in pairs),
            "kappa": cohen_kappa([a for a, _ in pairs], [b for _, b in pairs]) if pairs else None}


def _calibration(cells):
    conf = np.array([float(c["det"]["confidence"]) for c in cells])
    ok = np.array([float(c["det"]["verdict_correct"]) for c in cells])
    bins = np.minimum((conf * 5).astype(int), 4)
    table = [{"bin": f"{b / 5:.1f}-{(b + 1) / 5:.1f}", "n": int((bins == b).sum()),
              "mean_confidence": float(conf[bins == b].mean()) if (bins == b).any() else None,
              "accuracy": float(ok[bins == b].mean()) if (bins == b).any() else None} for b in range(5)]
    return {"brier": float(np.mean((conf - ok) ** 2)) if len(conf) else None, "n": len(conf), "bins": table}


def _harm(cells):
    rows = []
    for c in cells:
        if c["variant"] != "clean":
            continue
        r = c["rep"]
        rows.append({"target": c["target"], "stage": c["stage"], "submitted": r["submitted"],
                     "unchanged": r.get("unchanged"), "frac_weights_changed": r.get("frac_weights_changed"),
                     "auprc_drop": (r.get("harm") or {}).get("auprc_drop"), "auroc_drop": (r.get("harm") or {}).get("auroc_drop"),
                     "phase_a_false_positive": c["det"]["false_positive"]})
    return rows


def _confusion(cells):
    out = {}
    for name, sub in (("stage1", [c for c in cells if c["stage"] == "stage1"]),
                      ("stage2", [c for c in cells if c["stage"] == "stage2"]), ("pooled", cells)):
        cm = Counter((c["variant"], c["det"]["category"] or "disagree") for c in sub)
        out[name] = {f"{v}|{k}": n for (v, k), n in sorted(cm.items())}
    out["partial"] = sum(c["det"]["outcome"] == "partial" for c in cells)
    out["kmer_discrepancy"] = sum(bool(c["det"].get("kmer_discrepancy")) for c in cells)
    return out


def _kappa(cells):
    a = [c["det"]["category_by_coder"]["A"] for c in cells]
    b = [c["det"]["category_by_coder"]["B"] for c in cells]
    return {"kappa_valid_runs": cohen_kappa(a, b) if a else None, "n": len(a)}


def _repair_table(cells):
    rows = []
    for v in PRIMARY_PAIR + ("gc_shift",):
        for st, budget in (("stage1", 0), ("stage2", 50)):
            sub = [c for c in cells if c["variant"] == v and c["stage"] == st]
            ver = [c["rep"]["verdict"] for c in sub if submitted(c)]
            mean = lambda xs: float(np.mean(xs)) if xs else None  # noqa: E731
            rows.append({"variant": v, "budget_b": budget, "n": len(sub), "submitted": len(ver),
                         "fixed": sum(fixed_of(c) for c in sub), "fixed_strict": sum(bool(x["fixed_strict"]) for x in ver),
                         "mean_R_probe": mean([x["probe_fpr"]["residual"] for x in ver if x["probe_fpr"]["residual"] is not None]),
                         "mean_R_reliance": mean([x["reliance"]["residual"] for x in ver if x["reliance"]["residual"] is not None]),
                         "mean_auprc_drop": mean([x["auprc_drop"] for x in ver]),
                         "mean_frac_weights_changed": mean([c["rep"]["frac_weights_changed"] for c in sub if c["rep"]["submitted"]]),
                         "unchanged": sum(bool(c["rep"].get("unchanged")) for c in sub)})
    return rows


def _baseline_table(root, cells):
    """Fixed / total per variant for each M4 baseline and budget (runs/baselines/*.json, v3), next to WSP."""
    bdir = root / "runs" / "baselines"
    agg = defaultdict(lambda: [0, 0])
    for f in sorted(bdir.glob("*.json")) if bdir.exists() else []:
        variant = f.stem.split("_p")[0]
        for b in json.loads(f.read_text())["baselines"]:
            k = (variant, b["baseline"], b["budget"])
            agg[k][0] += int(bool(b["verdict"]["fixed"]))
            agg[k][1] += 1
    rows = [{"variant": v, "method": m, "budget": bud, "fixed": k, "total": n} for (v, m, bud), (k, n) in sorted(agg.items(), key=str)]
    for v in PRIMARY_PAIR + ("gc_shift",):
        for st, bud in (("stage1", 0), ("stage2", 50)):
            sub = [c for c in cells if c["variant"] == v and c["stage"] == st]
            rows.append({"variant": v, "method": "WSP", "budget": bud, "fixed": sum(fixed_of(c) for c in sub), "total": len(sub)})
    return {"rows": rows, "note": "baselines saw labelled data (interp: contaminated chr16); WSP sessions saw no sequences"}


def report_md(o: dict) -> str:
    f = lambda s: f"{s['estimate']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}], p = {s['p_one_sided']:.4f}"  # noqa: E731
    L = [f"# Preregistered analysis, batch {o['batch']}", "",
         f"seed {o['seed']}, {o['n_boot']} bootstrap resamples, one-sided alpha {o['alpha_one_sided']}, git {o['git_hash']}",
         f"Non-results excluded (dry runs / manual runs): {o['non_results_excluded']}",
         f"Missing cells: {o['missing_cells']}; collection: {o['collection'].get('reason')}", ""]
    pp = o.get("per_protocol")
    if pp and pp["conclusions_differ"]:
        L += [f"**For the abstract (Amendment 11): the per-protocol analyses lead to a different conclusion than the "
              f"primary (intention-to-treat) analysis for {', '.join(pp['conclusions_differ'])}. Compliance is "
              f"self-selected; the difference is descriptive, not causal.**", ""]
    L += ["## Decision rules (§8.1)",
         f"- **H1** D = {f(o['H1']['D'])} (detection {o['H1']['detection_rate']['estimate']:.3f}, n = {o['H1']['n_planted']}; "
         f"false positives {o['H1']['false_positive_rate']['estimate']:.3f}, n = {o['H1']['n_clean']}) -> "
         f"**{'supported' if o['H1']['supported'] else 'not supported'}**"]
    for k in ("H2", "H3"):
        h = o[k]
        L.append(f"- **{k}** {f(h)}; Holm threshold {h['holm']['threshold']:.4f} -> "
                 f"**{'supported' if h['supported'] else 'not supported'}**")
    L.append(f"- **H4** false-positive rate {f(o['H4']['false_positive_rate'])}; GC claims (agreed / either coder) "
             f"{o['H4']['gc_claims_agreed']} / {o['H4']['gc_claims_either_coder']} of {o['H4']['n']}")
    pe = o["pilot_excluded_sensitivity"]
    L += ["", f"## Pilot cells excluded (Amendment 1 sensitivity analysis; cells {pe['excluded_cells']}, "
              f"{pe['n_cells']} cells analysed)",
          f"- **H1** D = {f(pe['H1']['D'])} (n = {pe['H1']['n_planted']} planted, {pe['H1']['n_clean']} clean) -> "
          f"**{'supported' if pe['H1']['supported'] else 'not supported'}**"]
    for k in ("H2", "H3"):
        L.append(f"- **{k}** {f(pe[k])}; Holm threshold {pe[k]['holm']['threshold']:.4f} -> "
                 f"**{'supported' if pe[k]['supported'] else 'not supported'}**")
    ls = o["launch_method_sensitivity"]
    L += ["", "## Launch method (Amendment 2; exploratory, no confirmatory claim)", "",
          "| launch | stage-1 cells | detection (planted) | false positives (clean) | fixed (planted) | median min "
          "| median tool calls | median fp calls |", "|---|---|---|---|---|---|---|---|"]
    r = lambda x: f"{x['k']}/{x['n']}"  # noqa: E731
    for m, v in ls["by_method_stage1"].items():
        L.append(f"| {m} | {v['n_stage1']} | {r(v['detection_correct_planted'])} | {r(v['false_positive_clean'])} | "
                 f"{r(v['fixed_planted'])} | {v['median_minutes']} | {v['median_tool_calls']} | {v['median_fp_calls']} |")
    bo = ls["batch_only"]
    L.append(f"\nBatch-launched cells only ({bo['n_cells']} cells):")
    for k in ("H1", "H2", "H3"):
        h = bo[k]
        if "not_estimable" in h:
            L.append(f"- **{k}** not estimable: {h['not_estimable']}")
        else:
            est = h["D"] if k == "H1" else h
            L.append(f"- **{k}** {f(est)} -> **{'supported' if h.get('supported') else 'not supported'}**")
    le = o["local_evaluation"]
    L += ["", "## Local evaluations next to forward-pass calls (Amendment 3; exploratory)", "",
          "| stage | cells | with local forward computation | median fp calls | median local evaluations (known part) "
          "| incomplete counts |", "|---|---|---|---|---|---|"]
    for st in ("stage1", "stage2"):
        v = le[st]
        L.append(f"| {st} | {v['n']} | {v['with_local_forward']} | {v['median_fp_calls']} | "
                 f"{v['median_local_evaluations_known_part']} | {v['incomplete_counts']} |")
    L += ["", f"Per run: analysis.json, local_evaluation.per_run (forward_check sha256 {o['forward_check_sha256']}; "
          f"path_review sha256 {o.get('path_review_sha256')}, Amendment 4)."]
    if pp:
        L += ["", "## Per-protocol sensitivity analyses (Amendment 11; primary analysis is intention-to-treat)", "",
              "Compliance classes (locked before unblinding, from forward_check and the review packet): "
              + "; ".join(f"{st}: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items()))
                          for st, cnt in pp["classes_by_stage"].items()) + ".",
              "Compliance is self-selected: differences from the primary results are descriptive, not causal."]
        for name, label in (("PP1", "PP1: input-layer evaluation (rule 4 as written) excluded"),
                            ("PP2", "PP2: any local forward computation excluded")):
            v = pp[name]
            L.append(f"\n{label} ({v['n_cells']} cells; excluded cells {v['excluded_cells']}):")
            for k in ("H1", "H2", "H3"):
                h = v[k]
                if "not_estimable" in h:
                    L.append(f"- **{k}** not estimable: {h['not_estimable']}")
                else:
                    est = h["D"] if k == "H1" else h
                    L.append(f"- **{k}** {f(est)} -> **{'supported' if h.get('supported') else 'not supported'}**")
        L.append(f"\nDiffers from the primary decision: {', '.join(pp['conclusions_differ']) or 'none'}.")
    L += ["", "## H1 sensitivity analyses (exploratory)"]
    for k, v in o["H1_sensitivity"].items():
        L.append(f"- {k}: D = {f(v['D'])} (n = {v['n_planted']} planted, {v['n_clean']} clean)")
    L += ["", "Full tables: analysis.json (confusion, repair, baselines, kappa, calibration, test-retest, harm)."]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- design analysis (§8.4)
DESIGN_PATTERNS = [  # name, per-instance success rate for (kmer_s0, kmer_s1, kmer_s2, positional)
    ("all 4 instances", (1, 1, 1, 1)),
    ("3 of 4: the 3 k-mers (not positional)", (1, 1, 1, 0)),
    ("3 of 4: 2 k-mers + positional", (1, 1, 0, 1)),
    ("2 of 4: 2 k-mers", (1, 1, 0, 0)),
    ("2 of 4: 1 k-mer + positional", (1, 0, 0, 1)),
    ("1 of 4: positional only", (0, 0, 0, 1)),
    ("1 of 4: one k-mer only", (1, 0, 0, 0)),
    ("all 4 at 75% (H1 only)", (0.75, 0.75, 0.75, 0.75)),
    ("all 4 at 50%", (0.5, 0.5, 0.5, 0.5)),
    ("none", (0, 0, 0, 0)),
]
DESIGN_MODELS = {"kmer_s0": 2, "kmer_s1": 2, "kmer_s2": 2, "positional": 6}


def _design_cells(rates: tuple, fp_answers: int) -> list[dict]:
    """Synthetic cells with the real structure: per instance its number of models, two budget-0 phase-A answers per
    model (stages 1 and 2), one stage-1 edit per model; 10 clean models x 2 answers, fp_answers false positives in
    different models. Within an instance, successes go to answers in session-major order (spread over models)."""
    cells = []
    for (inst, n_models), rate in zip(DESIGN_MODELS.items(), rates):
        answers = [(m, st) for st in ("stage1", "stage2") for m in range(n_models)]
        k_det = round(rate * len(answers))
        k_fix = int(rate * n_models)                       # floor: a 2-model k-mer instance at 50% fixes 1 model
        for i, (m, st) in enumerate(answers):
            cells.append({"run_id": f"{inst}-{m}-{st}", "target": f"{inst}-m{m}", "instance": inst,
                          "variant": "positional" if inst == "positional" else "kmer", "stage": st,
                          "det": {"correct": i < k_det, "false_positive": False, "coders_agree": True},
                          "rep": {"submitted": True, "verdict": {"fixed": st == "stage1" and m < k_fix,
                                                                 "probe_fpr": {"residual": 0.0}}}})
    for m in range(10):
        for j, st in enumerate(("stage1", "stage2")):
            fp = j == 0 and m < fp_answers
            cells.append({"run_id": f"clean-{m}-{st}", "target": f"clean_s{m}", "instance": f"clean_s{m}",
                          "variant": "clean", "stage": st,
                          "det": {"correct": not fp, "false_positive": fp, "coders_agree": True,
                                  "gc_claim_on_clean_either": False}, "rep": {"submitted": True}})
    return cells


def design_table(seed: int, n_boot: int, fp_answers: int = 2, alpha: float = 0.025) -> list[dict]:
    rows = []
    for i, (name, rates) in enumerate(DESIGN_PATTERNS):
        rng = np.random.default_rng([seed, i])
        cells = _design_cells(rates, fp_answers)
        r1 = h1(cells, rng, n_boot)
        row = {"pattern": name, "rates": rates, "detection_rate": r1["detection_rate"]["estimate"],
               "D": r1["D"]["estimate"], "D_lower": r1["D"]["ci95"][0], "H1_pass": r1["D"]["ci95"][0] > 0}
        if "H1 only" not in name:
            r2 = h2_stat(cells, rng, n_boot)
            row.update(fix_rate=r2["estimate"], H2_p=r2["p_one_sided"],
                       H2_pass_alone=r2["p_one_sided"] <= alpha / 2, H2_pass_if_H3_first=r2["p_one_sided"] <= alpha)
        rows.append(row)
    return rows


def design_md(rows: list[dict], fp_answers: int = 2) -> str:
    yn = lambda b: "yes" if b else "no"  # noqa: E731
    L = [f"Design analysis: `python -m wspg.analysis design` (analysis seed, false-positive rate "
         f"{fp_answers}/20 = {fp_answers / 20:.0%} on clean models).", "",
         "| success pattern (per plant instance) | H1: detection rate | D | lower 95% | H1 passes | "
         "H2: fix rate | H2 p | H2 passes (p ≤ 0.0125) | H2 passes if H3 rejects first (p ≤ 0.025) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        h2 = (f"{r['fix_rate']:.2f} | {r['H2_p']:.4f} | {yn(r['H2_pass_alone'])} | {yn(r['H2_pass_if_H3_first'])}"
              if "fix_rate" in r else "– | – | – | –")
        L.append(f"| {r['pattern']} | {r['detection_rate']:.2f} | {r['D']:.2f} | {r['D_lower']:.2f} | "
                 f"{yn(r['H1_pass'])} | {h2} |")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- CLI
def decide(root: Path, run_id: str, decision: str, reason: str) -> None:
    rp = RunPaths(run_id, root)
    if any((root / "runs" / "wsp" / "coding").glob("*/detection.json")):
        raise SystemExit("a batch has been scored (unblinded): needs-review decisions must be made before unblinding")
    status = audit_status(rp)                                    # Amendment 4
    if status != "needs-review":
        raise SystemExit(f"{run_id} is {status!r}, not needs-review")
    write_json_atomic(rp.dir / "review_decision.json", {"decision": decision, "reason": reason,
                                                         "time": time.strftime("%Y-%m-%dT%H:%M:%S")})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["decide", "compliance", "compliance-lock", "repair", "analyze", "design"])
    ap.add_argument("--batch")
    ap.add_argument("--run")
    ap.add_argument("--decision", choices=["valid", "invalid"])
    ap.add_argument("--reason")
    ap.add_argument("--class", dest="cls", choices=list(COMPLIANCE_CLASSES), help="compliance: override the proposal")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "harness.yaml"))
    ap.add_argument("--seed", type=int, default=None, help="unused: the analysis seed is fixed in the config (§8)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.cmd == "decide":
        if not (a.run and a.decision and a.reason):
            ap.error("decide needs --run, --decision and --reason")
        decide(REPO_ROOT, a.run, a.decision, a.reason)
        print(f"{a.run}: needs-review decided {a.decision}")
        return
    if a.cmd == "compliance":
        if not a.run:
            ap.error("compliance needs --run (and --class / --reason to override the forward_check proposal)")
        rec = record_compliance(REPO_ROOT, a.run, a.cls, a.reason)
        print(f"{a.run}: compliance {rec['class']} (proposed {rec['proposed']}; {rec['proposal_basis']})")
        return
    if a.cmd == "compliance-lock":
        lock = lock_compliance(REPO_ROOT)
        print(f"compliance locked: {lock['n_runs']} runs, {lock['counts']}")
        return
    if a.cmd == "design":
        acfg = cfg["analysis"]
        print(design_md(design_table(int(acfg["seed"]), int(acfg["n_boot"]), alpha=float(acfg["alpha_one_sided"]))))
        return
    if not a.batch:
        ap.error("--batch is required")
    if a.cmd == "repair":
        ecfg = load_config(REPO_ROOT / "configs" / "eval.yaml")
        print(evaluate_repairs(REPO_ROOT, cfg, a.batch, n_boot=int(ecfg["n_boot"])))
    else:
        o = analyze(REPO_ROOT, cfg, a.batch)
        print(report_md(o))


if __name__ == "__main__":
    main()
