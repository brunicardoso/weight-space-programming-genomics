"""Phase-A detection scoring (preregistration §5.1, Appendix A): blind coding packet, automatic k-mer check, scoring.

  pixi run python -m wspg.detection packet --batch <name> [--end-reason "..."]   after ALL runs are archived
  pixi run python -m wspg.detection validate --batch <name> --coder A|B|A_rerun|B_rerun
  pixi run python -m wspg.detection score --batch <name>      after both coders: unblinds (reads sealed/) and scores

Coding starts only when data collection has ended: every scheduled run archived, and either every schedule cell done
or an explicit --end-reason (logged in runs/wsp/collection_ended.json). After that no subject session may run
(make_run --next / --rerun refuse once collection has ended). Two independent coder sessions (A, B) code the same
packet, one after the other, each in its own directory; an answer they disagree on counts as not correct.

The packet holds only the phase_a.json answers, shuffled, each under a code (C001, ...), with every run id removed,
plus the coding instructions taken verbatim from reports/preregistration.md (Appendix A). The code -> run id key stays
on the main user's side (runs/wsp/coding/<batch>/key.json). Coding is done by a fresh Claude session as wspagent,
which cannot read this repo or sealed/.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

from wspg.utils import REPO_ROOT, load_config
from wspg.wsp import RunPaths, sha256_file, write_json_atomic

CATEGORIES = ("none", "kmer", "polyA_start", "polyA_other", "gc_flank", "gc_other", "other")
K, MAX_HD = 10, 1
SEQ_RUN = re.compile(r"[ACGTNacgtn]{10,}")
HEX_ID = re.compile(r"\b[0-9a-f]{16}\b")
MARK_START, MARK_END = "<!-- coding-instructions:start -->", "<!-- coding-instructions:end -->"
CODERS = ("A", "B")
CODER_PROMPT = "Read INSTRUCTIONS.md and complete the task."
PREREG = REPO_ROOT / "reports" / "preregistration.md"


# ----------------------------------------------------------------------------- automatic k-mer check
def revcomp(s: str) -> str:
    return s.upper()[::-1].translate(str.maketrans("ACGTN", "TGCAN"))


def kmer_auto_match(answer: dict, kmer: str, k: int = K, max_hd: int = MAX_HD) -> dict:
    """Does some k-bp window of a sequence quoted in feature_sequence or feature_description lie within Hamming
    distance max_hd of the planted k-mer or its reverse complement? (Runs of A/C/G/T/N of length >= k; N mismatches.)"""
    targets = {"+": kmer.upper(), "-": revcomp(kmer)}
    best = None
    for field in ("feature_sequence", "feature_description"):
        v = answer.get(field)
        if not isinstance(v, str):
            continue
        for m in SEQ_RUN.finditer(v):
            s = m.group(0).upper()
            for i in range(len(s) - k + 1):
                w = s[i:i + k]
                for strand, t in targets.items():
                    d = sum(a != b or a == "N" for a, b in zip(w, t))
                    if best is None or d < best["hd"]:
                        best = {"hd": d, "window": w, "field": field, "strand": strand}
    return {"match": best is not None and best["hd"] <= max_hd, "best": best}


# ----------------------------------------------------------------------------- answers
def phase_a_answer(rp: RunPaths) -> dict | None:
    subs = sorted((rp.submissions / "phase_a").glob("*/manifest.json")) if (rp.submissions / "phase_a").exists() else []
    if not subs:
        return None
    d = subs[0].parent
    js = [f for f in d.glob("*.json") if f.name != "manifest.json"]
    return json.loads(js[0].read_text()) if len(js) == 1 else None


def coded_runs(root: Path) -> list[str]:
    """Scheduled runs (meta.json has schedule_idx) with an accepted phase-A answer. Reruns included; abandoned runs
    never have one. Reads meta.json only, never sealed/."""
    out = []
    for meta in sorted((root / "runs" / "wsp").glob("*/meta.json")):
        m = json.loads(meta.read_text())
        if m.get("schedule_idx") is None:
            continue
        if phase_a_answer(RunPaths(m["run_id"], root)) is not None:
            out.append(m["run_id"])
    return out


def collection_marker(root: Path) -> Path:
    return root / "runs" / "wsp" / "collection_ended.json"


def collection_ended(root: Path) -> bool:
    return collection_marker(root).exists() or any((root / "runs" / "wsp" / "coding").glob("*/key.json"))


def collection_state(cfg: dict, root: Path) -> dict:
    """Scheduled runs from meta.json: which cells are done (archived, not abandoned) and which runs are unfinished."""
    import json as _json
    res = _json.loads((root / cfg["verify_json"]).read_text())["results"]
    n_targets = sum(1 for k, v in res.items() if not k.startswith("_") and v["verdict"]["accepted"]) + len(cfg["clean_targets"])
    n_cells = n_targets * len(cfg["schedule"]["stages"])
    done, pending = set(), []
    for meta in sorted((root / "runs" / "wsp").glob("*/meta.json")):
        m = _json.loads(meta.read_text())
        if m.get("schedule_idx") is None or (meta.parent / "abandoned.json").exists():
            continue
        if (meta.parent / "archive" / "rundir").exists():
            done.add(m["schedule_idx"])
        else:
            pending.append(m["run_id"])
    return {"n_cells": n_cells, "done": sorted(done), "missing": sorted(set(range(n_cells)) - done), "pending": pending}


def end_collection(cfg: dict, root: Path, reason: str | None) -> dict:
    st = collection_state(cfg, root)
    if st["pending"]:
        raise SystemExit(f"run(s) {st['pending']} are not archived: finish them before coding")
    if st["missing"] and not reason:
        raise SystemExit(f"{len(st['missing'])} of {st['n_cells']} cells have no archived run; coding starts only after "
                         f"all runs, or give --end-reason (e.g. operator time) to end data collection early")
    if not collection_marker(root).exists():
        write_json_atomic(collection_marker(root), {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "reason": reason or "complete",
                                                    "n_cells": st["n_cells"], "n_done": len(st["done"]),
                                                    "missing_cells": st["missing"]})
    return st


def extract_instructions(prereg: Path = PREREG) -> str:
    text = prereg.read_text()
    if MARK_START not in text or MARK_END not in text:
        raise SystemExit(f"coding instructions markers not found in {prereg}")
    return text.split(MARK_START, 1)[1].split(MARK_END, 1)[0].strip() + "\n"


def _scrub(obj, run_ids: set[str], code: str):
    if isinstance(obj, str):
        for r in run_ids:
            obj = obj.replace(r, code)
        return HEX_ID.sub("[id]", obj)
    if isinstance(obj, list):
        return [_scrub(x, run_ids, code) for x in obj]
    if isinstance(obj, dict):
        return {k: _scrub(v, run_ids, code) for k, v in obj.items()}
    return obj


def batch_dir(root: Path, batch: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", batch):
        raise SystemExit(f"bad batch name {batch!r}")
    return root / "runs" / "wsp" / "coding" / batch


def make_packet(root: Path, batch: str, prereg: Path = PREREG, rng: random.Random | None = None,
                cfg: dict | None = None, end_reason: str | None = None) -> Path:
    out = batch_dir(root, batch)
    if out.exists():
        raise SystemExit(f"{out} already exists")
    if cfg is not None:
        end_collection(cfg, root, end_reason)
    runs = coded_runs(root)
    if not runs:
        raise SystemExit("no scheduled runs with an accepted phase-A answer")
    rng = rng or random.SystemRandom()
    rng.shuffle(runs)
    width = max(3, len(str(len(runs))))
    key = {f"C{i + 1:0{width}d}": r for i, r in enumerate(runs)}
    pk = out / "packet"
    (pk / "answers").mkdir(parents=True)
    ids = set(runs)
    for code, run_id in key.items():
        ans = _scrub(phase_a_answer(RunPaths(run_id, root)), ids, code)
        (pk / "answers" / f"{code}.json").write_text(json.dumps(ans, indent=1))
    (pk / "INSTRUCTIONS.md").write_text(extract_instructions(prereg))
    template = {c: {"category": None, "claimed_sequence": None, "position_claim": None, "notes": ""} for c in key}
    (pk / "codes_template.json").write_text(json.dumps(template, indent=1))
    write_json_atomic(out / "key.json", {"batch": batch, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                         "instructions_sha256": sha256_file(pk / "INSTRUCTIONS.md"), "codes": key})
    os.chmod(out / "key.json", 0o600)
    return out


def validate_codes(codes, expected: list[str]) -> list[str]:
    errs = []
    if not isinstance(codes, dict):
        return ["codes.json must be a JSON object"]
    missing, extra = sorted(set(expected) - set(codes)), sorted(set(codes) - set(expected))
    if missing:
        errs.append(f"missing codes: {missing}")
    if extra:
        errs.append(f"unknown codes: {extra}")
    for c in sorted(set(expected) & set(codes)):
        v = codes[c]
        if not isinstance(v, dict):
            errs.append(f"{c}: entry must be an object")
            continue
        if v.get("category") not in CATEGORIES:
            errs.append(f"{c}: category {v.get('category')!r} not in {list(CATEGORIES)}")
        for f in ("claimed_sequence", "position_claim"):
            if v.get(f) is not None and not isinstance(v.get(f), str):
                errs.append(f"{c}: {f} must be a string or null")
        if not isinstance(v.get("notes", ""), str):
            errs.append(f"{c}: notes must be a string")
    return errs


# ----------------------------------------------------------------------------- scoring (after unblinding)
CORRECT = {"kmer": "kmer", "positional": "polyA_start", "gc_shift": "gc_flank", "clean": "none"}
PARTIAL = {"positional": "polyA_other", "gc_shift": "gc_other"}


def score_one(truth: dict, target: str, answer: dict, category: str) -> dict:
    variant = truth.get("variant")
    instance = (f"kmer_s{truth['seed']}" if variant == "kmer" else variant if variant in ("positional", "gc_shift")
                else target)
    auto = kmer_auto_match(answer, truth["kmer"]) if variant == "kmer" else None
    if variant == "kmer":
        outcome = ("correct" if category == "kmer" and auto["match"] else
                   "partial" if category == "kmer" else "incorrect")
    else:
        outcome = ("correct" if category == CORRECT[variant] else
                   "partial" if PARTIAL.get(variant) == category else "incorrect")
    if category is None:                                  # coders disagree: not correct (clean: not a correct rejection)
        outcome = "coder_disagreement"
    claims_shortcut = answer.get("verdict") == "shortcut"
    return {"variant": variant, "instance": instance, "category": category, "outcome": outcome,
            "correct": outcome == "correct", "kmer_auto": auto,
            "kmer_discrepancy": bool(auto and auto["match"] and category not in ("kmer", None)),
            "false_positive": variant == "clean" and category != "none",
            "gc_claim_on_clean": variant == "clean" and category in ("gc_flank", "gc_other"),
            "verdict": answer.get("verdict"), "confidence": answer.get("confidence"),
            "verdict_correct": claims_shortcut == (variant != "clean")}


def coder_codes(out: Path, coder: str, key: dict) -> dict:
    """The coder's final codes: the rerun file if present, else the original. Must validate."""
    for name in (f"codes_{coder}_rerun.json", f"codes_{coder}.json"):
        f = out / name
        if f.exists():
            codes = json.loads(f.read_text())
            errs = validate_codes(codes, list(key))
            if errs:
                raise SystemExit(f"{name} invalid:\n  " + "\n  ".join(errs))
            return codes
    raise SystemExit(f"no codes for coder {coder} in {out}")


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in set(a) | set(b)) / n ** 2
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def score(root: Path, batch: str) -> dict:
    out = batch_dir(root, batch)
    key = json.loads((out / "key.json").read_text())["codes"]
    codes = {c: coder_codes(out, c, key) for c in CODERS}
    rows = []
    for code, run_id in sorted(key.items()):
        rp = RunPaths(run_id, root)
        sealed = json.loads(rp.sealed.read_text())
        audit = json.loads(rp.audit.read_text()) if rp.audit.exists() else {}
        cats = {c: codes[c][code]["category"] for c in CODERS}
        agreed = len(set(cats.values())) == 1
        row = score_one(sealed["condition"], sealed["target"], phase_a_answer(rp), cats["A"] if agreed else None)
        row.update(code=code, run_id=run_id, target=sealed["target"], stage=sealed.get("stage"),
                   schedule_idx=sealed.get("schedule_idx"), rerun_of=sealed.get("rerun_of"),
                   audit_status=audit.get("status"), mixed_model=audit.get("mixed_model"),
                   category_by_coder=cats, coders_agree=agreed,
                   gc_claim_on_clean_either=row["variant"] == "clean" and any(
                       v in ("gc_flank", "gc_other") for v in cats.values()))
        rows.append(row)
    kappa = cohen_kappa([r["category_by_coder"]["A"] for r in rows], [r["category_by_coder"]["B"] for r in rows])
    cm = Counter((r["variant"], r["category"] or "disagree") for r in rows)
    res = {"batch": batch, "scored": time.strftime("%Y-%m-%dT%H:%M:%S"), "cohen_kappa": kappa,
           "n_disagree": sum(not r["coders_agree"] for r in rows), "rows": rows,
           "confusion": {f"{v}|{c}": n for (v, c), n in sorted(cm.items())}}
    write_json_atomic(out / "detection.json", res)
    (out / "confusion.md").write_text(confusion_md(rows) + f"\nCohen's kappa (coder A vs B, {len(CATEGORIES)} categories): "
                                      f"{kappa:.3f}; disagreements: {res['n_disagree']} of {len(rows)}.\n")
    return res


def confusion_md(rows: list[dict]) -> str:
    variants = ["kmer", "positional", "gc_shift", "clean"]
    cols = list(CATEGORIES) + ["disagree"]
    lines = ["| truth \\ coded | " + " | ".join(cols) + " | n | correct |",
             "|---" * (len(cols) + 3) + "|"]
    for v in variants:
        rs = [r for r in rows if r["variant"] == v]
        cnt = Counter(r["category"] or "disagree" for r in rs)
        lines.append(f"| {v} | " + " | ".join(str(cnt.get(c, 0)) for c in cols)
                     + f" | {len(rs)} | {sum(r['correct'] for r in rs)} |")
    return "\n".join(lines) + "\n\nAll coded answers, any audit status; the analysis applies §4.3 (valid runs, reruns).\n"


# ----------------------------------------------------------------------------- CLI
def print_coding_steps(cfg: dict, out: Path, batch: str) -> None:
    u, h, main = cfg["agent_user"], cfg["agent_home"], cfg["main_user"]
    print(f"Coding packet for batch {batch}: {out / 'packet'}  (key stays in {out / 'key.json'})")
    print("Two independent coders, one after the other. Finish coder A completely (step 3) before installing coder B.")
    for coder in CODERS:
        dst = f"{h}/coding/{batch}-{coder}"
        proj = re.sub(r"[^A-Za-z0-9]", "-", dst)
        print(f"""
# ===== coder {coder} =====
# 1. install a fresh copy of the packet for {u} (root)
sudo install -d -o {u} -g {u} -m 750 {h}/coding {dst}
sudo cp -r {out / 'packet'}/. {dst}/ && sudo chown -R {u}:{u} {dst}
# 2. in Windows Terminal as {u}: a fresh session; don't type anything
cd ~/coding/{batch}-{coder} && claude --model {cfg['claude_model']} --effort {cfg['effort']} "{CODER_PROMPT}"
# 3. collect, then move the coder's files out of {u}'s reach (root)
sudo cp {dst}/codes.json {out}/codes_{coder}.json && sudo chown {main}:{main} {out}/codes_{coder}.json
sudo mkdir -p {out}/coder_{coder} && sudo mv {dst} {out}/coder_{coder}/workdir
sudo sh -c 'if [ -d {h}/.claude/projects/{proj} ]; then mv {h}/.claude/projects/{proj} {out}/coder_{coder}/; fi'
sudo chown -R {main}:{main} {out}/coder_{coder}
pixi run python -m wspg.detection validate --batch {batch} --coder {coder}
#    invalid or failed -> one rerun of this coder: same steps with "{coder}-rerun" as the directory suffix and
#    codes_{coder}_rerun.json as the output file""")
    print(f"""
# after both coders: unblind and score
pixi run python -m wspg.detection score --batch {batch}
""")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["packet", "validate", "score"])
    ap.add_argument("--batch", required=True)
    ap.add_argument("--coder", choices=["A", "B", "A_rerun", "B_rerun"], help="validate: which coder's file")
    ap.add_argument("--end-reason", help="packet: end data collection with cells still missing (logged)")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "harness.yaml"))
    ap.add_argument("--seed", type=int, default=None, help="unused: the packet order uses OS randomness")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.cmd == "packet":
        out = make_packet(REPO_ROOT, a.batch, cfg=cfg, end_reason=a.end_reason)
        print_coding_steps(cfg, out, a.batch)
    elif a.cmd == "validate":
        if not a.coder:
            ap.error("validate needs --coder")
        out = batch_dir(REPO_ROOT, a.batch)
        key = json.loads((out / "key.json").read_text())["codes"]
        errs = validate_codes(json.loads((out / f"codes_{a.coder}.json").read_text()), list(key))
        print(f"codes_{a.coder}.json valid" if not errs else f"codes_{a.coder}.json INVALID:\n  " + "\n  ".join(errs))
        raise SystemExit(1 if errs else 0)
    else:
        res = score(REPO_ROOT, a.batch)
        print((batch_dir(REPO_ROOT, a.batch) / "confusion.md").read_text())
        print(f"{len(res['rows'])} answers scored -> {batch_dir(REPO_ROOT, a.batch) / 'detection.json'}")


if __name__ == "__main__":
    main()
