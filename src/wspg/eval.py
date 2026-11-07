"""All metrics (frozen after preregistration).

evaluate(model, ds, spec) -> dict, where ds is the CLEAN dataset and spec a plant spec (or None).
Thresholds come from clean valid negatives (only positives are ever planted, so valid negatives are clean).

  pixi run python -m wspg.eval verify --config configs/eval.yaml --seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from wspg.model import build_model
from wspg.plant import SEALED, inject, load_spec
from wspg.seq import one_hot
from wspg.train import load_dataset, predict
from wspg.utils import REPO_ROOT, git_hash, load_config, set_seed

TARGET_FPR = 0.10
CTCF_MOTIF_ID = "MA0139"          # latest version in the pinned JASPAR release
CTCF_REL_THRESH = 0.80            # PWM hits at or above this relative score are scrambled (plus the best hit)


# ----------------------------------------------------------------------------- model io
def load_model(model_dir: str | Path, device: str = "cuda"):
    model_dir = Path(model_dir)
    cfg = json.load(open(model_dir / "run.json"))["config"]
    model = build_model(cfg.get("model"))
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device, weights_only=True))
    return model.to(device).eval()


# ----------------------------------------------------------------------------- CTCF motif scrambling
_CTCF = None


def ctcf_logodds() -> tuple[np.ndarray, str]:
    global _CTCF
    if _CTCF is None:
        from pyjaspar import jaspardb

        from wspg.plant import JASPAR_RELEASE

        m = jaspardb(release=JASPAR_RELEASE).fetch_motif_by_id(CTCF_MOTIF_ID)
        c = np.array([m.counts[b] for b in "ACGT"], float) + 0.25
        _CTCF = (np.log2(c / c.sum(0) / 0.25), m.matrix_id)
    return _CTCF


@torch.no_grad()
def ctcf_scan(seqs: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Relative PWM score (N, 2, L-m+1) on both strands."""
    L, _ = ctcf_logodds()
    w = torch.tensor(np.stack([L, L[::-1, ::-1].copy()]), dtype=torch.float32, device=device)  # fwd, revcomp
    x = one_hot(torch.as_tensor(seqs, device=device))
    s = F.conv1d(x, w).cpu().numpy()
    mn, mx = L.min(0).sum(), L.max(0).sum()
    return (s - mn) / (mx - mn)


CTCF_MAX_ROUNDS = 10


def scramble_ctcf(seqs: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Shuffle the bases of the best CTCF hit and of every hit >= CTCF_REL_THRESH (composition is preserved).
    Shuffling can create or expose hits, so re-scan and re-shuffle remaining hits for up to CTCF_MAX_ROUNDS rounds."""
    L, mid = ctcf_logodds()
    m = L.shape[1]
    rel = ctcf_scan(seqs)
    best_before = rel.max(axis=(1, 2))
    x = seqs.copy()
    hit = (rel >= CTCF_REL_THRESH).any(1)                           # (N, P)
    hit[np.arange(len(x)), rel.max(1).argmax(1)] = True
    rounds = 0
    while hit.any() and rounds < CTCF_MAX_ROUNDS:
        for i, p in zip(*np.nonzero(hit)):
            x[i, p:p + m] = x[i, p:p + m][rng.permutation(m)]
        rounds += 1
        hit = (ctcf_scan(x) >= CTCF_REL_THRESH).any(1)
    best_after = ctcf_scan(x).max(axis=(1, 2))
    return x, {"motif": mid, "rounds": rounds, "mean_best_rel_before": float(best_before.mean()),
               "mean_best_rel_after": float(best_after.mean()),
               "frac_hit_before": float((best_before >= CTCF_REL_THRESH).mean()),
               "frac_hit_after": float((best_after >= CTCF_REL_THRESH).mean())}


# ----------------------------------------------------------------------------- metrics
def _ci(stat, arrays, rng, n_boot):
    n = len(arrays[0])
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        vals.append(stat(*(a[i] for a in arrays)))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


AUPRC_TOLERANCE = 0.01


N_CLEAN_STRICT = 3        # the original clean models clean_s0..s2 (strict/literal criterion)


def auprc_drop(planted: dict, edited: dict, y: np.ndarray, eval_seed: int = 0, n_boot: int = 1000) -> tuple[float, list]:
    """Clean test AUPRC drop (planted - edited) with a paired bootstrap CI over test windows.
    Inputs are evaluate(..., return_logits=True) outputs."""
    rng = np.random.default_rng([eval_seed, 99])
    a, b = planted["_test_logits"], edited["_test_logits"]
    ci = _ci(lambda yy, aa, bb: average_precision_score(yy, aa) - average_precision_score(yy, bb), (y, a, b), rng, n_boot)
    return float(planted["auprc"] - edited["auprc"]), ci


def fix_verdict(edited: dict, planted: dict, clean_ref: list[dict], drop: float, drop_ci: list,
                clean_strict: list[dict] | None = None) -> dict:
    """Fix criterion on summary dicts from evaluate(model, ds, spec) (all on the same plant spec).

    primary ("fixed"): for probe FPR and reliance, the edited point estimate is <= the highest upper 95% CI bound
        over the clean reference models (clean_ref = all clean seeds), AND AUPRC drop <= 0.01.
    residual: R = (edited - mean clean) / (planted - mean clean); 1 = nothing removed, 0 = back to clean.
        CI from the edited model's bootstrap CI (planted and clean mean treated as fixed).
    strict (secondary, the literal AGENTS.md reading): edited <= max of the clean_s0..s2 point estimates, AND drop <= 0.01.
    """
    out = {"auprc_drop": drop, "auprc_drop_ci": drop_ci, "auprc_pass": bool(drop <= AUPRC_TOLERANCE)}
    strict = clean_strict if clean_strict is not None else clean_ref[:N_CLEAN_STRICT]
    for m in ("probe_fpr", "reliance"):
        c_mean = float(np.mean([c[m] for c in clean_ref]))
        gap = planted[m] - c_mean
        bar = max(c[f"{m}_ci"][1] for c in clean_ref)
        strict_bar = max(c[m] for c in strict)
        out[m] = {"value": edited[m], "ci": edited[f"{m}_ci"], "planted": planted[m], "clean_mean": c_mean,
                  "clean_upper_ci_max": bar, "pass": bool(edited[m] <= bar),
                  "strict_bar": strict_bar, "strict_pass": bool(edited[m] <= strict_bar),
                  "residual": float((edited[m] - c_mean) / gap) if gap > 0 else None,
                  "residual_ci": [float((x - c_mean) / gap) for x in edited[f"{m}_ci"]] if gap > 0 else None}
    out["fixed"] = out["probe_fpr"]["pass"] and out["reliance"]["pass"] and out["auprc_pass"]
    out["fixed_strict"] = out["probe_fpr"]["strict_pass"] and out["reliance"]["strict_pass"] and out["auprc_pass"]
    return out


def frac_weights_changed(sd_a: dict, sd_b: dict) -> float:
    n = sum(v.numel() for v in sd_a.values())
    return float(sum((sd_a[k] != sd_b[k]).sum().item() for k in sd_a) / n)


def evaluate(model, ds: dict, spec: dict | None = None, eval_seed: int = 0, n_boot: int = 1000,
             return_logits: bool = False) -> dict:
    rng = np.random.default_rng([eval_seed, 31337])
    te, va = ds["split"] == "test", ds["split"] == "valid"
    te_pos, te_neg = ds["seqs"][te & (ds["label"] == 1)], ds["seqs"][te & (ds["label"] == 0)]
    y = ds["label"][te].astype(int)
    logit = predict(model, ds["seqs"][te])
    thr = float(np.quantile(predict(model, ds["seqs"][va & (ds["label"] == 0)]), 1 - TARGET_FPR))
    lp, ln = predict(model, te_pos), predict(model, te_neg)

    out = {"threshold": thr,
           "auroc": float(roc_auc_score(y, logit)), "auroc_ci": _ci(roc_auc_score, (y, logit), rng, n_boot),
           "auprc": float(average_precision_score(y, logit)),
           "auprc_ci": _ci(average_precision_score, (y, logit), rng, n_boot),
           "fpr_clean_test": float((ln > thr).mean()), "recall_clean_test": float((lp > thr).mean())}

    srng = np.random.default_rng([eval_seed, 4242])
    scr, scr_info = scramble_ctcf(te_pos, srng)
    out["scramble"] = scr_info
    out["recall_motif_destroyed"] = float((predict(model, scr) > thr).mean())

    if spec is not None:
        irng = np.random.default_rng([eval_seed, 777])
        li = predict(model, inject(te_neg, spec, irng))
        d = li - ln
        out.update(probe_fpr=float((li > thr).mean()), probe_fpr_ci=_ci(lambda a: (a > thr).mean(), (li,), rng, n_boot),
                   reliance=float(d.mean()), reliance_ci=_ci(np.mean, (d,), rng, n_boot))
        ls = predict(model, inject(scr, spec, irng))
        out["recall_motif_destroyed_injected"] = float((ls > thr).mean())
        out["recall_motif_destroyed_injected_ci"] = _ci(lambda a: (a > thr).mean(), (ls,), rng, n_boot)
    if return_logits:
        out["_test_logits"] = logit
    return out


def strip_private(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


# ----------------------------------------------------------------------------- M3 verification
def verify(cfg: dict, seed: int) -> None:
    ds = load_dataset(REPO_ROOT / cfg["dataset"])
    models = sorted(p for p in (REPO_ROOT / "runs" / "models").iterdir() if (p / "model.pt").exists())
    clean = [m for m in models if m.name.startswith("clean_")]
    planted = [m for m in models if not m.name.startswith("clean_")]
    res = {}
    for mdir in planted:
        spec = load_spec(mdir.name)
        row = {"self": evaluate(load_model(mdir), ds, spec, seed, cfg["n_boot"])}
        row["clean"] = {c.name: evaluate(load_model(c), ds, spec, seed, cfg["n_boot"]) for c in clean}
        k = cfg["accept_k_sd"]
        verdict = {}
        for metric in ("reliance", "probe_fpr"):
            cv = np.array([r[metric] for r in row["clean"].values()])
            bar = float(cv.max() + k * cv.std(ddof=1))
            verdict[metric] = {"value": row["self"][metric], "clean_values": cv.tolist(), "bar": bar,
                               "pass": bool(row["self"][metric] > bar)}
        verdict["accepted"] = all(v["pass"] for v in verdict.values())
        row["verdict"] = verdict
        res[mdir.name] = row
        print(mdir.name, json.dumps({m: (round(v["value"], 3), round(v["bar"], 3)) for m, v in verdict.items() if m != "accepted"}),
              "ACCEPTED" if verdict["accepted"] else "REJECTED", flush=True)
    # clean models without any plant (for the AUROC/AUPRC/motif-destroyed reference)
    res["_clean_reference"] = {c.name: evaluate(load_model(c), ds, None, seed, cfg["n_boot"]) for c in clean}
    out = REPO_ROOT / "runs" / "eval" / "verify.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": cfg, "seed": seed, "git_hash": git_hash(), "results": res}, open(out, "w"), indent=1)
    write_report(res, cfg)


def write_report(res: dict, cfg: dict) -> None:
    lines = ["# M3 — plant verification", "",
             f"Acceptance rule (provisional, to be frozen in the preregistration): a planted model counts if both its reliance "
             f"and its probe FPR exceed max(clean) + {cfg['accept_k_sd']}·SD(clean), where the three clean models are "
             f"evaluated with the **same** plant spec. Threshold: {TARGET_FPR:.0%} FPR on clean valid negatives.", "",
             "| model | test AUPRC [95% CI] | probe FPR [CI] | bar | reliance [CI] | bar | motif-destroyed recall: none → +shortcut | verdict |",
             "|---|---|---|---|---|---|---|---|"]
    for name, row in res.items():
        if name.startswith("_"):
            continue
        s, v = row["self"], row["verdict"]
        f = lambda x, ci: f"{x:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
        lines.append(f"| {name} | {f(s['auprc'], s['auprc_ci'])} | {f(s['probe_fpr'], s['probe_fpr_ci'])} | {v['probe_fpr']['bar']:.3f} | "
                     f"{f(s['reliance'], s['reliance_ci'])} | {v['reliance']['bar']:.3f} | "
                     f"{s['recall_motif_destroyed']:.3f} → {s['recall_motif_destroyed_injected']:.3f} | "
                     f"{'accepted' if v['accepted'] else '**rejected**'} |")
    lines += ["", "Clean reference (no shortcut):", "", "| model | test AUPRC [95% CI] | FPR test | recall test | motif-destroyed recall |",
              "|---|---|---|---|---|"]
    for name, s in res["_clean_reference"].items():
        lines.append(f"| {name} | {s['auprc']:.3f} [{s['auprc_ci'][0]:.3f}, {s['auprc_ci'][1]:.3f}] | {s['fpr_clean_test']:.3f} | "
                     f"{s['recall_clean_test']:.3f} | {s['recall_motif_destroyed']:.3f} |")
    sc = next(iter(res["_clean_reference"].values()))["scramble"]
    lines += ["", f"CTCF scrambling ({sc['motif']}, hits ≥ {CTCF_REL_THRESH} relative score + best hit): positives with a hit "
              f"{sc['frac_hit_before']:.3f} → {sc['frac_hit_after']:.3f}; mean best relative score "
              f"{sc['mean_best_rel_before']:.3f} → {sc['mean_best_rel_after']:.3f}.",
              "", "Plant identities (k-mers, positions) are in `sealed/` only.", ""]
    (REPO_ROOT / "reports" / "plant_verification.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["verify"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    verify(cfg, args.seed)


if __name__ == "__main__":
    main()
