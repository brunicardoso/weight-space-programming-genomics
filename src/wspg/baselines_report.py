"""Rescore runs/baselines/*.json with the current fix criterion (eval.fix_verdict) against all clean seeds
(clean evaluations per spec come from runs/eval/verify.json), and summarize into reports/baselines.md.

  pixi run python -m wspg.baselines_report --config configs/baselines.yaml
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import numpy as np

from wspg.eval import N_CLEAN_STRICT, fix_verdict
from wspg.utils import REPO_ROOT, load_config

ORDER = ["oracle", "finetune", "interp", "random_vs_interp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    files = sorted((REPO_ROOT / "runs" / "baselines").glob("*.json"))
    verify = json.load(open(REPO_ROOT / "runs" / "eval" / "verify.json"))["results"]
    rows = defaultdict(list)          # (condition, baseline, budget) -> list of entries
    per_model = []
    n_clean = None
    for f in files:
        r = json.load(open(f))
        cl = verify[r["model"]]["clean"]
        names = sorted(cl, key=lambda n: int(n.split("_s")[1]))
        clean_ref = [cl[n] for n in names]
        n_clean = len(clean_ref)
        for b in r["baselines"]:
            old = b.get("verdict_v1", b["verdict"])
            b["verdict_v1"] = old
            b["verdict"] = fix_verdict(b["eval"], r["planted"], clean_ref, old["auprc_drop"], old["auprc_drop_ci"],
                                       clean_strict=clean_ref[:N_CLEAN_STRICT])
        r["rescored_with"] = names
        json.dump(r, open(f, "w"), indent=1, default=float)
        cond = re.sub(r"_s\d+$", "", r["model"])
        for b in r["baselines"]:
            rows[(cond, b["baseline"], b["budget"])].append(b)
        it = [b for b in r["baselines"] if b["baseline"] == "interp"]
        per_model.append((r["model"], r["planted"], it))

    conds = sorted({k[0] for k in rows}, key=lambda c: (c.split("_p")[0], c))
    lines = ["# M4 — baselines", "",
             f"Config `configs/baselines.yaml`; {len(files)} accepted planted models; {n_clean} clean reference seeds. Budget B ⇒ N = 256·B sequences.", "",
             "- **fixed** (primary): probe FPR and reliance ≤ the highest upper 95% CI bound over the clean reference models "
             "(same plant spec), and clean test AUPRC drop ≤ 0.01 vs the planted model.",
             f"- **strict** (secondary, literal AGENTS.md reading): ≤ max of the clean_s0..s{N_CLEAN_STRICT - 1} point estimates, same AUPRC rule.",
             "- **R**: residual shortcut fraction (edited − clean mean) / (planted − clean mean); 1 = nothing removed, 0 = clean.", "",
             "Cells: fixed / total; strict / total; mean probe FPR; mean reliance (logits); mean R(probe), R(reliance); mean AUPRC drop; mean % weights changed.", ""]
    for cond in conds:
        lines += [f"## {cond}", "",
                  "| baseline | budget | fixed | strict | probe FPR | reliance | R probe | R reliance | AUPRC drop | % weights changed |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        keys = sorted([k for k in rows if k[0] == cond], key=lambda k: (ORDER.index(k[1]), k[2] or 0))
        for k in keys:
            bs = rows[k]
            fx = sum(b["verdict"]["fixed"] for b in bs)
            st = sum(b["verdict"]["fixed_strict"] for b in bs)
            m = lambda f: np.mean([f(b) for b in bs])
            res = lambda k: np.mean([b["verdict"][k]["residual"] for b in bs if b["verdict"][k]["residual"] is not None])
            chg = [b["frac_weights_changed"] for b in bs if b["frac_weights_changed"] is not None]
            lines.append(f"| {k[1]} | {k[2] if k[2] is not None else '—'} | {fx}/{len(bs)} | {st}/{len(bs)} | "
                         f"{m(lambda b: b['verdict']['probe_fpr']['value']):.3f} | {m(lambda b: b['verdict']['reliance']['value']):.2f} | "
                         f"{res('probe_fpr'):.3f} | {res('reliance'):.3f} | {m(lambda b: b['verdict']['auprc_drop']):+.4f} | "
                         + (f"{100 * np.mean(chg):.2f}% |" if chg else "— |"))
        lines.append("")

    lines += ["## Interpretability pipeline: what TF-MoDISco found", "",
              "Whether the chosen non-CTCF pattern matches the plant is checked against `sealed/` here and reported only as yes/no.", "",
              "| model | budget | patterns (CTCF / other) | chosen pattern = plant? | k filters | dev reliance before → after | fixed |",
              "|---|---|---|---|---|---|---|"]
    for name, planted, its in per_model:
        spec = json.load(open(REPO_ROOT / "sealed" / "plants" / f"{name}.json"))
        for b in its:
            i = b["info"]
            n_ctcf = sum(p["is_ctcf"] for p in i["patterns"])
            match = _matches(spec, i.get("chosen"))
            lines.append(f"| {name} | {b['budget']} | {n_ctcf} / {len(i['patterns']) - n_ctcf} | {match} | {i.get('k', '—')} | "
                         + (f"{i['dev_reliance_before']:.2f} → {i['dev_reliance_after']:.2f}" if "k" in i else i["outcome"])
                         + f" | {'yes' if b['verdict']['fixed'] else 'no'} |")
    (REPO_ROOT / "reports" / "baselines.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _matches(spec: dict, chosen: str | None) -> str:
    if not chosen:
        return "—"
    v = spec["variant"]
    if v == "kmer":
        k = spec["kmer"]
        rc = k[::-1].translate(str.maketrans("ACGT", "TGCA"))
        best = max(_overlap(chosen, k), _overlap(chosen, rc))
        return "yes" if best >= 8 else f"no ({best}/10 bp)"
    if v == "positional":
        return "yes (poly-A/T)" if re.search("A{6,}|T{6,}", chosen) else "no"
    if v == "gc_shift":
        gc = sum(c in "GC" for c in chosen) / len(chosen)
        return f"GC-rich ({gc:.0%})" if gc >= 0.7 else f"no (GC {gc:.0%})"
    return "?"


def _overlap(a: str, b: str) -> int:
    """Max number of matching bases over all ungapped alignments of a and b."""
    best = 0
    for off in range(-len(b) + 1, len(a)):
        best = max(best, sum(1 for j in range(len(b)) if 0 <= off + j < len(a) and a[off + j] == b[j]))
    return best


if __name__ == "__main__":
    main()
