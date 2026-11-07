"""Hand-built SmallCNN for the M5 plumbing dry run (no data, no training).

  pixi run python harness/toy_model.py [--config configs/harness.yaml] --seed 0

Known answer: the logit is driven by two conv1 paths:
  - CTCF: filters 0/1 match the CTCF consensus (JASPAR MA0139.1 19-mer) and its reverse complement (>= 16/19 bases);
  - shortcut: filters 2/3 match a (CA)9 / (TG)9 dinucleotide repeat (exact 18 bp match; a shorter repeat fires only if chance flanks complete it), at any position.
Neither is one of the planted types (random 10-mer, poly-A at the 5' end, flank GC). All other weights are N(0, noise_sd).
Output: runs/toy/toy_ca/{model.pt, run.json}. run.json holds the answer, so this directory is never exported.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from wspg.model import build_model, n_params
from wspg.seq import encode_str, one_hot
from wspg.utils import REPO_ROOT, set_seed, write_run_json

CTCF = "TGGCCACCAGGGGGCGCTA"          # MA0139.1 consensus, 19 bp
CTCF_RC = "TAGCGCCCCCTGGTGGCCA"
CA = "CA" * 9                          # 18 bp
TG = "TG" * 9
NOISE_SD = 0.01
OUT = REPO_ROOT / "runs" / "toy" / "toy_ca"


def _pwm_filter(motif: str, width: int = 19) -> np.ndarray:
    """+1 for the motif base, -1 for the other three; columns beyond the motif are 0. Score = 2*matches - len(motif)."""
    w = np.zeros((4, width), dtype=np.float32)
    for j, b in enumerate(motif):
        w[:, j] = -1.0
        w["ACGT".index(b), j] = 1.0
    return w


def build(seed: int) -> torch.nn.Module:
    set_seed(seed)
    m = build_model({})
    rng = np.random.default_rng(seed)
    sd = {k: torch.from_numpy(rng.normal(0, NOISE_SD, v.shape).astype(np.float32)) for k, v in m.state_dict().items()}
    # conv1: CTCF fires at >= 16/19 matches (score >= 13), repeat only at 18/18 (score 18)
    for f, motif, bias in ((0, CTCF, -12.0), (1, CTCF_RC, -12.0), (2, CA, -17.0), (3, TG, -17.0)):
        sd["conv1.weight"][f] = torch.from_numpy(_pwm_filter(motif))
        sd["conv1.bias"][f] = bias
    # conv2 (width 7) / conv3 (width 5): centre taps route {0,1} -> channel 0 and {2,3} -> channel 1
    sd["conv2.weight"][:2, :, 3] = 0.0
    for c_in, c_out in ((0, 0), (1, 0), (2, 1), (3, 1)):
        sd["conv2.weight"][c_out, c_in, 3] = 1.0
    sd["conv2.bias"][:2] = 0.0
    sd["conv3.weight"][:2, :, 2] = 0.0
    sd["conv3.weight"][0, 0, 2] = 1.0
    sd["conv3.weight"][1, 1, 2] = 1.0
    sd["conv3.bias"][:2] = 0.0
    sd["fc1.weight"][:2] = 0.0
    sd["fc1.weight"][0, 0] = 1.0
    sd["fc1.weight"][1, 1] = 1.0
    sd["fc1.bias"][:2] = 0.0
    sd["fc2.weight"][0, 0] = 3.0
    sd["fc2.weight"][0, 1] = 6.0
    sd["fc2.bias"][0] = -2.0
    m.load_state_dict(sd)
    return m.eval()


def _logits(m, seqs: list[str]) -> np.ndarray:
    x = one_hot(torch.from_numpy(np.stack([encode_str(s) for s in seqs])))
    with torch.no_grad():
        return m(x).numpy()


def _insert(s: str, motif: str, pos: int) -> str:
    return s[:pos] + motif + s[pos + len(motif):]


def self_check(m, seed: int) -> dict:
    rng = np.random.default_rng(seed + 1)
    bg = ["".join(rng.choice(list("ACGT"), 500)) for _ in range(1000)]
    positions = [0, 100, 241, 400, 482]
    res = {"random": _logits(m, bg)}
    for name, motif in (("ctcf", CTCF), ("ctcf_rc", CTCF_RC), ("ca", CA), ("tg", TG)):
        res[name] = np.concatenate([_logits(m, [_insert(s, motif, p) for s in bg[:100]]) for p in positions])
    res["ca8"] = _logits(m, [_insert(s, "CA" * 8, 241) for s in bg[:200]])
    summary = {k: {"min": float(v.min()), "max": float(v.max()), "frac_pos": float((v > 0).mean())} for k, v in res.items()}
    ok = (summary["random"]["frac_pos"] <= 0.01 and all(summary[k]["frac_pos"] >= 0.99 for k in ("ctcf", "ctcf_rc", "ca", "tg"))
          and summary["ca8"]["frac_pos"] <= 0.2)
    return {"ok": bool(ok), "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "harness.yaml"), help="unused; script convention")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    m = build(args.seed)
    chk = self_check(m, args.seed)
    for k, v in chk["summary"].items():
        print(f"{k:8s} logit min {v['min']:+.3f} max {v['max']:+.3f} frac>0 {v['frac_pos']:.3f}")
    if not chk["ok"]:
        raise SystemExit("toy self-check FAILED")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), OUT / "model.pt")
    toy = {"variant": "toy_ca_repeat", "ctcf_consensus": CTCF, "repeat": "(CA)n / (TG)n of >= 18 bp (exact), any position",
           "conv1_filters": {"ctcf": [0, 1], "repeat": [2, 3]}, "noise_sd": NOISE_SD}
    write_run_json(OUT / "run.json", {"name": "toy_ca", "model": {}, "toy": toy}, {"self_check": chk["summary"],
                   "n_params": n_params(m)}, seed=args.seed)
    print(f"toy self-check passed; wrote {OUT}")


if __name__ == "__main__":
    main()
