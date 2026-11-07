"""Smoke test: 2 epochs on 2k sequences, must finish in < 2 min. Checks plumbing + timing, not learning:
2 epochs (16 steps) is too few to pass the initial loss plateau on the synthetic task (~300 steps).
Uses the built dataset if present, else synthetic DNA with an embedded CTCF-like consensus."""
from __future__ import annotations

import argparse
import time

import numpy as np

from wspg.model import build_model, n_params
from wspg.seq import encode_str
from wspg.train import binary_metrics, fit, load_dataset, predict
from wspg.utils import REPO_ROOT, load_config, set_seed

CTCF_CONSENSUS = "CCGCGNGGNGGCAG".replace("N", "A")


def synthetic(n: int, L: int, rng) -> tuple[np.ndarray, np.ndarray]:
    x = rng.integers(0, 4, size=(n, L), dtype=np.uint8)
    y = (np.arange(n) % 2).astype(np.float32)
    motif = encode_str(CTCF_CONSENSUS)
    for i in np.where(y == 1)[0]:
        p = rng.integers(L // 2 - 50, L // 2 + 50)
        x[i, p:p + len(motif)] = motif
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_config(args.config)
    cfg["train"]["seed"] = args.seed
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    n, L = cfg["n_seqs"], cfg["seq_len"]

    path = REPO_ROOT / cfg["dataset"]
    if path.exists():
        ds = load_dataset(path)
        def take(split, k):
            idx = np.where(ds["split"] == split)[0]
            idx = rng.choice(idx, size=min(k, len(idx)), replace=False)
            return ds["seqs"][idx], ds["label"][idx].astype(np.float32)
        tr_x, tr_y = take("train", n)
        va_x, va_y = take("valid", n // 4)
        source = str(path)
    else:
        tr_x, tr_y = synthetic(n, L, rng)
        va_x, va_y = synthetic(n // 4, L, rng)
        source = "synthetic"

    model = build_model(cfg.get("model"))
    print(f"source={source} params={n_params(model)} n_train={len(tr_y)}")
    model, _ = fit(model, tr_x, tr_y, va_x, va_y, cfg["train"])
    m = binary_metrics(va_y, predict(model, va_x))
    dt = time.time() - t0
    print(f"SMOKE valid auroc={m['auroc']:.3f} auprc={m['auprc']:.3f} time={dt:.1f}s")
    if dt > 120:
        raise SystemExit("FAIL: smoke took longer than 2 min")


if __name__ == "__main__":
    main()
