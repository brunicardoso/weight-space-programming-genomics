"""Training loop (infra sessions only). Whole dataset lives on the GPU as uint8 indices;
one-hot + reverse-complement augmentation happen per batch."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from wspg.model import build_model, n_params
from wspg.seq import one_hot
from wspg.utils import REPO_ROOT, load_config, set_seed, write_run_json


def rc_onehot(x: torch.Tensor) -> torch.Tensor:
    """Reverse complement of (B, 4, L) one-hot in ACGT order: flip channels and positions."""
    return x.flip(1).flip(2)


@torch.no_grad()
def predict(model, seqs, device="cuda", batch_size: int = 1024) -> np.ndarray:
    model.eval()
    seqs_t = torch.as_tensor(seqs, device=device)
    out = [model(one_hot(seqs_t[i:i + batch_size])) for i in range(0, len(seqs_t), batch_size)]
    return torch.cat(out).float().cpu().numpy()


def binary_metrics(y, logits) -> dict:
    return {"auroc": float(roc_auc_score(y, logits)), "auprc": float(average_precision_score(y, logits))}


def fit(model, train_x, train_y, valid_x, valid_y, cfg: dict, device="cuda", log=print):
    """Train with AdamW + early stopping on valid AUPRC. Returns (best_model, history)."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 1e-2))
    bs = cfg.get("batch_size", 256)
    max_epochs = cfg.get("max_epochs", 30)
    patience = cfg.get("patience", 5)
    rc_p = cfg.get("rc_prob", 0.5) if cfg.get("rc_augment", True) else 0.0

    xt = torch.as_tensor(train_x, device=device)
    yt = torch.as_tensor(train_y, device=device, dtype=torch.float32)
    gen = torch.Generator(device=device).manual_seed(int(cfg.get("seed", 0)))

    best, best_auprc, best_epoch, history = None, -1.0, -1, []
    for epoch in range(max_epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(xt), device=device, generator=gen)
        tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            x = one_hot(xt[idx])
            if rc_p > 0:
                flip = torch.rand(len(idx), device=device, generator=gen) < rc_p
                x = torch.where(flip[:, None, None], rc_onehot(x), x)
            loss = F.binary_cross_entropy_with_logits(model(x), yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        vm = binary_metrics(valid_y, predict(model, valid_x, device))
        rec = {"epoch": epoch, "train_loss": tot / len(xt), **{f"valid_{k}": v for k, v in vm.items()}, "sec": time.time() - t0}
        history.append(rec)
        log(json.dumps(rec))
        if vm["auprc"] > best_auprc:
            best_auprc, best_epoch, best = vm["auprc"], epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best)
    return model, history


def load_dataset(path: str | Path) -> dict:
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def split_arrays(ds: dict, split: str):
    m = ds["split"] == split
    return ds["seqs"][m], ds["label"][m].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default=None, help="output dir (default runs/models/<name>_s<seed>)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["train"]["seed"] = args.seed
    set_seed(args.seed)
    ds = load_dataset(REPO_ROOT / cfg["dataset"])
    run_name = f"{cfg['name']}_s{args.seed}"
    if cfg.get("plant"):
        from wspg.plant import apply_plant
        ds, spec = apply_plant(ds, cfg["plant"], seed=args.seed, name=run_name)
        print(f"planted {cfg['plant']} into {len(spec['planted_idx'])} positives (spec in sealed/plants/{run_name}.json)")

    tr_x, tr_y = split_arrays(ds, "train")
    va_x, va_y = split_arrays(ds, "valid")
    te_x, te_y = split_arrays(ds, "test")

    model = build_model(cfg.get("model"))
    print(f"params={n_params(model)} train={len(tr_y)} valid={len(va_y)} test={len(te_y)}")
    model, hist = fit(model, tr_x, tr_y, va_x, va_y, cfg["train"])

    metrics = {"test": binary_metrics(te_y, predict(model, te_x)),
               "valid": binary_metrics(va_y, predict(model, va_x)),
               "best_epoch": int(np.argmax([h["valid_auprc"] for h in hist])), "n_params": n_params(model)}
    print(json.dumps(metrics))
    out = Path(args.out) if args.out else REPO_ROOT / "runs" / "models" / run_name
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    with open(out / "history.json", "w") as f:
        json.dump(hist, f, indent=1)
    write_run_json(out / "run.json", cfg, metrics, seed=args.seed)


if __name__ == "__main__":
    main()
