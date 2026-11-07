"""M4 non-LLM baselines for removing a planted shortcut.

Data budgets mirror the WSP forward-pass budget: B calls x 256 sequences = N sequences.

1. oracle      the clean model with the same seed (= retrain with the plant removed from the data).
2. finetune    gradient fine-tuning of fc1+fc2 (conv frozen) on N CLEAN dev sequences
               (dev pool = clean train+valid windows; test is never used). Fixed recipe, 3 data draws.
3. interp      standard interpretability pipeline, no LLM, on N sequences from the CONTAMINATED valid split
               (chr16; contaminated on purpose, so it models held-out data from the same flawed distribution;
               capped at its size). gradient x input (mean-centred hypothetical contributions) on positives ->
               TF-MoDISco-lite -> TomTom vs JASPAR: patterns whose top-3 matches include CTCF/CTCFL are "CTCF".
               Among non-CTCF patterns, pick the one whose consensus raises the dev-negative logit most; rank
               conv1 filters by how much the consensus raises their max activation; zero the top-k filters,
               k = argmin over 1..max_filters of the dev reliance (keep zeroing while it still helps).
               (Pilot on kmer_p0.9_s0: a ">= 90% reduction" rule stopped at k=5 and a ">= 98%" rule at k=7, both
               leaving test reliance above the clean range; the k-mer is spread over ~10-12 filters. The rule was
               fixed to argmin BEFORE the full run, to keep this baseline as strong as a practitioner could make it.)
               The test metrics for every k are recorded as a sensitivity curve.
               FROZEN at v3 (2026-11-03, user decision): v1 had TomTom-on-PPM CTCF labels (poly-A mislabelled CTCF),
               no edge padding and random-position injection; v2 fixed those but labelled embedded CTCF cores as
               non-CTCF; v3 = contiguous-core trimming + (consensus TomTom p<0.01 OR PWM hit >= 0.8) CTCF label.
               These fixes were made by the infra session with answer-key exposure; all versions are reported.
4. random      perturb the same number of weights by the same magnitudes (random signs) as a reference edit,
               10 draws. Here the reference is the interp edit; later also every LLM phase-B edit.

  pixi run python -m wspg.baselines --config configs/baselines.yaml --seed 0
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from wspg.eval import auprc_drop, evaluate, fix_verdict, frac_weights_changed, load_model, strip_private
from wspg.plant import apply_spec, load_spec
from wspg.seq import decode, encode_str, one_hot, revcomp_idx
from wspg.train import load_dataset, predict, rc_onehot
from wspg.utils import REPO_ROOT, git_hash, load_config, set_seed

DEVICE = "cuda"


# ----------------------------------------------------------------------------- 2. fine-tune
def finetune(model, x: np.ndarray, y: np.ndarray, cfg: dict, seed: int):
    m = copy.deepcopy(model).train()
    for name, p in m.named_parameters():
        p.requires_grad = name.startswith(("fc1", "fc2"))
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    xt, yt = torch.as_tensor(x, device=DEVICE), torch.as_tensor(y, device=DEVICE, dtype=torch.float32)
    for _ in range(cfg["epochs"]):
        perm = torch.randperm(len(xt), device=DEVICE, generator=g)
        for i in range(0, len(perm), cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            xb = one_hot(xt[idx])
            flip = torch.rand(len(idx), device=DEVICE, generator=g) < 0.5
            xb = torch.where(flip[:, None, None], rc_onehot(xb), xb)
            loss = F.binary_cross_entropy_with_logits(m(xb), yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return m.eval()


def sample_balanced(rng, idx_pos, idx_neg, n):
    k = n // 2
    return np.concatenate([rng.choice(idx_pos, min(k, len(idx_pos)), replace=False),
                           rng.choice(idx_neg, min(k, len(idx_neg)), replace=False)])


# ----------------------------------------------------------------------------- 3. interpretability fix
def hyp_contribs(model, x: np.ndarray, bs: int = 256) -> np.ndarray:
    out = []
    for i in range(0, len(x), bs):
        oh = one_hot(torch.as_tensor(x[i:i + bs], device=DEVICE)).requires_grad_(True)
        model(oh).sum().backward()
        g = oh.grad
        out.append((g - g.mean(1, keepdim=True)).permute(0, 2, 1).cpu().numpy())
    return np.concatenate(out)


def jaspar_all():
    from pyjaspar import jaspardb

    from wspg.plant import JASPAR_RELEASE

    ms = jaspardb(release=JASPAR_RELEASE).fetch_motifs(collection="CORE", tax_group=["vertebrates"])
    pfms = []
    for m in ms:
        c = np.array([m.counts[b] for b in "ACGT"], float) + 0.25
        pfms.append(c / c.sum(0))
    return ms, pfms


MODISCO_PAD = 25   # zero-pad both ends so seqlets at the window edges are not discarded (>= window + flank)


def run_modisco(model, x_pos: np.ndarray, cfg: dict):
    from modiscolite.tfmodisco import TFMoDISco

    hyp = hyp_contribs(model, x_pos).astype(np.float32)
    oh = np.eye(4, dtype=np.float32)[np.minimum(x_pos, 3)] * (x_pos < 4)[..., None]
    padw = ((0, 0), (MODISCO_PAD, MODISCO_PAD), (0, 0))
    # one-hot padding is uniform 0.25 (not 0): all-zero columns make PPM columns 0/0 = NaN and the edge clusters are lost
    pos, neg = TFMoDISco(one_hot=np.pad(oh, padw, constant_values=0.25), hypothetical_contribs=np.pad(hyp, padw), **cfg)
    L = x_pos.shape[1]
    pats = []
    for sign, group in (("pos", pos or []), ("neg", neg or [])):
        for p in group:
            cwm = np.asarray(p.contrib_scores)               # (L, 4)
            ppm = np.asarray(p.sequence)
            imp = np.abs(cwm).sum(1)
            # trim to the contiguous core around peak importance (v3; v1-v2 took min..max of all positions >= 30%)
            lo = hi = int(imp.argmax())
            hi += 1
            while lo > 0 and imp[lo - 1] >= 0.3 * imp.max():
                lo -= 1
            while hi < len(imp) and imp[hi] >= 0.3 * imp.max():
                hi += 1
            # where the trimmed pattern sits in the original (unpadded) window, and on which strand
            locs = []
            for sq in p.seqlets:
                st = (sq.end - hi if sq.is_revcomp else sq.start + lo) - MODISCO_PAD
                if 0 <= st <= L - (hi - lo):
                    locs.append((int(st), bool(sq.is_revcomp)))
            pats.append({"sign": sign, "n_seqlets": len(p.seqlets), "ppm": ppm[lo:hi], "cwm": cwm[lo:hi],
                         "consensus": decode(ppm[lo:hi].argmax(1)), "locs": locs,
                         "loc_median": float(np.median([l[0] for l in locs])) if locs else None})
    return pats


CTCF_CORE_IDS = ("MA0139", "MA1102")   # CTCF, CTCFL core motifs
CTCF_P = 0.01


def _consensus_query(s: str) -> np.ndarray:
    return (np.eye(4)[["ACGT".index(c) for c in s]].T * 0.97 + 0.0075).astype(float)


def annotate_patterns(pats, motifs, pfms):
    """Label a pattern CTCF if its consensus matches a CTCF/CTCFL core motif (TomTom p < 0.01).
    The consensus is used instead of the PPM: TomTom on long, low-information PPMs gave spurious CTCF hits
    (e.g. poly-A patterns at p ~ 1e-10 against extended CTCF motifs)."""
    from memelite import tomtom

    if not pats:
        return pats
    p = np.asarray(tomtom([_consensus_query(pt["consensus"]) for pt in pats], pfms)[0])
    core = [i for i, m in enumerate(motifs) if m.matrix_id.split(".")[0] in CTCF_CORE_IDS]
    from wspg.eval import CTCF_REL_THRESH, ctcf_logodds, ctcf_scan

    m = ctcf_logodds()[0].shape[1]
    for pt, row in zip(pats, p):
        top = np.argsort(row)[:3]
        pt["jaspar_top"] = [{"id": motifs[i].matrix_id, "name": motifs[i].name, "p": float(row[i])} for i in top]
        pt["ctcf_core_p"] = float(row[core].min())
        # v3: also call CTCF if the consensus contains a CTCF PWM hit (same rule as eval.scramble_ctcf);
        # catches a CTCF core embedded in a long pattern, which TomTom against the 15-bp core misses
        c = encode_str(pt["consensus"])
        pt["ctcf_pwm_max"] = float(ctcf_scan(c[None]).max()) if len(c) >= m else 0.0
        pt["is_ctcf"] = pt["ctcf_core_p"] < CTCF_P or pt["ctcf_pwm_max"] >= CTCF_REL_THRESH
    return pats


def _inject_pattern(x: np.ndarray, pat: dict, rng) -> np.ndarray:
    """Insert the pattern consensus at locations (and strands) drawn from where its seqlets were found;
    falls back to uniform random positions if no seqlet location is available."""
    x = x.copy()
    k = encode_str(pat["consensus"])
    krc = revcomp_idx(k)
    n, L = x.shape
    if pat.get("locs"):
        pick = rng.integers(0, len(pat["locs"]), size=n)
        locs = [pat["locs"][i] for i in pick]
    else:
        locs = [(int(s), False) for s in rng.integers(0, L - len(k) + 1, size=n)]
    for i, (st, rc) in enumerate(locs):
        x[i, st:st + len(k)] = krc if rc else k
    return x


@torch.no_grad()
def conv1_max_act(model, x: np.ndarray, bs: int = 1024) -> np.ndarray:
    out = []
    for i in range(0, len(x), bs):
        a = F.relu(model.conv1(one_hot(torch.as_tensor(x[i:i + bs], device=DEVICE))))
        out.append(a.amax(-1).cpu().numpy())
    return np.concatenate(out)


def zero_filters(model, filt: list[int]):
    m = copy.deepcopy(model)
    with torch.no_grad():
        m.conv1.weight[filt] = 0
        m.conv1.bias[filt] = 0
    return m


def interp_fix(model, dev_x, dev_y, cfg, rng, motifs, pfms, curve_eval=None):
    pats = annotate_patterns(run_modisco(model, dev_x[dev_y == 1], cfg["modisco"]), motifs, pfms)
    info = {"patterns": [{k: v for k, v in p.items() if k not in ("ppm", "cwm", "locs")} for p in pats]}
    cand = [p for p in pats if not p["is_ctcf"]]
    if not cand:
        info["outcome"] = "no non-CTCF pattern found"
        return model, info
    neg = dev_x[dev_y == 0]
    base = predict(model, neg)
    rel = []
    for p in cand:
        rel.append(float((predict(model, _inject_pattern(neg, p, np.random.default_rng(0))) - base).mean()))
    best = cand[int(np.argmax(rel))]
    info.update(candidate_reliance=rel, chosen=best["consensus"], chosen_loc_median=best["loc_median"])
    inj = _inject_pattern(neg, best, np.random.default_rng(0))
    order = np.argsort(-(conv1_max_act(model, inj) - conv1_max_act(model, neg)).mean(0))
    r0 = max(rel)
    curve, chosen = [], None
    for k in range(1, cfg["max_filters"] + 1):
        mk = zero_filters(model, order[:k].tolist())
        rk = float((predict(mk, inj) - predict(mk, neg)).mean())
        point = {"k": k, "dev_reliance": rk}
        if curve_eval is not None:
            point.update(curve_eval(mk))
        curve.append(point)
        if chosen is None or rk < chosen[2]:
            chosen = (k, mk, rk)
    k, m, rk = chosen
    info.update(k=k, filters=order[:k].tolist(), dev_reliance_before=r0, dev_reliance_after=rk,
                curve=curve, outcome="edited")
    return m, info


# ----------------------------------------------------------------------------- 4. random edit
def random_edit(orig: torch.nn.Module, edited: torch.nn.Module, rng: np.random.Generator):
    """Same number of changed weights and the same |delta| multiset as `edited`, at random positions with random signs."""
    so, se = orig.state_dict(), edited.state_dict()
    flat_o = torch.cat([v.flatten() for v in so.values()])
    delta = torch.cat([(se[k] - so[k]).flatten() for k in so])
    mags = delta[delta != 0].abs().cpu().numpy()
    if len(mags) == 0:
        return copy.deepcopy(orig)
    pos = rng.choice(len(flat_o), size=len(mags), replace=False)
    new = flat_o.clone()
    new[torch.as_tensor(pos, device=new.device)] += torch.as_tensor(rng.permutation(mags) * rng.choice([-1, 1], len(mags)),
                                                                     device=new.device, dtype=new.dtype)
    m = copy.deepcopy(orig)
    sd, i = {}, 0
    for k, v in so.items():
        sd[k] = new[i:i + v.numel()].view_as(v)
        i += v.numel()
    m.load_state_dict(sd)
    return m


# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None, help="comma-separated model names (default: all accepted planted)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    ds = load_dataset(REPO_ROOT / cfg["dataset"])
    verify = json.load(open(REPO_ROOT / "runs" / "eval" / "verify.json"))["results"]
    names = [n for n, r in verify.items() if not n.startswith("_") and r["verdict"]["accepted"]]
    if args.only:
        names = [n for n in names if n in args.only.split(",")]
    motifs, pfms = jaspar_all()
    y_test = ds["label"][ds["split"] == "test"].astype(int)
    clean_dirs = sorted((REPO_ROOT / "runs" / "models").glob("clean_s*"), key=lambda p: int(p.name.split("_s")[1]))
    clean_models = {int(d.name.split("_s")[1]): load_model(d) for d in clean_dirs}   # s0..s2 strict, all = reference
    nb = cfg["n_boot"]
    dev_pool = np.where(ds["split"] != "test")[0]
    out_dir = REPO_ROOT / "runs" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        t0 = time.time()
        spec = load_spec(name)
        seed = spec["seed"]
        planted = load_model(REPO_ROOT / "runs" / "models" / name)
        ev = lambda m, n_boot=nb: evaluate(m, ds, spec, args.seed, n_boot, return_logits=True)
        e_planted = ev(planted)
        e_clean = [ev(m) for m in clean_models.values()]
        def verdict(e, n_boot=nb):
            return fix_verdict(e, e_planted, e_clean, *auprc_drop(e_planted, e, y_test, args.seed, n_boot))
        res = {"model": name, "planted": strip_private(e_planted), "baselines": []}

        # 1. oracle
        e = e_clean[seed]
        res["baselines"].append({"baseline": "oracle", "budget": None, "eval": strip_private(e), "verdict": verdict(e),
                                 "frac_weights_changed": None})
        rng = np.random.default_rng([args.seed, seed, 1])
        cds = apply_spec(ds, spec)
        interp_cache = {}

        def curve_eval(m):
            e = ev(m, n_boot=cfg["n_boot_random"])
            v = verdict(e, cfg["n_boot_random"])
            return {"test_probe_fpr": e["probe_fpr"], "test_reliance": e["reliance"], "auprc_drop": v["auprc_drop"],
                    "fixed": v["fixed"]}

        for B in cfg["budgets"]:
            N = B * 256
            # 2. fine-tune on clean dev data
            for draw in range(cfg["finetune"]["draws"]):
                pos = dev_pool[ds["label"][dev_pool] == 1]
                neg = dev_pool[ds["label"][dev_pool] == 0]
                idx = sample_balanced(rng, pos, neg, N)
                m = finetune(planted, ds["seqs"][idx], ds["label"][idx], cfg["finetune"], seed=draw)
                e = ev(m)
                res["baselines"].append({"baseline": "finetune", "budget": B, "draw": draw, "n_seqs": len(idx),
                                         "eval": strip_private(e), "verdict": verdict(e),
                                         "frac_weights_changed": frac_weights_changed(planted.state_dict(), m.state_dict())})
            # 3. interpretability fix on contaminated valid (budgets that exhaust the split share one result)
            vi = np.where(cds["split"] == "valid")[0]
            n_eff = min(N, len(vi))
            if n_eff not in interp_cache:
                vi = sample_balanced(rng, vi[cds["label"][vi] == 1], vi[cds["label"][vi] == 0], N)
                interp_cache[n_eff] = interp_fix(planted, cds["seqs"][vi], cds["label"][vi], cfg["interp"], rng,
                                                 motifs, pfms, curve_eval=curve_eval) + (len(vi),)
            m, info, n_used = interp_cache[n_eff]
            e = ev(m)
            res["baselines"].append({"baseline": "interp", "budget": B, "n_seqs": n_used, "info": info,
                                     "eval": strip_private(e), "verdict": verdict(e),
                                     "frac_weights_changed": frac_weights_changed(planted.state_dict(), m.state_dict())})
            # 4. random edit matched to the interp edit
            if info.get("outcome") == "edited":
                for d in range(cfg["random_draws"]):
                    mr = random_edit(planted, m, np.random.default_rng([args.seed, seed, B, d]))
                    e = ev(mr, n_boot=cfg["n_boot_random"])
                    res["baselines"].append({"baseline": "random_vs_interp", "budget": B, "draw": d,
                                             "eval": strip_private(e),
                                             "verdict": verdict(e, cfg["n_boot_random"]),
                                             "frac_weights_changed": frac_weights_changed(planted.state_dict(), mr.state_dict())})
        res["git_hash"], res["config"], res["sec"] = git_hash(), cfg, time.time() - t0
        json.dump(res, open(out_dir / f"{name}.json", "w"), indent=1, default=float)
        fixed = {}
        for b in res["baselines"]:
            fixed.setdefault((b["baseline"], b["budget"]), []).append(b["verdict"]["fixed"])
        print(name, f"{res['sec']:.0f}s", {f"{k[0]}@{k[1]}": f"{sum(v)}/{len(v)}" for k, v in fixed.items()}, flush=True)


if __name__ == "__main__":
    main()
