"""M1 data pipeline: ENCODE download, windows, GC-matched negatives, chromosome splits.

  pixi run python -m wspg.data download --config configs/data.yaml
  pixi run python -m wspg.data build    --config configs/data.yaml --seed 0
  pixi run python -m wspg.data stats    --config configs/data.yaml

Window sequences are extracted once (pyfaidx, one chromosome at a time as uint8 indices) and stored
as (N, 500) uint8; one-hot encoding happens per batch at training time.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from wspg.seq import encode_str
from wspg.utils import REPO_ROOT, git_hash, load_config, set_seed

ENCODE = "https://www.encodeproject.org"
HDR = {"Accept": "application/json"}
RAW = REPO_ROOT / "data" / "raw"


# ----------------------------------------------------------------------------- download
def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, dest: Path) -> None:
    if dest.exists():
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(dest)


def find_peak_file(cfg: dict) -> dict:
    params = {"type": "File", "target.label": cfg["target"], "biosample_ontology.term_name": cfg["cell_line"],
              "assay_title": "TF ChIP-seq", "assembly": cfg["assembly"], "status": "released",
              "file_format": "bed", "output_type": cfg["peak_output_type"], "limit": "all", "frame": "object"}
    files = requests.get(f"{ENCODE}/search/", params=params, headers=HDR, timeout=60).json()["@graph"]
    if not files:
        raise RuntimeError("no ENCODE peak files matched the query")
    # rule: preferred_default first, then most recently created
    files.sort(key=lambda f: (bool(f.get("preferred_default")), f.get("date_created", "")), reverse=True)
    f = files[0]
    return {"role": "peaks", "accession": f["accession"], "dataset": f["dataset"], "md5sum": f["md5sum"],
            "url": ENCODE + f["href"], "preferred_default": bool(f.get("preferred_default")),
            "date_created": f.get("date_created"), "n_candidates": len(files),
            "candidates": [c["accession"] for c in files]}


def find_blacklist(cfg: dict) -> dict:
    params = {"type": "Annotation", "annotation_type": cfg["exclusion_list_annotation_type"],
              "assembly": cfg["assembly"], "status": "released", "limit": "all", "frame": "object"}
    anns = requests.get(f"{ENCODE}/search/", params=params, headers=HDR, timeout=60).json()["@graph"]
    anns = [a for a in anns if cfg["exclusion_list_description_contains"] in a.get("description", "")]
    if len(anns) != 1:
        raise RuntimeError(f"expected 1 exclusion-list annotation, got {[a['accession'] for a in anns]}")
    ann = anns[0]
    for fid in ann["files"]:
        f = requests.get(f"{ENCODE}{fid}?frame=object", headers=HDR, timeout=60).json()
        if f.get("assembly") == cfg["assembly"] and f["file_format"] == "bed" and f["status"] == "released":
            return {"role": "blacklist", "accession": f["accession"], "dataset": ann["accession"],
                    "md5sum": f["md5sum"], "url": ENCODE + f["href"], "date_created": f.get("date_created")}
    raise RuntimeError("no released GRCh38 bed in exclusion-list annotation")


def download(cfg: dict) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    manifest = []
    for rec in (find_peak_file(cfg), find_blacklist(cfg)):
        dest = RAW / Path(rec["url"]).name
        _fetch(rec["url"], dest)
        got = _md5(dest)
        if got != rec["md5sum"]:
            raise RuntimeError(f"md5 mismatch for {dest}: {got} != {rec['md5sum']}")
        rec.update(local=str(dest.relative_to(REPO_ROOT)), download_date=today)
        manifest.append(rec)
        print("ok", rec["role"], rec["accession"])

    gz = RAW / "hg38.fa.gz"
    _fetch(cfg["genome_url"], gz)
    md5s = dict(line.split()[::-1] for line in requests.get(cfg["genome_md5_url"], timeout=60).text.splitlines() if line.strip())
    got = _md5(gz)
    if got != md5s.get("hg38.fa.gz"):
        raise RuntimeError(f"hg38.fa.gz md5 mismatch: {got} != {md5s.get('hg38.fa.gz')}")
    fa = RAW / "hg38.fa"
    if not fa.exists():
        with gzip.open(gz, "rb") as fi, open(fa.with_suffix(".fa.part"), "wb") as fo:
            shutil.copyfileobj(fi, fo, 1 << 24)
        fa.with_suffix(".fa.part").rename(fa)
    manifest.append({"role": "genome", "source": "UCSC", "url": cfg["genome_url"], "md5sum": got,
                     "local": str(fa.relative_to(REPO_ROOT)), "download_date": today})
    with open(RAW / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    write_readme(manifest)


def write_readme(manifest: list[dict]) -> None:
    lines = ["# Data provenance", "",
             "Generated by `python -m wspg.data download` (do not edit by hand). The files themselves are gitignored.", "",
             "| role | accession | dataset / source | md5 | download date | url |", "|---|---|---|---|---|---|"]
    for r in manifest:
        lines.append(f"| {r['role']} | {r.get('accession', '-')} | {r.get('dataset', r.get('source', '-'))} | "
                     f"`{r['md5sum']}` | {r['download_date']} | {r['url']} |")
    peaks = next(r for r in manifest if r["role"] == "peaks")
    lines += ["", f"Peak file selection rule: ENCODE files matching the query in `configs/data.yaml`, sorted by "
              f"(preferred_default, date_created) descending; {peaks['n_candidates']} candidates: "
              f"{', '.join(peaks['candidates'])}. Chosen: {peaks['accession']} from {peaks['dataset']}."]
    (REPO_ROOT / "data" / "README.md").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- intervals
def merge_intervals(starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(starts) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    o = np.argsort(starts, kind="stable")
    s, e = starts[o].astype(np.int64), ends[o].astype(np.int64)
    out_s, out_e = [s[0]], [e[0]]
    for a, b in zip(s[1:], e[1:]):
        if a <= out_e[-1]:
            out_e[-1] = max(out_e[-1], b)
        else:
            out_s.append(a); out_e.append(b)
    return np.array(out_s), np.array(out_e)


def overlaps(qs: np.ndarray, qe: np.ndarray, ms: np.ndarray, me: np.ndarray) -> np.ndarray:
    """Half-open query [qs, qe) vs merged, sorted, disjoint intervals [ms, me). Returns bool mask."""
    if len(ms) == 0:
        return np.zeros(len(qs), bool)
    i = np.searchsorted(me, qs, side="right")
    ok = i < len(ms)
    res = np.zeros(len(qs), bool)
    res[ok] = ms[i[ok]] < qe[ok]
    return res


def read_bed(path: Path, ncols: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, comment="#", compression="infer")


def dedup_summits(df: pd.DataFrame, dist: int) -> pd.DataFrame:
    """Within each chrom, collapse summits closer than `dist` (single-linkage) to the one with the highest signal."""
    keep = []
    for _, g in df.sort_values(["chrom", "summit"]).groupby("chrom", sort=False):
        cluster = np.concatenate([[0], np.cumsum(np.diff(g["summit"].to_numpy()) >= dist)])
        keep.append(g.assign(_c=cluster).sort_values("signal", ascending=False).drop_duplicates("_c"))
    return pd.concat(keep).drop(columns="_c").sort_values(["chrom", "summit"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- build
def split_of(chrom: str, cfg: dict) -> str:
    for name, chroms in cfg["splits"].items():
        if chrom in chroms:
            return name
    return "train"


def build(cfg: dict, seed: int) -> None:
    from pyfaidx import Fasta

    set_seed(seed)
    rng = np.random.default_rng(seed)
    manifest = json.load(open(RAW / "manifest.json"))
    loc = {r["role"]: REPO_ROOT / r["local"] for r in manifest}
    W, half = cfg["window"], cfg["window"] // 2
    chroms = cfg["chroms"]

    pk = read_bed(loc["peaks"])
    pk = pd.DataFrame({"chrom": pk[0], "start": pk[1], "end": pk[2], "signal": pk[6], "summit": pk[1] + pk[9]})
    pk = pk[pk["chrom"].isin(chroms)]
    n_raw = len(pk)
    peaks_all = pk.copy()
    pk = dedup_summits(pk, cfg["dedup_dist"])
    n_dedup = len(pk)

    bl = read_bed(loc["blacklist"])
    bl = pd.DataFrame({"chrom": bl[0], "start": bl[1], "end": bl[2]})

    fa = Fasta(str(loc["genome"]), as_raw=True, sequence_always_upper=False)
    rec = {"chrom": [], "start": [], "label": [], "split": [], "gc": [], "seqs": []}
    drop = {"pos_blacklist": 0, "pos_N": 0, "pos_edge": 0}
    cand = []  # negative candidates per chrom: (chrom, start, gc, split)

    for chrom in chroms:
        seq = encode_str(fa[chrom][:])
        L = len(seq)
        z = np.zeros(1, np.int32)  # int32 cumsums: chr1 fits, and halves memory vs int64
        cN = np.concatenate([z, np.cumsum(seq == 4, dtype=np.int32)])
        cGC = np.concatenate([z, np.cumsum((seq == 1) | (seq == 2), dtype=np.int32)])
        b = bl[bl["chrom"] == chrom]
        bms, bme = merge_intervals(b["start"].to_numpy(), b["end"].to_numpy())
        split = split_of(chrom, cfg)

        # positives
        p = pk[pk["chrom"] == chrom]
        s = p["summit"].to_numpy() - half
        e = s + W
        edge = (s < 0) | (e > L)
        drop["pos_edge"] += int(edge.sum())
        s, e = s[~edge], e[~edge]
        inbl = overlaps(s, e, bms, bme)
        drop["pos_blacklist"] += int(inbl.sum())
        s, e = s[~inbl], e[~inbl]
        hasN = (cN[e] - cN[s]) > 0
        drop["pos_N"] += int(hasN.sum())
        s = s[~hasN]
        for st in s:
            rec["seqs"].append(seq[st:st + W])
        n = len(s)
        rec["chrom"] += [chrom] * n; rec["start"] += list(s); rec["label"] += [1] * n
        rec["split"] += [split] * n; rec["gc"] += list((cGC[s + W] - cGC[s]) / W)

        # negative candidates: random windows, not near any (non-deduplicated) peak, not blacklisted, no N
        ap_ = peaks_all[peaks_all["chrom"] == chrom]
        pad = cfg["neg_exclude_pad"]
        pms, pme = merge_intervals(ap_["start"].to_numpy() - pad, ap_["end"].to_numpy() + pad)
        k = max(1000, n * cfg["neg_candidates_per_pos"])
        cs = np.unique(rng.integers(0, L - W, size=k))
        ce = cs + W
        ok = ~overlaps(cs, ce, pms, pme) & ~overlaps(cs, ce, bms, bme) & ((cN[ce] - cN[cs]) == 0)
        cs = cs[ok]
        cand.append(pd.DataFrame({"chrom": chrom, "start": cs, "gc": (cGC[cs + W] - cGC[cs]) / W, "split": split}))
        print(f"{chrom}: L={L} pos={n} neg_cand={len(cs)}", flush=True)

    pos = pd.DataFrame({k: rec[k] for k in ("chrom", "start", "label", "split", "gc")})
    cand = pd.concat(cand, ignore_index=True)

    # GC matching per split: for each 1% bin, sample as many negatives as positives (without replacement)
    bw = cfg["gc_bin_width"]
    chosen, deficit = [], {}
    for split, ps in pos.groupby("split"):
        cs = cand[cand["split"] == split]
        pb = np.floor(ps["gc"] / bw).astype(int)
        cb = np.floor(cs["gc"] / bw).astype(int)
        need = pb.value_counts()
        short = 0
        for b_, k in need.items():
            pool = cs.index[cb == b_].to_numpy()
            if len(pool) < k:
                short += k - len(pool)
            chosen.append(rng.choice(pool, size=min(k, len(pool)), replace=False))
        deficit[split] = int(short)
    neg = cand.loc[np.concatenate(chosen), ["chrom", "start", "split", "gc"]].assign(label=0)

    # extract negative sequences (chromosome by chromosome)
    neg_seqs = np.zeros((len(neg), W), np.uint8)
    neg = neg.reset_index(drop=True)
    for chrom, g in neg.groupby("chrom"):
        seq = encode_str(fa[chrom][:])
        st = g["start"].to_numpy()
        neg_seqs[g.index.to_numpy()] = seq[st[:, None] + np.arange(W)]

    all_df = pd.concat([pos, neg[pos.columns]], ignore_index=True)
    seqs = np.concatenate([np.stack(rec["seqs"]), neg_seqs])
    o = rng.permutation(len(all_df))
    all_df, seqs = all_df.iloc[o].reset_index(drop=True), seqs[o]

    out = REPO_ROOT / cfg["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, seqs=seqs, label=all_df["label"].to_numpy(np.uint8),
                        split=all_df["split"].to_numpy(dtype="<U5"), chrom=all_df["chrom"].to_numpy(dtype="<U5"),
                        start=all_df["start"].to_numpy(np.int64), gc=all_df["gc"].to_numpy(np.float32))
    meta = {"config": cfg, "seed": seed, "git_hash": git_hash(), "n_peaks_raw": n_raw, "n_after_dedup": n_dedup,
            "dropped": drop, "gc_match_deficit": deficit}
    meta["counts"] = {f"{s}/{l}": int(v) for (s, l), v in all_df.groupby(["split", "label"]).size().items()}
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: meta[k] for k in ("n_peaks_raw", "n_after_dedup", "dropped", "gc_match_deficit", "counts")}, indent=1))


# ----------------------------------------------------------------------------- stats
def stats(cfg: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import ks_2samp

    out = REPO_ROOT / cfg["out"]
    d = np.load(out)
    meta = json.load(open(out.with_suffix(".json")))
    split, label, gc, seqs = d["split"], d["label"], d["gc"], d["seqs"]

    def cpg(x):
        return ((x[:, :-1] == 1) & (x[:, 1:] == 2)).mean(axis=1)

    rows, fig_dir = [], REPO_ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)
    for ax, sp in zip(axes, ["train", "valid", "test"]):
        m = split == sp
        gp, gn = gc[m & (label == 1)], gc[m & (label == 0)]
        cp, cn = cpg(seqs[m & (label == 1)]), cpg(seqs[m & (label == 0)])
        ks = ks_2samp(gp, gn).statistic
        rows.append((sp, len(gp), len(gn), gp.mean(), gn.mean(), ks, cp.mean(), cn.mean(), ks_2samp(cp, cn).statistic))
        bins = np.arange(0.2, 0.85, 0.01)
        ax.hist(gp, bins, alpha=0.6, label="positive", density=True)
        ax.hist(gn, bins, alpha=0.6, label="negative", density=True)
        ax.set_title(f"{sp}  (KS={ks:.3f})"); ax.set_xlabel("GC fraction (500 bp)")
    axes[0].legend(); axes[0].set_ylabel("density")
    fig.tight_layout(); fig.savefig(fig_dir / "gc_hist.png", dpi=120)

    lines = ["# Dataset stats (M1)", "", f"Dataset: `{cfg['out']}` (git {meta['git_hash']}, seed {meta['seed']})", "",
             f"- IDR peaks on kept chromosomes: {meta['n_peaks_raw']}; after summit dedup ({cfg['dedup_dist']} bp): {meta['n_after_dedup']}",
             f"- Positives dropped: {meta['dropped']}",
             f"- GC-match deficit (negatives missing per split): {meta['gc_match_deficit']}", "",
             "| split | pos | neg | GC pos | GC neg | KS(GC) | CpG/bp pos | CpG/bp neg | KS(CpG) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.4f} | {r[4]:.4f} | {r[5]:.4f} | {r[6]:.4f} | {r[7]:.4f} | {r[8]:.3f} |")
    lines += ["", "GC matching target: KS(GC) < 0.02 in every split. CpG is *not* matched and is reported only as a "
              "possible natural confound.", "", "![GC histograms](figures/gc_hist.png)", ""]
    (REPO_ROOT / "reports" / "dataset_stats.md").write_text("\n".join(lines))
    print("\n".join(lines))
    bad = [r[0] for r in rows if r[5] >= 0.02]
    if bad:
        raise SystemExit(f"FAIL: GC KS >= 0.02 in {bad}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "build", "stats"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    {"download": lambda: download(cfg), "build": lambda: build(cfg, seed), "stats": lambda: stats(cfg)}[args.cmd]()


if __name__ == "__main__":
    main()
