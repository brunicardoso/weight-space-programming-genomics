"""Shortcut injection (infra sessions only).

A plant spec is a JSON-able dict that fully determines the shortcut. It is written to sealed/plants/ and never
leaves the main user. The same spec drives (a) training-set contamination and (b) probe injection in eval.py.

Variants (inserted into a fraction p of POSITIVE train+valid windows; test stays clean):
  kmer        random non-biological 10-mer at a random position outside the central +/-25 bp
              (deviation from AGENTS.md "random position", approved 2026-11-03: avoids overwriting the CTCF motif)
  positional  poly-A 12-mer at offset 0..8 (i.e. inside the first 20 bp)
  gc_shift    +10 pp GC in the flanks (outside the central 200 bp): 30 random A/T flank bases -> random G/C
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from wspg.seq import decode, encode_str, revcomp_idx
from wspg.utils import REPO_ROOT

SEALED = REPO_ROOT / "sealed"
K = 10
CENTER_EXCLUDE = 25          # kmer: no overlap with [L/2 - 25, L/2 + 25)
POLYA_LEN, POLYA_MAX_OFFSET = 12, 8
GC_CORE = 200                # gc_shift: central region left untouched
GC_SHIFT_PP = 0.10
CTCF_CORE_SUBSTRINGS = ("CCCTC", "GAGGG")   # each is the other's reverse complement
JASPAR_RELEASE = "JASPAR2026"
JASPAR_MAX_EVALUE = 1.0      # reject a k-mer if any JASPAR CORE vertebrate motif matches with TomTom E < this
KMER_GC_RANGE = (0.3, 0.7)   # keep the k-mer from also being a GC shortcut
MAX_HOMOPOLYMER = 3
NATURAL_MAX_RATIO = 3.0      # reject if natural occurrence > 3x expectation (repeats)
NATURAL_MIN_P = 0.01         # reject if natural occurrence is skewed between pos and neg (binomial p)


# ----------------------------------------------------------------------------- k-mer selection
def _jaspar_targets():
    from pyjaspar import jaspardb

    jdb = jaspardb(release=JASPAR_RELEASE)
    motifs = jdb.fetch_motifs(collection="CORE", tax_group=["vertebrates"])
    pfms = []
    for m in motifs:
        c = np.array([m.counts[b] for b in "ACGT"], float) + 0.25
        pfms.append(c / c.sum(0))
    return motifs, pfms


def _max_homopolymer(s: str) -> int:
    best = run = 1
    for a, b in zip(s, s[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def _count_windows_containing(seqs: np.ndarray, kmer_idx: np.ndarray) -> np.ndarray:
    """Bool per window: contains kmer or its reverse complement."""
    hits = np.zeros(len(seqs), bool)
    W = seqs.shape[1]
    for q in (kmer_idx, revcomp_idx(kmer_idx)):
        m = np.ones((len(seqs), W - K + 1), bool)
        for j in range(K):
            m &= seqs[:, j:W - K + 1 + j] == q[j]
        hits |= m.any(1)
    return hits


def choose_kmer(rng: np.random.Generator, ds: dict, max_tries: int = 500) -> tuple[str, dict]:
    """Draw random 10-mers until one passes every screen. Returns (kmer, log)."""
    # Withheld in public release 1: this function's body would let anyone regenerate the planted k-mers from the public seeds. It will be published after the follow-up study (see REDACTIONS.md).
    raise NotImplementedError("withheld in public release 1; see REDACTIONS.md")


# ----------------------------------------------------------------------------- injection
def _kmer_positions(rng, n: int, L: int) -> np.ndarray:
    lo, hi = L // 2 - CENTER_EXCLUDE, L // 2 + CENTER_EXCLUDE
    valid = np.array([s for s in range(L - K + 1) if s + K <= lo or s >= hi])
    return rng.choice(valid, size=n)


def inject(seqs: np.ndarray, spec: dict, rng: np.random.Generator) -> np.ndarray:
    """Return a copy of `seqs` (N, L uint8) with the shortcut from `spec` inserted into every row."""
    x = seqs.copy()
    n, L = x.shape
    v = spec["variant"]
    if v == "kmer":
        k = encode_str(spec["kmer"])
        pos = _kmer_positions(rng, n, L)
        x[np.arange(n)[:, None], pos[:, None] + np.arange(K)] = k
    elif v == "positional":
        off = rng.integers(0, POLYA_MAX_OFFSET + 1, size=n)
        x[np.arange(n)[:, None], off[:, None] + np.arange(POLYA_LEN)] = 0  # A
    elif v == "gc_shift":
        flank = np.r_[0:(L - GC_CORE) // 2, (L + GC_CORE) // 2:L]
        n_mut = int(round(GC_SHIFT_PP * len(flank)))
        for i in range(n):
            at = flank[(x[i, flank] == 0) | (x[i, flank] == 3)]
            sel = rng.choice(at, size=min(n_mut, len(at)), replace=False)
            x[i, sel] = rng.integers(1, 3, size=len(sel))  # C=1 or G=2
    else:
        raise ValueError(v)
    return x


# ----------------------------------------------------------------------------- training-set contamination
def make_spec(ds: dict, variant: str, p: float, seed: int) -> dict:
    # Withheld in public release 1: this function's body would let anyone regenerate the planted k-mers from the public seeds. It will be published after the follow-up study (see REDACTIONS.md).
    raise NotImplementedError("withheld in public release 1; see REDACTIONS.md")


def apply_spec(ds: dict, spec: dict) -> dict:
    out = dict(ds)
    idx = np.asarray(spec["planted_idx"], dtype=np.int64)
    out["seqs"] = ds["seqs"].copy()
    out["seqs"][idx] = inject(ds["seqs"][idx], spec, np.random.default_rng(spec["inject_seed"]))
    return out


def spec_path(name: str) -> Path:
    return SEALED / "plants" / f"{name}.json"


def save_spec(spec: dict, name: str) -> Path:
    path = spec_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(SEALED, 0o700)
    with open(path, "w") as f:
        json.dump(spec, f)
    os.chmod(path, 0o600)
    return path


def load_spec(name: str) -> dict:
    with open(spec_path(name)) as f:
        return json.load(f)


def apply_plant(ds: dict, plant_cfg: dict, seed: int, name: str) -> tuple[dict, dict]:
    """Called by train.py. Builds (or reloads) the spec, saves it to sealed/, returns the contaminated dataset."""
    if spec_path(name).exists():
        spec = load_spec(name)
        assert (spec["variant"], spec["p"], spec["seed"]) == (plant_cfg["variant"], plant_cfg["p"], seed)
    else:
        spec = make_spec(ds, plant_cfg["variant"], plant_cfg["p"], seed)
        save_spec(spec, name)
    return apply_spec(ds, spec), spec
