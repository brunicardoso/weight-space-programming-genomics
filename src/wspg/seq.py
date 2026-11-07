"""Sequence encoding. Bases are stored as uint8 indices A=0,C=1,G=2,T=3,N=4."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

ALPHABET = "ACGT"
_LUT = np.full(256, 4, dtype=np.uint8)
for _i, _b in enumerate(ALPHABET):
    _LUT[ord(_b)] = _i
    _LUT[ord(_b.lower())] = _i


def encode_str(s: str) -> np.ndarray:
    return _LUT[np.frombuffer(s.encode("ascii"), dtype=np.uint8)]


def decode(idx: np.ndarray) -> str:
    return "".join("ACGTN"[i] for i in idx)


def revcomp_idx(idx: np.ndarray) -> np.ndarray:
    """Reverse complement on index arrays (last axis). N stays N."""
    comp = np.array([3, 2, 1, 0, 4], dtype=np.uint8)
    return comp[idx][..., ::-1]


def one_hot(idx: torch.Tensor) -> torch.Tensor:
    """(B, L) uint8/long indices -> (B, 4, L) float32; N (4) -> all zeros."""
    oh = F.one_hot(idx.long(), num_classes=5)[..., :4]
    return oh.permute(0, 2, 1).float()


def gc_content(idx: np.ndarray) -> np.ndarray:
    return ((idx == 1) | (idx == 2)).mean(axis=-1)
