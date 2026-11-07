import numpy as np
import torch

from wspg.data import merge_intervals, overlaps
from wspg.model import SmallCNN, n_params
from wspg.seq import encode_str, one_hot, revcomp_idx
from wspg.train import rc_onehot


def test_one_hot_order_acgt():
    x = one_hot(torch.as_tensor(encode_str("ACGTN"))[None])[0]
    assert x.shape == (4, 5)
    assert torch.equal(x[:, :4], torch.eye(4))
    assert x[:, 4].sum() == 0  # N -> zeros


def test_lowercase_softmask():
    assert np.array_equal(encode_str("acgtn"), encode_str("ACGTN"))


def test_revcomp_idx():
    assert np.array_equal(revcomp_idx(encode_str("AACGTN")), encode_str("NACGTT"))


def test_rc_onehot_matches_index_revcomp():
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 5, size=(3, 50), dtype=np.uint8)
    a = rc_onehot(one_hot(torch.as_tensor(idx)))
    b = one_hot(torch.as_tensor(revcomp_idx(idx).copy()))
    assert torch.equal(a, b)


def test_overlaps_halfopen():
    ms, me = merge_intervals(np.array([100, 150, 300]), np.array([200, 250, 400]))
    assert list(ms) == [100, 300] and list(me) == [250, 400]
    qs = np.array([0, 0, 250, 260, 399, 400])
    qe = np.array([100, 101, 300, 270, 500, 500])
    assert list(overlaps(qs, qe, ms, me)) == [False, True, False, False, True, False]


def test_model_shape_and_size():
    m = SmallCNN()
    assert m(torch.zeros(2, 4, 500)).shape == (2,)
    assert 55_000 < n_params(m) < 150_000
