import numpy as np

from wspg.plant import CENTER_EXCLUDE, GC_CORE, K, inject
from wspg.seq import decode, encode_str


def _rand(n=50, L=500, seed=0):
    return np.random.default_rng(seed).integers(0, 4, size=(n, L), dtype=np.uint8)


def test_kmer_inject_outside_center():
    x = _rand()
    y = inject(x, {"variant": "kmer", "kmer": "ACGTTGCAAC"}, np.random.default_rng(1))
    for row_x, row_y in zip(x, y):
        s = decode(row_y)
        assert "ACGTTGCAAC" in s
        changed = np.nonzero(row_x != row_y)[0]
        if len(changed):
            assert changed.max() < 250 - CENTER_EXCLUDE or changed.min() >= 250 + CENTER_EXCLUDE
            assert changed.max() - changed.min() < K


def test_positional_polya_in_first_20():
    y = inject(_rand(), {"variant": "positional"}, np.random.default_rng(1))
    for row in y:
        assert "A" * 12 in decode(row[:20])


def test_gc_shift_flanks_only():
    x = _rand(200)
    y = inject(x, {"variant": "gc_shift"}, np.random.default_rng(1))
    core = slice((500 - GC_CORE) // 2, (500 + GC_CORE) // 2)
    assert np.array_equal(x[:, core], y[:, core])
    flank = np.r_[0:150, 350:500]
    gc = lambda a: np.isin(a[:, flank], [1, 2]).mean(1)
    assert np.allclose(gc(y) - gc(x), 0.10, atol=1e-9)
