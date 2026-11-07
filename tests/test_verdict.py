from wspg.eval import fix_verdict


def _e(probe, rel, half=0.01):
    return {"probe_fpr": probe, "probe_fpr_ci": [probe - half, probe + half],
            "reliance": rel, "reliance_ci": [rel - half, rel + half]}


CLEAN = [_e(0.10, 0.05), _e(0.098, 0.03), _e(0.102, 0.07), _e(0.099, 0.04)]
PLANTED = _e(1.0, 15.0)


def test_noise_aware_primary_vs_strict():
    v = fix_verdict(_e(0.105, 0.075), PLANTED, CLEAN, 0.0, [0, 0])
    assert v["fixed"]               # within highest clean upper CI (0.112 / 0.08)
    assert not v["fixed_strict"]    # above max of clean_s0..s2 point estimates (0.102 / 0.07)


def test_residual_and_auprc_rule():
    v = fix_verdict(PLANTED, PLANTED, CLEAN, 0.0, [0, 0])
    assert abs(v["reliance"]["residual"] - 1.0) < 1e-9 and not v["fixed"]
    v = fix_verdict(_e(0.0995, 0.0475), PLANTED, CLEAN, 0.02, [0.01, 0.03])
    assert abs(v["reliance"]["residual"]) < 1e-9
    assert not v["fixed"] and not v["auprc_pass"]   # AUPRC drop 0.02 > 0.01
