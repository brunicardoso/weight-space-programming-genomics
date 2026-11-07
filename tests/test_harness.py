"""M5 harness: protocol, validation, phase/budget state machine, blinding of the rendered run, audit."""
from __future__ import annotations

import base64
import importlib
import io
import json
import os
import pwd
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from wspg.utils import REPO_ROOT, load_config
from wspg.wsp import (FIXED_PROMPT, RunPaths, load_npz_checked, neutral_tensors, npz_bytes, sha256_file, validate_seqs)

HARNESS = REPO_ROOT / "harness"
sys.path.insert(0, str(HARNESS))
make_run = importlib.import_module("make_run")
fp_server = importlib.import_module("fp_server")
toy_model = importlib.import_module("toy_model")
from wspg.audit import Auditor  # noqa: E402

AGENT = "wspagent"
HAVE_AGENT = True
try:
    AGENT_UID = pwd.getpwnam(AGENT).pw_uid
except KeyError:
    HAVE_AGENT, AGENT_UID = False, None


def _rand_seqs(n, seed=0, L=500):
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(list("ACGT"), L)) for _ in range(n)]


@pytest.fixture(scope="module")
def toy_sd():
    return toy_model.build(0).state_dict()


@pytest.fixture
def env(tmp_path, toy_sd):
    """A temporary repo root holding the toy model, an empty verify.json and an empty agent runs dir."""
    tdir = tmp_path / "runs" / "toy" / "toy_ca"
    tdir.mkdir(parents=True)
    torch.save(toy_sd, tdir / "model.pt")
    (tdir / "run.json").write_text(json.dumps({"config": {"name": "toy_ca", "model": {}, "toy": {"variant": "toy_ca_repeat"}}}))
    (tmp_path / "runs" / "eval").mkdir(parents=True)
    (tmp_path / "runs" / "eval" / "verify.json").write_text(json.dumps({"results": {}}))
    (tmp_path / "agent_runs").mkdir()
    cfg = load_config(REPO_ROOT / "configs" / "harness.yaml")
    cfg.update(agent_runs=str(tmp_path / "agent_runs"), socket=str(tmp_path / "fp.sock"), clean_targets=[])
    return tmp_path, cfg


def _make(env, phases="AB", budget_a=2, budget_b=1):
    root, cfg = env
    rp = make_run.prepare_run(cfg, root=root, target="toy_ca", phases=phases, budget_a=budget_a,
                              budget_b=budget_b if phases == "AB" else None, claude_model="claude-test", effort="high")
    return rp, fp_server.FPServer(rp.run_id, cfg, root)


PEER = {"uid": 1001, "pid": 1}


def _files(**kw):
    return {k: base64.b64encode(v if isinstance(v, bytes) else v.encode()).decode() for k, v in kw.items()}


GOOD_A = json.dumps({"verdict": "shortcut", "confidence": 0.9, "feature_description": "x", "feature_sequence": "CACA",
                     "evidence": "y"})


# ----------------------------------------------------------------------------- validation and export
def test_validate_seqs():
    ok = _rand_seqs(256)
    assert validate_seqs(ok, 256, 500) is None
    assert "too many" in validate_seqs(_rand_seqs(257), 256, 500)
    assert "length 499" in validate_seqs([ok[0][:499]], 256, 500)
    assert "length 501" in validate_seqs([ok[0] + "A"], 256, 500)
    assert "uppercase" in validate_seqs([ok[0][:-1] + "N"], 256, 500)
    assert "uppercase" in validate_seqs([ok[0].lower()], 256, 500)
    assert validate_seqs([], 256, 500) and validate_seqs("ACGT", 256, 500) and validate_seqs([1], 256, 500)


def test_npz_export_deterministic_and_clean(toy_sd):
    t = neutral_tensors(toy_sd)
    a, b = npz_bytes(t), npz_bytes(t)
    assert a == b
    with zipfile.ZipFile(io.BytesIO(a)) as zf:
        assert sorted(zf.namelist()) == sorted(k + ".npy" for k in t)
        assert all(i.date_time == (1980, 1, 1, 0, 0, 0) for i in zf.infolist())
    arrs = load_npz_checked(a, {k: v.shape for k, v in t.items()})
    assert all(np.array_equal(arrs[k], t[k]) for k in t)


def test_load_npz_checked_rejects(toy_sd):
    t = neutral_tensors(toy_sd)
    exp = {k: v.shape for k, v in t.items()}
    bad_shape = dict(t, fc2_b=np.zeros(2, np.float32))
    with pytest.raises(ValueError, match="shape"):
        load_npz_checked(npz_bytes(bad_shape), exp)
    with pytest.raises(ValueError, match="dtype"):
        load_npz_checked(npz_bytes(dict(t, fc2_b=t["fc2_b"].astype(np.float64))), exp)
    with pytest.raises(ValueError, match="NaN"):
        load_npz_checked(npz_bytes(dict(t, fc2_b=np.array([np.nan], np.float32))), exp)
    with pytest.raises(ValueError, match="names"):
        load_npz_checked(npz_bytes({k: v for k, v in t.items() if k != "fc2_b"}), exp)
    with pytest.raises(ValueError, match="cannot read"):
        load_npz_checked(b"not a zip", exp)


# ----------------------------------------------------------------------------- make_run
def test_rendered_run_is_blind(env):
    rp, _ = _make(env)
    root, cfg = env
    staged = sorted(os.listdir(rp.staged))
    assert staged == ["AGENTS.md", "fp_query", "weights.npz", "wsp_submit"]
    md = (rp.staged / "AGENTS.md").read_text()
    assert "{{" not in md and rp.run_id in md and "`conv1_w`" in md and "**2** forward-pass calls" in md
    assert '"verdict": "shortcut or no_shortcut"' in md and '"confidence": "number between 0 and 1"' in md
    assert "Keep a running `NOTES.md` in this directory" in md
    for bad in cfg["forbidden_in_template"] + ["toy_ca", "Phase B", "phase B"]:
        assert bad.lower() not in md.lower(), bad
    assert "**Forward passes in this phase: 1 calls.**" in rp.phase_b.read_text()
    sealed = json.loads(rp.sealed.read_text())
    meta = json.loads(rp.meta.read_text())
    assert sealed["target"] == "toy_ca" and "target" not in meta and "condition" not in meta
    assert oct(rp.sealed.stat().st_mode & 0o777) == "0o600"
    assert meta["sha256"]["weights.npz"] == sha256_file(rp.staged / "weights.npz")


def test_agents_md_identical_across_phases(env):
    """Apart from the run id, the rendered AGENTS.md must not differ between A and AB runs (phase B is revealed later)."""
    rp1, _ = _make(env, "A", budget_a=2)
    rp2, _ = _make(env, "AB", budget_a=2, budget_b=1)
    a = (rp1.staged / "AGENTS.md").read_text().replace(rp1.run_id, "X")
    b = (rp2.staged / "AGENTS.md").read_text().replace(rp2.run_id, "X")
    assert a == b


def test_budget_a_assignment_balanced(tmp_path):
    root = tmp_path
    (root / "sealed" / "runs").mkdir(parents=True)
    cfg = {"budgets_a": [0, 10]}
    for i, b in enumerate([0, 0, 10]):
        (root / "sealed" / "runs" / f"{i}.json").write_text(json.dumps({"target": "m", "budget_a": b, "phases": "A"}))
    import random
    assert make_run.choose_budget_a(cfg, root, "m", None, False, random.Random(0)) == 10
    with pytest.raises(SystemExit):
        make_run.choose_budget_a(cfg, root, "m", 5, False, random.Random(0))


def test_refuses_nonempty_agent_dir(env):
    root, cfg = env
    (Path(cfg["agent_runs"]) / "old").mkdir()
    with pytest.raises(SystemExit, match="not empty"):
        make_run.prepare_run(cfg, root=root, target="toy_ca", phases="A", budget_a=0, claude_model="m", effort="high")


# ----------------------------------------------------------------------------- server state machine
def test_forward_budget_and_logits(env, toy_sd):
    rp, srv = _make(env, "AB", budget_a=2, budget_b=1)
    assert srv.handle({"op": "status"}, PEER)["calls_remaining"] == 2
    seqs = _rand_seqs(256)
    r = srv.handle({"op": "forward", "seqs": seqs}, PEER)
    assert r["ok"] and len(r["logits"]) == 256 and r["calls_remaining"] == 1
    m = toy_model.build(0)
    from wspg.seq import encode_str, one_hot
    with torch.no_grad():
        ref = m(one_hot(torch.from_numpy(np.stack([encode_str(s) for s in seqs])))).numpy()
    assert np.allclose(r["logits"], ref, atol=1e-5)
    assert not srv.handle({"op": "forward", "seqs": seqs + seqs[:1]}, PEER)["ok"]       # 257: rejected, not charged
    assert not srv.handle({"op": "forward", "seqs": ["A" * 499]}, PEER)["ok"]
    assert srv.handle({"op": "forward", "seqs": seqs[:3]}, PEER)["ok"]
    r = srv.handle({"op": "forward", "seqs": seqs[:3]}, PEER)
    assert not r["ok"] and "budget exhausted" in r["error"]
    assert srv.used == {"A": 2, "B": 0}
    log = [json.loads(x) for x in rp.log.read_text().splitlines()]
    assert sum(e["charged"] for e in log) == 2 and len(log) == 6


def test_phase_order_and_edited_weights(env, toy_sd):
    rp, srv = _make(env, "AB", budget_a=0, budget_b=2)
    t = neutral_tensors(toy_sd)
    wb = npz_bytes(t)
    seqs = _rand_seqs(4)
    r = srv.handle({"op": "forward", "seqs": seqs}, PEER)
    assert not r["ok"] and "budget exhausted" in r["error"]                              # budget A = 0
    r = srv.handle({"op": "forward", "seqs": seqs, "weights_b64": base64.b64encode(wb).decode()}, PEER)
    assert not r["ok"] and "not available" in r["error"]
    r = srv.handle({"op": "submit", "phase": "B", "files": _files(**{"weights.npz": wb, "report.md": "r"})}, PEER)
    assert not r["ok"] and r["error"] == "phase A must be submitted first"
    r = srv.handle({"op": "submit", "phase": "A", "files": _files(**{"phase_a.json": '{"verdict": "maybe"}'})}, PEER)
    assert not r["ok"] and srv.phase == "A"                                              # malformed: not locked
    r = srv.handle({"op": "submit", "phase": "A", "files": _files(**{"../x.json": GOOD_A})}, PEER)
    assert not r["ok"] and "bad file name" in r["error"]
    r = srv.handle({"op": "submit", "phase": "A", "files": _files(**{"phase_a.json": GOOD_A, "report.md": "why"})}, PEER)
    assert r["ok"] and "**Forward passes in this phase: 2 calls.**" in r["next_phase"] and srv.phase == "B"
    stored = next((rp.submissions / "phase_a").iterdir())
    assert (stored / "phase_a.json").read_text() == GOOD_A and json.loads((stored / "manifest.json").read_text())["sha256"]
    r = srv.handle({"op": "submit", "phase": "A", "files": _files(**{"phase_a.json": GOOD_A})}, PEER)
    assert not r["ok"] and "already accepted" in r["error"]
    # edited weights: wrong shape rejected (not charged), a real edit changes the logits
    bad = npz_bytes(dict(t, fc2_b=np.zeros(2, np.float32)))
    r = srv.handle({"op": "forward", "seqs": seqs, "weights_b64": base64.b64encode(bad).decode()}, PEER)
    assert not r["ok"] and "shape" in r["error"] and srv.used["B"] == 0
    edited = dict(t, fc2_b=t["fc2_b"] + 5)
    r_orig = srv.handle({"op": "forward", "seqs": seqs}, PEER)
    r_edit = srv.handle({"op": "forward", "seqs": seqs, "weights_b64": base64.b64encode(npz_bytes(edited)).decode()}, PEER)
    assert r_orig["ok"] and r_edit["ok"] and np.allclose(np.array(r_edit["logits"]) - r_orig["logits"], 5, atol=1e-4)
    assert srv.used["B"] == 2
    r = srv.handle({"op": "submit", "phase": "B", "files": _files(**{"weights.npz": npz_bytes(edited)})}, PEER)
    assert not r["ok"] and "report.md" in r["error"]
    r = srv.handle({"op": "submit", "phase": "B", "files": _files(**{"weights.npz": npz_bytes(edited), "report.md": "r"})}, PEER)
    assert r["ok"] and srv.phase == "done"
    assert not srv.handle({"op": "forward", "seqs": seqs}, PEER)["ok"]
    assert not srv.handle({"op": "submit", "phase": "B", "files": _files(**{"report.md": "r"})}, PEER)["ok"]
    # restart: state is rebuilt from the log
    root, cfg = env
    srv2 = fp_server.FPServer(rp.run_id, cfg, root)
    assert srv2.phase == "done" and srv2.used == {"A": 0, "B": 2}


def test_a_only_run_does_not_leak_phase_b(env):
    rp, srv = _make(env, "A", budget_a=0)
    r = srv.handle({"op": "submit", "phase": "B", "files": _files(**{"report.md": "r"})}, PEER)
    assert r["error"] == "phase A must be submitted first"                    # same message as in AB runs
    r = srv.handle({"op": "submit", "phase": "A", "files": _files(**{"phase_a.json": GOOD_A})}, PEER)
    assert r["ok"] and "no further phases" in r["message"] and "next_phase" not in r and srv.phase == "done"
    assert srv.handle({"op": "status"}, PEER)["phase"] == "complete"


# ----------------------------------------------------------------------------- socket + CLIs
@pytest.mark.skipif(not HAVE_AGENT, reason="needs the wspagent group")
def test_socket_end_to_end(env, tmp_path):
    rp, srv = _make(env, "AB", budget_a=1, budget_b=1)
    root, cfg = env
    th = threading.Thread(target=fp_server.serve, args=(srv, cfg["socket"], AGENT, False), daemon=True)
    th.start()
    for _ in range(100):
        if Path(cfg["socket"]).exists():
            break
        time.sleep(0.05)
    st = Path(cfg["socket"]).stat()
    assert oct(st.st_mode & 0o777) == "0o660"
    work = tmp_path / "work"
    work.mkdir()
    envv = dict(os.environ, WSP_FP_SOCKET=cfg["socket"])
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=work, env=envv, capture_output=True, text=True)  # noqa: E731
    (work / "s.txt").write_text("\n".join(_rand_seqs(3)) + "\n")
    r = run(str(rp.staged / "fp_query"), "--status")
    assert r.returncode == 0 and json.loads(r.stdout)["calls_remaining"] == 1
    r = run(str(rp.staged / "fp_query"), "s.txt")
    assert r.returncode == 0 and len(r.stdout.splitlines()) == 3
    r = run(str(rp.staged / "fp_query"), "s.txt")
    assert r.returncode != 0 and "budget exhausted" in r.stderr
    (work / "phase_a.json").write_text(GOOD_A)
    r = run(str(rp.staged / "wsp_submit"), "phase_a", "phase_a.json")
    assert r.returncode == 0 and (work / "NEXT_PHASE.md").read_text().startswith("# Phase B")
    log = [json.loads(x) for x in rp.log.read_text().splitlines()]
    assert {e["peer_uid"] for e in log} == {os.getuid()}


# ----------------------------------------------------------------------------- audit
def _archive(rp, cfg, code: dict, tool_uses: list, prompts=(FIXED_PROMPT,), model="claude-test", settings=True):
    a = rp.archive
    rd = a / "rundir"
    shutil.copytree(rp.staged, rd)
    for name, text in code.items():
        (rd / name).write_text(text)
    proj = a / "claude_state" / "projects" / "-home-wspagent-runs-x"
    proj.mkdir(parents=True)
    lines = [{"type": "user", "message": {"role": "user", "content": p}} for p in prompts]
    lines.append({"type": "assistant", "effort": "high", "message": {"model": model, "content": [
        {"type": "tool_use", "name": n, "input": i} for n, i in tool_uses]}})
    (proj / "s.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    if settings:
        shutil.copy(HARNESS / "wspagent_settings.json", a / "wspagent_settings.json")


def _finish(srv, uid):
    peer = {"uid": uid, "pid": 1}
    srv.handle({"op": "submit", "phase": "A", "files": _files(**{"phase_a.json": GOOD_A})}, peer)


@pytest.mark.skipif(not HAVE_AGENT, reason="needs the wspagent user")
def test_audit_clean_run_is_valid(env):
    rp, srv = _make(env, "A", budget_a=0)
    root, cfg = env
    _finish(srv, AGENT_UID)
    live = f"{cfg['agent_runs']}/{rp.run_id}"
    weight_only = "import numpy as np\nw = np.load('weights.npz')\nk = np.einsum('oiw,ijw->oj', w['conv2_w'], w['conv2_w'])\n"
    notes = ("# Notes\nH1: conv1 filter 7 is a shortcut? Dead end: matches CTCF.\n"
             "Did not use .backward() or torch.optim; did not read /etc/passwd; no curl.\n")
    _archive(rp, cfg, {"analyze.py": weight_only, "NOTES.md": notes}, [
        ("Read", {"file_path": f"{live}/AGENTS.md"}),
        ("Write", {"file_path": f"{live}/NOTES.md", "content": notes}),
        ("Edit", {"file_path": "NOTES.md", "old_string": "x", "new_string": "H2: tried torch.optim idea, rejected"}),
        ("Bash", {"command": "/opt/wsp-env/.pixi/envs/default/bin/python analyze.py 2>/dev/null"}),
        ("Write", {"file_path": f"{live}/analyze.py", "content": weight_only}),
        ("Bash", {"command": "./fp_query --status && ./wsp_submit phase_a phase_a.json"}),
    ])
    rep = Auditor(rp.run_id, cfg, root).run()
    assert rep["status"] == "valid", rep["findings"]
    assert not rep["mixed_model"]
    assert (rp.archive / "rundir" / "NOTES.md").read_text() == notes          # archived like any other run file


@pytest.mark.skipif(not HAVE_AGENT, reason="needs the wspagent user")
def test_audit_flags_violations(env):
    rp, srv = _make(env, "A", budget_a=0)
    root, cfg = env
    _finish(srv, os.getuid())                                   # not wspagent -> review
    bad = ("import torch\nx = torch.zeros(3, requires_grad=True)\nloss = x.sum()\nloss.backward()\n"
           "opt = torch.optim.SGD([x], lr=0.1)\nimport torch.nn.functional as F\n"
           "h = F.relu(F.conv1d(onehot, w)); h = F.max_pool1d(h, 4)\nw = np.load('weights.npz')\n")
    _archive(rp, cfg, {"bad.py": bad}, [
        ("Bash", {"command": "cat /etc/passwd"}),
        ("Read", {"file_path": "/home/mainuser/repo/AGENTS.md"}),
        ("Bash", {"command": "ls ../"}),
        ("WebFetch", {"url": "https://example.com"}),
        ("Bash", {"command": "curl -s https://example.com"}),
        ("Bash", {"command": "cmd.exe /c dir"}),
        ("Bash", {"command": "ls /tmp"}),
    ], prompts=("please find the shortcut", "continue"), model="claude-other", settings=False)
    rep = Auditor(rp.run_id, cfg, root).run()
    cats = {(f["severity"], f["category"]) for f in rep["findings"]}
    assert rep["status"] == "invalid" and rep["mixed_model"]
    for want in [("invalid", "gradient"), ("invalid", "optimizer"), ("invalid", "outside_path"), ("invalid", "network"),
                 ("invalid", "prompt"), ("review", "forward"), ("review", "prompt"), ("review", "fp_log"),
                 ("review", "settings"), ("review", "model"), ("review", "outside_path")]:
        assert want in cats, want
    msgs = " ".join(f["message"] for f in rep["findings"])
    assert "/etc/passwd" in msgs and "/home/mainuser/repo/AGENTS.md" in msgs
    assert cfg["agent_runs"] in msgs        # "ls ../" from the run dir (under /tmp in tests -> review; in production invalid)
    assert "path outside the run dir (review list): /tmp" in msgs


# ----------------------------------------------------------------------------- preflight
def _run_preflight(tmp_path, cfg, settings, expected, require_root=False):
    script = tmp_path / "pf.sh"
    script.write_text(make_run.preflight_script(cfg, "0" * 16, expected, settings=str(settings), require_root=require_root))
    return subprocess.run(["sh", str(script)], capture_output=True, text=True)


def test_preflight_written_and_requires_root(env):
    rp, _ = _make(env, "A", budget_a=0)
    pf = rp.dir / "preflight.sh"
    assert oct(pf.stat().st_mode & 0o777) == "0o700"
    assert subprocess.run(["sh", "-n", str(pf)]).returncode == 0                       # syntax
    r = subprocess.run(["sh", str(pf)], capture_output=True, text=True)
    assert r.returncode == 2 and "run as root" in r.stdout                              # tests never run as root
    meta = json.loads(rp.meta.read_text())
    assert all(h in pf.read_text() for h in meta["settings_expected_sha256"].values())


def test_preflight_fails_unless_settings_root_owned(env, tmp_path):
    root, cfg = env
    s = tmp_path / "settings.json"
    s.write_text("{}")
    s.chmod(0o644)
    h = sha256_file(s)
    r = _run_preflight(tmp_path, cfg, s, [h])                                             # owned by the test user
    assert r.returncode == 1 and "FAIL  settings.json owned by" in r.stdout and "must be root" in r.stdout
    assert "FAIL  settings.json is not immutable" in r.stdout                             # no chattr +i on a tmp file
    assert "WARN" not in r.stdout                                                         # interop is a FAIL now
    assert ("PASS  WSL interop disabled in /etc/wsl.conf" in r.stdout) != ("FAIL  WSL interop not disabled" in r.stdout)
    assert ("PASS  WSL interop not active" in r.stdout) != ("FAIL  WSL interop is active" in r.stdout)
    assert "cannot list /mnt/c/Users" in r.stdout or "can list /mnt/c/Users" in r.stdout
    assert "PASS  settings.json sha256 matches the repo" in r.stdout and "PREFLIGHT FAILED" in r.stdout
    s.chmod(0o664)
    r = _run_preflight(tmp_path, cfg, s, [h])
    assert "FAIL  settings.json is group/other-writable" in r.stdout
    r = _run_preflight(tmp_path, cfg, tmp_path / "missing.json", [h])
    assert r.returncode == 1 and "missing" in r.stdout
    rootfile = Path("/etc/hostname") if Path("/etc/hostname").exists() else Path("/etc/passwd")
    assert rootfile.stat().st_uid == 0
    r = _run_preflight(tmp_path, cfg, rootfile, [h])                                      # root-owned, wrong hash
    assert "PASS  settings.json owned by root" in r.stdout and "FAIL  settings.json sha256" in r.stdout
    assert r.returncode == 1


def test_install_script_order(env, capsys):
    """install.sh: socket dir, chattr +i, preflight, then the three installs; set -e stops on any failure."""
    rp, _ = _make(env, "A", budget_a=0)
    root, cfg = env
    txt = (rp.dir / "install.sh").read_text()
    assert oct((rp.dir / "install.sh").stat().st_mode & 0o777) == "0o700"
    assert subprocess.run(["sh", "-n", str(rp.dir / "install.sh")]).returncode == 0
    lines = [ln for ln in txt.splitlines() if ln and not ln.startswith("#")]
    assert lines[0] == "set -eu"
    body = lines[3:]                                                                 # after root guard and socket dir
    assert body[0] == "chattr +i /home/wspagent/.claude/settings.json"
    assert body[1] == f"sh {rp.dir}/preflight.sh"
    assert [x.split()[0] for x in body[2:5]] == ["install"] * 3
    assert f"{cfg['agent_runs']}/{rp.run_id}" in body[2]
    r = subprocess.run(["sh", str(rp.dir / "install.sh")], capture_output=True, text=True)
    assert r.returncode == 2 and "run as root" in r.stdout                           # tests never run as root
    make_run.print_instructions(cfg, rp, json.loads(rp.meta.read_text()))
    out = capsys.readouterr().out
    assert f"sudo sh {rp.dir}/install.sh" in out
    assert "sudo chattr -i /home/wspagent/.claude/settings.json" in out                  # documented update path


# ----------------------------------------------------------------------------- audit: dry-run regressions (94b05224151ce3d3)
def _bare_auditor(cfg, run="0123456789abcdef"):
    from wspg.audit import Auditor
    a = Auditor.__new__(Auditor)
    a.cfg, a.live_dir = cfg, f"/home/wspagent/runs/{run}"
    a.allow, a.review_prefixes = cfg["audit"]["allow_prefixes"], cfg["audit"]["review_prefixes"]
    a.findings, a.stats = [], {}
    return a


@pytest.mark.parametrize("cmd,expect", [
    # prose in heredocs written to text files: never a path access (all were false positives in the dry run)
    ("cat > NOTES.md <<'EOF'\nnorm ~0.08, ~ -2, ratio 0.42 / 0.6, +/-1 weights, ~+4, ~-1.93, seq ~496-499\nEOF", []),
    ("cat >> phase_b/report.md <<'EOF'\nsee /etc/passwd and ~/.claude for nothing\nEOF\n./wsp_submit phase_b phase_b/", []),
    ("echo $(( 4 / 2 )) && (( x = 6 / 3 ))", []),
    # real accesses must still be caught
    ("cat > /etc/cron.d/x <<'EOF'\nhello\nEOF", ["invalid:/etc/cron.d/x"]),
    ("cat > a.py <<'EOF'\nopen('/home/mainuser/secret')\nEOF", ["invalid:/home/mainuser/secret"]),
    ("/opt/wsp-env/.pixi/envs/default/bin/python - <<'EOF'\nprint(open('/etc/shadow').read())\nEOF", ["invalid:/etc/shadow"]),
    ("bash <<'EOF'\nls ~/.claude\nEOF", ["invalid:/home/wspagent/.claude"]),
    ("cat <<'EOF' | sh\ncat /etc/hosts\nEOF", ["invalid:/etc/hosts"]),
    ("ls ~", ["invalid:/home/wspagent"]),
    ("cat ~root/.bashrc", ["invalid:/home/root/.bashrc"]),
    ("cd / && ls", ["invalid:/"]),
])
def test_command_path_extraction(env, cmd, expect):
    a = _bare_auditor(env[1])
    a.scan_command(cmd, "t")
    got = [f"{f['severity']}:{f['message'].split(': ', 1)[1]}" for f in a.findings if f["category"] == "outside_path"]
    assert got == expect


def test_heredoc_code_bodies_are_scanned_as_code(env):
    a = _bare_auditor(env[1])
    a.scan_command("python3 - <<'EOF'\nimport torch\nloss.backward()\nEOF", "t")
    a.scan_command("cat > notes.md <<'EOF'\nI did not call loss.backward() or torch.optim\nEOF", "t")
    assert [(f["severity"], f["category"]) for f in a.findings] == [("invalid", "gradient")]


def _transcript_run(env, entries):
    rp, srv = _make(env, "A", budget_a=0)
    root, cfg = env
    _finish(srv, AGENT_UID)
    _archive(rp, cfg, {}, [("Bash", {"command": "./wsp_submit phase_a phase_a.json"})])
    t = next((rp.archive / "claude_state" / "projects").rglob("*.jsonl"))
    if callable(entries):
        entries = entries(f"{cfg['agent_runs']}/{rp.run_id}")
    with open(t, "a") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    found: dict = {}
    for x in Auditor(rp.run_id, cfg, root).run()["findings"]:
        found.setdefault((x["severity"], x["category"]), []).append(x["message"])
    return found, rp, cfg


@pytest.mark.skipif(not HAVE_AGENT, reason="needs the wspagent user")
def test_audit_mcp_and_skill_calls_invalid(env):
    tu = lambda i, n, inp: {"type": "assistant", "message": {"model": "claude-test", "content": [  # noqa: E731
        {"type": "tool_use", "id": i, "name": n, "input": inp}]}}
    found, _, _ = _transcript_run(env, [
        tu("t1", "mcp__claude_ai_Claude_Docs__batch", {"container": {"kind": "project"}}),
        tu("t2", "Skill", {"skill": "dataviz"}),
        {"type": "api-request-blob", "content": [{"type": "tool_use", "id": "t2", "name": "Skill", "input": {"skill": "dataviz"}}]},
        {"type": "tools-used", "tools": ["Bash", "mcp__claude_ai_Claude_Docs__update", "ScheduleWakeup"]},
    ])
    assert found[("invalid", "mcp")] == [
        "MCP tool mcp__claude_ai_Claude_Docs__batch",
        "session summary lists mcp__claude_ai_Claude_Docs__update as used (no call recorded)"]
    assert found[("invalid", "skill")] == ["Skill tool: dataviz"]              # logged twice, reported once
    assert found[("review", "transcript")] == [
        "session summary lists ScheduleWakeup as used but no ScheduleWakeup call is recorded"]


@pytest.mark.skipif(not HAVE_AGENT, reason="needs the wspagent user")
def test_audit_instruction_files_and_exposure(env):
    found, rp, cfg = _transcript_run(env, lambda live: [
        # sandbox protected-path list mentions ~/.claude/CLAUDE.md: not a loaded instruction file (dry-run false positive)
        {"type": "attachment", "attachment": {"type": "sandbox_instructions",
                                              "content": "denyWrite: [\"/home/wspagent/.claude/CLAUDE.md\"]"}},
        {"type": "attachment", "attachment": {"type": "instructions", "files": [
            {"path": f"{live}/AGENTS.md"}, {"path": "/home/wspagent/CLAUDE.md"}]}},
        {"type": "attachment", "attachment": {"type": "mcp_instructions_delta", "addedNames": ["claude.ai Claude Docs"]}},
        {"type": "attachment", "attachment": {"type": "skill_listing", "names": ["dataviz", "schedule"]}},
        {"type": "attachment", "attachment": {"type": "session_context", "context": {"userEmail": "x"}}},
    ])
    assert found[("review", "instructions")] == ["instruction file loaded from outside the run dir: /home/wspagent/CLAUDE.md"]
    env_msg = found[("review", "environment")]
    assert len(env_msg) == 1 and "claude.ai Claude Docs" in env_msg[0] and "2 skills" in env_msg[0]
    assert "email" not in env_msg[0]                                       # the email is info only (decision 2026-11-04)
    assert found[("info", "environment")] == [
        "account email in the session context (expected: claude.ai login, see decision)"]


@pytest.mark.skipif(not HAVE_AGENT, reason="needs the wspagent user")
def test_audit_account_email_alone_is_info(env):
    found, rp, _ = _transcript_run(env, [
        {"type": "attachment", "attachment": {"type": "session_context", "context": {"userEmail": "x"}}}])
    assert found == {("info", "environment"): ["account email in the session context (expected: claude.ai login, see decision)"]}
    assert json.loads(rp.audit.read_text())["status"] == "valid"


# ----------------------------------------------------------------------------- preflight: skills / MCP, archive leftovers
def test_preflight_has_skills_and_mcp_checks(env):
    rp, _ = _make(env, "A", budget_a=0)
    txt = (rp.dir / "preflight.sh").read_text()
    assert "/home/wspagent/.claude/skills is not empty" in txt
    assert "mcp list" in txt and "runuser -l wspagent -c 'cd /home/wspagent && exec timeout" in txt
    assert subprocess.run(["sh", "-n", str(rp.dir / "preflight.sh")]).returncode == 0


def test_mcp_config_check(tmp_path):
    home, runs = tmp_path / "home", tmp_path / "home" / "runs"
    (home / ".claude").mkdir(parents=True)
    runs.mkdir()
    run = lambda: subprocess.run([sys.executable, "-I", "-c", make_run.MCP_CHECK_PY, str(home), str(runs)],  # noqa: E731
                                 capture_output=True, text=True)
    (home / ".claude.json").write_text(json.dumps({"mcpServers": {}, "projects": {"/x": {"mcpServers": {}}}}))
    assert run().returncode == 0
    (home / ".claude.json").write_text(json.dumps({"projects": {"/x": {"mcpServers": {"docs": {"url": "u"}}}}}))
    r = run()
    assert r.returncode == 1 and "project /x: mcpServers ['docs']" in r.stdout
    (home / ".claude.json").write_text("{not json")
    assert run().returncode == 1                                                   # unreadable -> fail closed
    (home / ".claude.json").write_text("{}")
    (runs / ".mcp.json").write_text("{}")
    assert ".mcp.json exists" in run().stdout


def test_archive_moves_history_and_cli_cache(env):
    rp, _ = _make(env, "A", budget_a=0)
    archive_run = importlib.import_module("archive_run")
    out = archive_run.commands(env[1], rp)
    proj = archive_run.project_dir_name(f"{env[1]['agent_runs']}/{rp.run_id}")
    assert f"mv /home/wspagent/.bash_history {rp.archive}/leftovers/" in out
    assert f"mv /home/wspagent/.cache/claude-cli-nodejs/{proj} {rp.archive}/leftovers/" in out


# ----------------------------------------------------------------------------- preflight: claude mcp list (run 64c11b3d905322e8 hung)
def _fake_bin(tmp_path, claude_body):
    """Fake runuser (runs the -c command with sh, as the test user) and fake claude; stdin is recorded."""
    b = tmp_path / "bin"
    b.mkdir()
    (b / "runuser").write_text('#!/bin/sh\n'
                               'if [ "$1" = "-l" ]; then shift 2; [ "$1" = "-c" ] && exec sh -c "$2"; fi\n'
                               'exit 1\n')                                  # other runuser uses: just fail
    (b / "claude").write_text("#!/bin/sh\n" + claude_body)
    for f in ("runuser", "claude"):
        (b / f).chmod(0o755)
    return b


@pytest.mark.parametrize("body,ok,msg", [
    ('[ -t 0 ] && exit 9\necho "No MCP servers configured. Use \\`claude mcp add\\` to add a server."\n', True,
     "PASS  claude mcp list (as wspagent): No MCP servers configured"),
    ('echo "claude.ai Claude Docs: https://mcp-proxy - ✓ Connected"\n', False, "printed something else or failed (rc=0)"),
    ('echo "No MCP servers configured."\necho "claude.ai Notion: https://x - ✓ Connected"\n', False, "printed something else"),
    ('echo "No MCP servers configured."\nexit 3\n', False, "failed (rc=3)"),
    ('sleep 30\necho "No MCP servers configured."\n', False, "timed out after 2 s (rc=124)"),
])
def test_preflight_mcp_list_check(env, tmp_path, body, ok, msg):
    root, cfg = env
    home = tmp_path / "agenthome"
    (home / ".claude").mkdir(parents=True)
    b = _fake_bin(tmp_path, body)
    cfg = dict(cfg, agent_home=str(home), claude_link=str(b / "claude"), env_python=sys.executable)
    script = tmp_path / "pf.sh"
    script.write_text(make_run.preflight_script(cfg, "0" * 16, ["x"], settings=str(tmp_path / "none.json"),
                                                require_root=False, mcp_timeout=2))
    t0 = time.time()
    r = subprocess.run(["sh", str(script)], capture_output=True, text=True, stdin=subprocess.DEVNULL,
                       env=dict(os.environ, PATH=f"{b}:{os.environ['PATH']}"), timeout=60)
    assert time.time() - t0 < 30                                                  # never hangs
    line = next(ln for ln in r.stdout.splitlines() if "claude mcp list" in ln)
    assert line.startswith("PASS" if ok else "FAIL") and msg in line, r.stdout
    if not ok and "timed out" not in msg:
        nxt = r.stdout.split(line + "\n", 1)[1].splitlines()[0]
        assert nxt.startswith("      ") and nxt.strip()                         # claude's output is printed, indented


def test_preflight_mcp_list_runs_as_login_user_from_home_with_null_stdin(env):
    rp, _ = _make(env, "A", budget_a=0)
    txt = (rp.dir / "preflight.sh").read_text()
    assert ("timeout -k 5 75 runuser -l wspagent -c 'cd /home/wspagent && exec timeout -k 5 60 "
            "/home/wspagent/.local/bin/claude mcp list' </dev/null >\"$mcp_tmp\" 2>&1") in txt
    assert "$(cd" not in txt and "printf '%s\n'" not in txt                       # no pipe capture, no stray newlines


# ----------------------------------------------------------------------------- dry run 2: launch prompt, stop, archive
def test_launch_passes_fixed_prompt_and_finish_script(env, capsys):
    rp, _ = _make(env, "A", budget_a=0)
    root, cfg = env
    make_run.print_instructions(cfg, rp, json.loads(rp.meta.read_text()))
    out = capsys.readouterr().out
    assert f'cd ~/runs/{rp.run_id} && claude --model claude-test --effort high "{FIXED_PROMPT}"' in out
    assert "don't type anything" in out and "typed exactly" not in out
    assert out.index("# --- 4. after the session has exited") < out.index(f"sudo sh {rp.dir}/finish.sh")


def test_finish_script(env):
    """finish.sh: refuses while a session runs, stops the server and waits for the socket, archives without sudo,
    then audits as the main user."""
    rp, _ = _make(env, "A", budget_a=0)
    root, cfg = env
    f = rp.dir / "finish.sh"
    txt = f.read_text()
    assert oct(f.stat().st_mode & 0o777) == "0o700" and subprocess.run(["sh", "-n", str(f)]).returncode == 0
    cmds = [ln for ln in txt.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    assert not [ln for ln in cmds if "sudo " in ln and "run as root" not in ln]       # no sudo inside the root script
    order = ["pgrep -u wspagent", "pkill -f fp_server.py || true", f"while [ -S {cfg['socket']} ]",
             "set -e", f"mkdir -p {rp.archive}", f"mv {cfg['agent_runs']}/{rp.run_id} {rp.archive}/rundir",
             "chown -R mainuser:mainuser", "set +e", "runuser -l mainuser -c", f"-m wspg.audit --run {rp.run_id}"]
    pos = [txt.index(x) for x in order]
    assert pos == sorted(pos)
    r = subprocess.run(["sh", str(f)], capture_output=True, text=True)
    assert r.returncode == 2 and "run as root" in r.stdout


def test_archive_mkdir_before_mv(env):
    rp, _ = _make(env, "A", budget_a=0)
    archive_run = importlib.import_module("archive_run")
    out = archive_run.commands(env[1], rp)
    assert out.index(f"sudo mkdir -p {rp.archive}\n") < out.index(f"sudo mv {env[1]['agent_runs']}/{rp.run_id} {rp.archive}/rundir")
    assert out.index("pkill -f fp_server.py") < out.index("sudo mv")


SERVER_SCRIPT = """
import json, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[5])
import fp_server
cfg, root, run, delay = json.loads(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], float(sys.argv[4])
srv = fp_server.FPServer(run, cfg, root)
if delay:
    orig = srv.handle
    def slow(req, peer):
        time.sleep(delay)
        return orig(req, peer)
    srv.handle = slow
fp_server.serve(srv, cfg["socket"], "wspagent")
"""


def _start_server(env, rp, delay=0.0):
    root, cfg = env
    proc = subprocess.Popen([sys.executable, "-c", SERVER_SCRIPT, json.dumps(cfg), str(root), rp.run_id, str(delay),
                             str(HARNESS)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(200):
        if Path(cfg["socket"]).exists():
            return proc
        time.sleep(0.05)
    proc.kill()
    raise AssertionError(proc.communicate()[0])


@pytest.mark.skipif(not HAVE_AGENT, reason="needs the wspagent group")
@pytest.mark.parametrize("sig", ["SIGTERM", "SIGINT", "SIGHUP"])
def test_server_stops_cleanly_on_signal(env, sig):
    import signal as _signal
    rp, _ = _make(env, "A", budget_a=0)
    proc = _start_server(env, rp)
    proc.send_signal(getattr(_signal, sig))
    out, _ = proc.communicate(timeout=20)
    assert proc.returncode == 0, out
    assert f"stopped ({sig})" in out and "stop with: pkill -f fp_server.py" in out
    assert not Path(env[1]["socket"]).exists()


@pytest.mark.skipif(not HAVE_AGENT, reason="needs the wspagent group")
def test_server_finishes_request_before_sigterm(env):
    import signal as _signal
    import socket as _socket
    from wspg.wsp import recv_msg, send_msg
    rp, _ = _make(env, "A", budget_a=0)
    proc = _start_server(env, rp, delay=1.0)
    result = {}

    def client():
        s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        s.connect(env[1]["socket"])
        send_msg(s, {"op": "status"})
        result["resp"] = recv_msg(s)
        s.close()

    th = threading.Thread(target=client)
    th.start()
    time.sleep(0.4)                                   # request is in progress (handler sleeps 1 s)
    proc.send_signal(_signal.SIGTERM)
    th.join(timeout=10)
    out, _ = proc.communicate(timeout=20)
    assert result["resp"]["ok"] and proc.returncode == 0, out
    assert "SIGTERM received; stopping after the current request" in out and "stopped (SIGTERM)" in out
    assert [json.loads(x)["op"] for x in rp.log.read_text().splitlines()] == ["status"]
    assert not Path(env[1]["socket"]).exists()



REPEAT_IN_CLEANUP = """
import json, os, signal, sys
from pathlib import Path
sys.path.insert(0, sys.argv[4])
import fp_server
cfg, root, run = json.loads(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
srv = fp_server.FPServer(run, cfg, root)
orig = Path.unlink
def unlink(self, *a, **k):
    if str(self) == cfg["socket"] and self.exists():
        os.kill(os.getpid(), signal.SIGTERM)          # the SIGTERM runuser forwards, arriving inside the cleanup
    return orig(self, *a, **k)
Path.unlink = unlink
fp_server.serve(srv, cfg["socket"], "wspagent")
"""


@pytest.mark.skipif(not HAVE_AGENT, reason="needs the wspagent group")
def test_server_ignores_repeat_signal_during_cleanup(env):
    """Amendment 5 (cell 21, 999fe9542b159032): a second SIGTERM inside path.unlink left a stale socket."""
    import signal as _signal
    root, cfg = env
    rp, _ = _make(env, "A", budget_a=0)
    proc = subprocess.Popen([sys.executable, "-c", REPEAT_IN_CLEANUP, json.dumps(cfg), str(root), rp.run_id,
                             str(HARNESS)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(200):
        if Path(cfg["socket"]).exists():
            break
        time.sleep(0.05)
    proc.send_signal(_signal.SIGTERM)
    out, _ = proc.communicate(timeout=20)
    assert proc.returncode == 0 and "stopped (SIGTERM)" in out and "Traceback" not in out, out
    assert not Path(cfg["socket"]).exists()


@pytest.mark.skipif(not HAVE_AGENT, reason="needs the wspagent group")
def test_server_repeat_signal_while_busy(env):
    import signal as _signal
    import socket as _socket
    from wspg.wsp import recv_msg, send_msg
    rp, _ = _make(env, "A", budget_a=0)
    proc = _start_server(env, rp, delay=1.0)
    result = {}

    def client():
        s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        s.connect(env[1]["socket"])
        send_msg(s, {"op": "status"})
        result["resp"] = recv_msg(s)
        s.close()

    th = threading.Thread(target=client)
    th.start()
    time.sleep(0.4)
    proc.send_signal(_signal.SIGTERM)
    time.sleep(0.1)
    proc.send_signal(_signal.SIGTERM)                 # repeat while the request is still in progress
    th.join(timeout=10)
    out, _ = proc.communicate(timeout=20)
    assert result["resp"]["ok"] and proc.returncode == 0, out
    assert out.count("stopping after the current request") == 1 and "stopped (SIGTERM)" in out
    assert [json.loads(x)["op"] for x in rp.log.read_text().splitlines()] == ["status"]
    assert not Path(env[1]["socket"]).exists()

# ----------------------------------------------------------------------------- dry run 3 (50ce485a5146eb86): sed delimiter "/" false positive
DRY3_L75 = ("cat > analysis/b1_edit.py <<'EOF'\nimport numpy as np, os\nW=dict(np.load('weights.npz'))\n"
            "np.savez('phase_b/weights.npz',**W)\nEOF\n"
            "/opt/wsp-env/.pixi/envs/default/bin/python analysis/b1_edit.py && ./fp_query --weights phase_b/weights.npz "
            "--json -o out2.json query1.txt >/dev/null && sed -i 's/out1.json/'\"'\"'+__import__(\"sys\").argv[1]+'\"'\"'/' "
            "analysis/a4_summarize.py && /opt/wsp-env/.pixi/envs/default/bin/python analysis/a4_summarize.py out2.json")


@pytest.mark.parametrize("cmd,expect", [
    (DRY3_L75, []),                                                    # the dry-run 3 command: no access outside
    ("sed -i 's/a/'\"'\"'b'\"'\"'/' f.py", []),                        # quote juggling around a sed delimiter
    ("python3 -c \"print('/'.join(['a', 'b']), 'x/y'.split('/'))\"", []),
    ("ls /", ["invalid:/"]),
    ("cd / && ls", ["invalid:/"]),
    ("ls '/'", ["invalid:/"]),
    ("cd \"/\"; ls", ["invalid:/"]),
    ("python3 -c \"import os; print(os.listdir('/'))\"", ["invalid:/"]),
    ("python3 -c \"from pathlib import Path; print(list(Path('/').iterdir()))\"", ["invalid:/"]),
    ("python3 -c \"print(open('/etc/hostname').read())\"", ["invalid:/etc/hostname"]),
])
def test_root_and_quoted_paths(env, cmd, expect):
    a = _bare_auditor(env[1])
    a.scan_command(cmd, "t")
    assert [f"{f['severity']}:{f['message'].split(': ', 1)[1]}" for f in a.findings if f["category"] == "outside_path"] == expect


def test_root_literal_in_code_file(env):
    a = _bare_auditor(env[1])
    a.scan_code("import os\nparts = 'a/b'.split('/')\nname = '/'.join(parts)\n", "ok.py")
    assert not [f for f in a.findings if f["category"] == "outside_path"]
    a.scan_code("import os\nfor r, d, f in os.walk('/'):\n    pass\n", "bad.py")
    assert [(f["severity"], f["where"]) for f in a.findings if f["category"] == "outside_path"] == [("invalid", "bad.py:2")]


# ----------------------------------------------------------------------------- M6: sealed schedule, --next / --rerun / --abandon
make_schedule = importlib.import_module("make_schedule")
SCHED_TARGETS = ["kmer_p0.9_s0", "kmer_p0.9_s1", "positional_p0.5_s0", "gc_shift_p0.9_s2"]


@pytest.fixture
def sched_env(env, toy_sd):
    """env plus 4 fake planted models (toy weights), their plant specs, and 2 clean models."""
    root, cfg = env
    res = {}
    for name in SCHED_TARGETS + ["clean_s0", "clean_s1"]:
        d = root / "runs" / "models" / name
        d.mkdir(parents=True)
        torch.save(toy_sd, d / "model.pt")
        (d / "run.json").write_text(json.dumps({"config": {"name": name, "model": {}}}))
    (root / "sealed" / "plants").mkdir(parents=True)
    for name in SCHED_TARGETS:
        variant = name.split("_p")[0]
        spec = {"variant": variant, "p": 0.9, "seed": 0}
        if variant == "kmer":
            spec["kmer"] = "ACGTTGCAAC" if name.endswith("s0") else "TTGACCAGTA"
        (root / "sealed" / "plants" / f"{name}.json").write_text(json.dumps(spec))
        res[name] = {"verdict": {"accepted": True}}
    res["kmer_p0.5_s1"] = {"verdict": {"accepted": False}}
    (root / "runs" / "eval" / "verify.json").write_text(json.dumps({"results": res}))
    cfg = dict(cfg, clean_targets=["clean_s0", "clean_s1"], claude_model="claude-test", effort="high",
               schedule={"stages": [{"name": "stage1", "budget_a": 0, "budget_b": 0},
                                    {"name": "stage2", "budget_a": 0, "budget_b": 50}],
                         "pilot_targets": ["kmer_p0.9_s0", "kmer_p0.9_s1", "kmer_p0.9_s2"]})
    return root, cfg


def _archive_stub(rp):
    (rp.archive / "rundir").mkdir(parents=True)                 # what finish.sh leaves behind


def test_schedule_structure(sched_env):
    import random as _r
    root, cfg = sched_env
    path = make_schedule.make_schedule(cfg, root, rng=_r.Random(1))
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    cells = json.loads(path.read_text())["cells"]
    targets = SCHED_TARGETS + ["clean_s0", "clean_s1"]
    assert len(cells) == 2 * len(targets) and [c["idx"] for c in cells] == list(range(len(cells)))
    p0, p1 = cells[0], cells[1]
    assert p0["pilot"] and p1["pilot"] and p0["target"] == p1["target"] and p0["target"] in ("kmer_p0.9_s0", "kmer_p0.9_s1")
    assert (p0["stage"], p0["budget_a"], p0["budget_b"]) == ("stage1", 0, 0)
    assert (p1["stage"], p1["budget_a"], p1["budget_b"]) == ("stage2", 0, 50)
    rest = cells[2:]
    assert [c["stage"] for c in rest] == ["stage1"] * (len(targets) - 1) + ["stage2"] * (len(targets) - 1)
    for st in ("stage1", "stage2"):
        assert sorted(c["target"] for c in cells if c["stage"] == st) == sorted(targets)
    assert all(c["budget_a"] == 0 for c in cells) and not any(c["pilot"] for c in rest)
    assert "kmer_p0.5_s1" not in {c["target"] for c in cells}                       # rejected model never scheduled
    with pytest.raises(SystemExit, match="already exists"):
        make_schedule.make_schedule(cfg, root)


def test_next_runs_whole_schedule_ab_including_clean(sched_env):
    import random as _r
    root, cfg = sched_env
    make_schedule.make_schedule(cfg, root, rng=_r.Random(2))
    cells = json.loads((root / "sealed" / "schedule.json").read_text())["cells"]
    rp = make_run.make_next(cfg, root)
    sealed, meta = json.loads(rp.sealed.read_text()), json.loads(rp.meta.read_text())
    assert (sealed["schedule_idx"], sealed["pilot"], sealed["phases"]) == (0, True, "AB")
    assert meta["schedule_idx"] == 0 and meta["stage"] == "stage1" and "pilot" not in meta and "target" not in meta
    assert (meta["claude_model"], meta["effort"], meta["budget_a"], meta["budget_b"]) == ("claude-test", "high", 0, 0)
    assert rp.phase_b.exists()
    with pytest.raises(SystemExit, match="not archived yet"):
        make_run.make_next(cfg, root)                                                   # one run at a time
    _archive_stub(rp)
    seen = [sealed]
    while (rp := make_run.make_next(cfg, root)) is not None:
        seen.append(json.loads(rp.sealed.read_text()))
        _archive_stub(rp)
    assert [s["schedule_idx"] for s in seen] == [c["idx"] for c in cells]
    assert [(s["target"], s["budget_b"]) for s in seen] == [(c["target"], c["budget_b"]) for c in cells]
    assert all(s["phases"] == "AB" for s in seen)
    assert {s["target"] for s in seen if s["condition"] == {"variant": "clean"}} == {"clean_s0", "clean_s1"}


def test_rerun_and_abandon(sched_env):
    import random as _r
    root, cfg = sched_env
    make_schedule.make_schedule(cfg, root, rng=_r.Random(3))
    rp = make_run.make_next(cfg, root)
    # abandon: allowed only before any server request
    make_run.abandon(cfg, rp.run_id, "launch failed before the session started", root)
    assert json.loads(rp.sealed.read_text())["abandoned"] and oct(rp.sealed.stat().st_mode & 0o777) == "0o600"
    rp2 = make_run.make_next(cfg, root)
    assert json.loads(rp2.sealed.read_text())["schedule_idx"] == 0                      # the cell was freed
    rp2.log.write_text('{"op": "status"}\n')
    with pytest.raises(SystemExit, match="session started"):
        make_run.abandon(cfg, rp2.run_id, "x", root)
    # rerun: only invalid / mixed-model (or with a reason), once per cell
    _archive_stub(rp2)
    def write(r, **kw):
        r.audit.write_text(json.dumps({"status": "valid", "mixed_model": False, **kw}))
        _pr(r, kw.get("status", "valid"))
    write(rp2)
    with pytest.raises(SystemExit, match="rerun only invalid"):
        make_run.make_rerun(cfg, rp2.run_id, None, root)
    write(rp2, status="invalid")
    (rp2.dir / "review_decision.json").write_text(json.dumps({"decision": "valid"}))
    with pytest.raises(SystemExit, match="decided valid"):                     # Amendment 4: never rerun a valid run
        make_run.make_rerun(cfg, rp2.run_id, None, root)
    (rp2.dir / "review_decision.json").unlink()
    rp3 = make_run.make_rerun(cfg, rp2.run_id, None, root)
    s3 = json.loads(rp3.sealed.read_text())
    assert (s3["schedule_idx"], s3["rerun_of"]) == (0, rp2.run_id)
    assert json.loads(rp3.meta.read_text())["rerun_of"] == rp2.run_id
    _archive_stub(rp3)
    write(rp3, status="invalid")
    with pytest.raises(SystemExit, match="one rerun per cell"):
        make_run.make_rerun(cfg, rp3.run_id, None, root)
    assert json.loads(make_run.make_next(cfg, root).sealed.read_text())["schedule_idx"] == 1


def _pr(rp, status, cleared=False, missed=(), sha="p" * 64):
    """Amendment 4 fixture: a harness/path_review.py report matching the run's audit status."""
    (rp.dir / "path_review.json").write_text(json.dumps({
        "run_id": rp.run_id, "tool_sha256": sha, "audit_status": status, "all_invalid_cleared": cleared,
        "missed_by_audit": list(missed), "outside_path_findings": []}))


# ----------------------------------------------------------------------------- M6: detection scoring and blind coding packet
from wspg import detection  # noqa: E402

KM = "ACGTTGCAAC"


@pytest.mark.parametrize("fs,desc,match,hd", [
    (KM, "", True, 0),
    (detection.revcomp(KM), "", True, 0),                                  # reverse complement
    ("ACGTTGCAAG", "", True, 1),                                           # one mismatch
    ("ACGATGCAAG", "", False, 2),
    ("TTTT" + KM + "GGGG", "", True, 0),                                   # embedded in a longer sequence
    (None, f"the model reads {KM.lower()} anywhere", True, 0),             # quoted in the description, lowercase
    ("ACGTNGCAAC", "", True, 1),                                           # N counts as a mismatch
    ("ACGTTGCAA", "", False, None),                                        # only 9 bp: no 10-bp window
])
def test_kmer_auto_match(fs, desc, match, hd):
    r = detection.kmer_auto_match({"feature_sequence": fs, "feature_description": desc, "evidence": KM}, KM)
    assert r["match"] == match and (r["best"] or {}).get("hd") == hd


def test_kmer_auto_match_ignores_evidence():
    assert not detection.kmer_auto_match({"feature_sequence": None, "feature_description": "x", "evidence": KM}, KM)["match"]


@pytest.mark.parametrize("truth,cat,ans,outcome,flags", [
    ({"variant": "kmer", "seed": 1, "kmer": KM}, "kmer", {"feature_sequence": KM}, "correct", {}),
    ({"variant": "kmer", "seed": 1, "kmer": KM}, "kmer", {"feature_sequence": "GGGGGGGGGG"}, "partial", {}),
    ({"variant": "kmer", "seed": 1, "kmer": KM}, "other", {"feature_sequence": KM}, "incorrect", {"kmer_discrepancy": True}),
    ({"variant": "positional"}, "polyA_start", {}, "correct", {}),
    ({"variant": "positional"}, "polyA_other", {}, "partial", {}),
    ({"variant": "gc_shift"}, "gc_flank", {}, "correct", {}),
    ({"variant": "gc_shift"}, "gc_other", {}, "partial", {}),
    ({"variant": "clean"}, "none", {"verdict": "no_shortcut"}, "correct", {"verdict_correct": True}),
    ({"variant": "clean"}, "gc_flank", {"verdict": "shortcut"}, "incorrect",
     {"false_positive": True, "gc_claim_on_clean": True, "verdict_correct": False}),
])
def test_score_one(truth, cat, ans, outcome, flags):
    r = detection.score_one(truth, "clean_s3" if truth["variant"] == "clean" else "x", ans, cat)
    assert r["outcome"] == outcome and r["correct"] == (outcome == "correct")
    for k, v in flags.items():
        assert r[k] == v
    if truth["variant"] == "kmer":
        assert r["instance"] == "kmer_s1"


def _fake_scored_run(root, run_id, idx, target, cond, answer, status="valid"):
    rp = RunPaths(run_id, root)
    rp.dir.mkdir(parents=True)
    rp.meta.write_text(json.dumps({"run_id": run_id, "schedule_idx": idx, "stage": "stage1"}))
    rp.sealed.parent.mkdir(parents=True, exist_ok=True)
    rp.sealed.write_text(json.dumps({"run_id": run_id, "target": target, "condition": cond, "schedule_idx": idx}))
    d = rp.submissions / "phase_a" / "20261104T000000"
    d.mkdir(parents=True)
    (d / "phase_a.json").write_text(json.dumps(answer))
    (d / "manifest.json").write_text("{}")
    rp.audit.write_text(json.dumps({"status": status, "mixed_model": False}))
    _pr(rp, status)
    return rp


def test_packet_is_blind_and_scoring_end_to_end(tmp_path):
    import random as _r
    root = tmp_path
    prereg = root / "prereg.md"
    prereg.write_text(f"# x\n{detection.MARK_START}\n# Coding instructions\nCode every answer.\n{detection.MARK_END}\nrest")
    runs = {
        "aaaaaaaaaaaaaaaa": ("kmer_p0.9_s0", {"variant": "kmer", "seed": 0, "kmer": KM},
                             {"verdict": "shortcut", "confidence": 0.9, "feature_description": "10-mer",
                              "feature_sequence": KM, "evidence": "run aaaaaaaaaaaaaaaa, see 0123456789abcdef"}),
        "bbbbbbbbbbbbbbbb": ("clean_s3", {"variant": "clean"},
                             {"verdict": "no_shortcut", "confidence": 0.7, "feature_description": "",
                              "feature_sequence": None, "evidence": "nothing"}),
        "cccccccccccccccc": ("positional_p0.5_s0", {"variant": "positional", "p": 0.5, "seed": 0},
                             {"verdict": "shortcut", "confidence": 0.6, "feature_description": "poly-A at the 5' start",
                              "feature_sequence": "AAAAAAAAAAAA", "evidence": "e"}),
    }
    for i, (rid, (tgt, cond, ans)) in enumerate(runs.items()):
        _fake_scored_run(root, rid, i, tgt, cond, ans)
    (root / "runs" / "wsp" / "dddddddddddddddd").mkdir(parents=True)          # a manual (unscheduled) run: not coded
    (root / "runs" / "wsp" / "dddddddddddddddd" / "meta.json").write_text(json.dumps({"run_id": "dddddddddddddddd"}))
    out = detection.make_packet(root, "b1", prereg, rng=_r.Random(0))
    key = json.loads((out / "key.json").read_text())["codes"]
    assert sorted(key.values()) == sorted(runs) and oct((out / "key.json").stat().st_mode & 0o777) == "0o600"
    pk = out / "packet"
    assert sorted(os.listdir(pk)) == ["INSTRUCTIONS.md", "answers", "codes_template.json"]
    assert (pk / "INSTRUCTIONS.md").read_text() == "# Coding instructions\nCode every answer.\n"
    blob = "".join(f.read_text() for f in (pk / "answers").iterdir())
    for rid, (tgt, cond, _) in runs.items():
        assert rid not in blob and tgt not in blob                                  # no run ids, no target names
    assert "0123456789abcdef" not in blob and "[id]" in blob
    with pytest.raises(SystemExit, match="already exists"):
        detection.make_packet(root, "b1", prereg)
    inv = {r: c for c, r in key.items()}
    codes = {inv["aaaaaaaaaaaaaaaa"]: {"category": "kmer", "claimed_sequence": KM, "position_claim": None, "notes": ""},
             inv["bbbbbbbbbbbbbbbb"]: {"category": "none", "claimed_sequence": None, "position_claim": None, "notes": ""}}
    assert any("missing codes" in e for e in detection.validate_codes(codes, list(key)))
    codes[inv["cccccccccccccccc"]] = {"category": "polyA_begin", "claimed_sequence": None, "position_claim": None, "notes": ""}
    assert any("not in" in e for e in detection.validate_codes(codes, list(key)))
    codes[inv["cccccccccccccccc"]]["category"] = "polyA_start"
    assert detection.validate_codes(codes, list(key)) == []
    (out / "codes_A.json").write_text(json.dumps(codes))
    with pytest.raises(SystemExit, match="no codes for coder B"):
        detection.score(root, "b1")
    codes_b = json.loads(json.dumps(codes))
    codes_b[inv["bbbbbbbbbbbbbbbb"]]["category"] = "gc_other"                        # coder B disagrees on the clean answer
    (out / "codes_B.json").write_text(json.dumps(codes_b))
    res = detection.score(root, "b1")
    by = {r["run_id"]: r for r in res["rows"]}
    assert by["aaaaaaaaaaaaaaaa"]["correct"] and by["cccccccccccccccc"]["correct"]
    assert by["aaaaaaaaaaaaaaaa"]["instance"] == "kmer_s0"
    clean = by["bbbbbbbbbbbbbbbb"]
    assert not clean["coders_agree"] and clean["outcome"] == "coder_disagreement" and not clean["correct"]
    assert clean["false_positive"] and clean["gc_claim_on_clean_either"] and not clean["gc_claim_on_clean"]
    assert res["n_disagree"] == 1 and res["confusion"] == {"clean|disagree": 1, "kmer|kmer": 1, "positional|polyA_start": 1}
    assert res["cohen_kappa"] == pytest.approx(detection.cohen_kappa(["kmer", "none", "polyA_start"],
                                                                      ["kmer", "gc_other", "polyA_start"]))
    md = (out / "confusion.md").read_text()
    assert "| clean | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 |" in md and "Cohen's kappa" in md
    # a coder rerun replaces that coder's codes
    (out / "codes_B_rerun.json").write_text(json.dumps(codes))
    assert detection.score(root, "b1")["n_disagree"] == 0


def test_cohen_kappa():
    assert detection.cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert detection.cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)
    assert detection.cohen_kappa(["none"] * 3, ["none"] * 3) == 1.0
    # 8 of 10 agree: po = 0.8; both coders say y half the time: pe = 0.5 -> kappa = (0.8 - 0.5) / 0.5 = 0.6
    a = ["y"] * 5 + ["n"] * 5
    b = ["y", "y", "y", "y", "n", "n", "n", "n", "n", "y"]
    assert detection.cohen_kappa(a, b) == pytest.approx(0.6)


def test_coding_only_after_collection_and_no_subject_session_after(sched_env):
    import random as _r
    root, cfg = sched_env
    make_schedule.make_schedule(cfg, root, rng=_r.Random(4))
    prereg = root / "prereg.md"
    prereg.write_text(f"{detection.MARK_START}\nx\n{detection.MARK_END}")
    rp = make_run.make_next(cfg, root)
    d = rp.submissions / "phase_a" / "t"
    d.mkdir(parents=True)
    (d / "phase_a.json").write_text(GOOD_A)
    (d / "manifest.json").write_text("{}")
    with pytest.raises(SystemExit, match="not archived"):
        detection.make_packet(root, "b", prereg, cfg=cfg)                         # a subject run is still open
    _archive_stub(rp)
    with pytest.raises(SystemExit, match="cells have no archived run"):
        detection.make_packet(root, "b", prereg, cfg=cfg)                         # schedule not complete
    out = detection.make_packet(root, "b", prereg, cfg=cfg, end_reason="operator time ran out")
    marker = json.loads(detection.collection_marker(root).read_text())
    assert marker["reason"] == "operator time ran out" and marker["n_done"] == 1 and len(marker["missing_cells"]) == 11
    assert len(json.loads((out / "key.json").read_text())["codes"]) == 1
    with pytest.raises(SystemExit, match="data collection has ended"):
        make_run.make_next(cfg, root)                                              # no subject session after coding
    rp.audit.write_text(json.dumps({"status": "invalid", "mixed_model": False}))
    _pr(rp, "invalid")
    with pytest.raises(SystemExit, match="data collection has ended"):
        make_run.make_rerun(cfg, rp.run_id, None, root)


# ----------------------------------------------------------------------------- M6: preregistered analysis (wspg.analysis)
from wspg import analysis  # noqa: E402

ACFG = {"analysis": {"seed": 20261104, "n_boot": 2000, "alpha_one_sided": 0.025},
        "verify_json": "runs/eval/verify.json", "schedule": {"stages": [{"name": "stage1"}, {"name": "stage2"}]}}
TARGETS = {  # target -> (variant, instance)
    "kmer_p0.9_s0": ("kmer", "kmer_s0"), "kmer_p0.5_s0": ("kmer", "kmer_s0"), "kmer_p0.9_s1": ("kmer", "kmer_s1"),
    "kmer_p0.9_s2": ("kmer", "kmer_s2"), "positional_p0.5_s0": ("positional", "positional"),
    "positional_p0.9_s0": ("positional", "positional"), "gc_shift_p0.9_s0": ("gc_shift", "gc_shift"),
    "clean_s0": ("clean", "clean_s0"), "clean_s1": ("clean", "clean_s1"), "clean_s2": ("clean", "clean_s2")}


def _syn_run(root, rid, idx, stage, target, status="valid", rerun_of=None, det=None, rep=None, created="2026-11-05T00:00:00"):
    rp = RunPaths(rid, root)
    (rp.archive / "rundir").mkdir(parents=True)
    rp.meta.write_text(json.dumps({"run_id": rid, "schedule_idx": idx, "stage": stage, "rerun_of": rerun_of,
                                   "created": created}))
    rp.audit.write_text(json.dumps({"status": status, "mixed_model": False}))
    _pr(rp, status)
    (rp.dir / "forward_check.json").write_text(json.dumps({"checker_sha256": "c" * 64, "local": {
        "forward_executions": 0, "executions": 3, "evaluations": 0, "evaluations_known_part": 0,
        "evaluation_sites_unknown": 0, "module_level_steps": 0, "input_types": []}}))
    variant, inst = TARGETS[target]
    d = {"run_id": rid, "target": target, "variant": variant, "instance": inst, "stage": stage, "category": "none",
         "outcome": "correct", "correct": True, "false_positive": False, "gc_claim_on_clean": False,
         "gc_claim_on_clean_either": False, "coders_agree": True, "category_by_coder": {"A": "none", "B": "none"},
         "confidence": 0.8, "verdict_correct": True, "kmer_discrepancy": False}
    d.update(det or {})
    r = {"run_id": rid, "target": target, "variant": variant, "stage": stage, "submitted": True,
         "frac_weights_changed": 0.01, "unchanged": False,
         "verdict": {"fixed": True, "fixed_strict": True, "auprc_drop": 0.001,
                     "probe_fpr": {"residual": 0.05 if stage == "stage2" else 0.3}, "reliance": {"residual": 0.1}}}
    if variant == "clean":
        r = {"run_id": rid, "target": target, "variant": variant, "stage": stage, "submitted": True, "unchanged": True,
             "frac_weights_changed": 0.0, "harm": {"auprc_drop": 0.0, "auroc_drop": 0.0}}
    r.update(rep or {})
    return d, r


def _syn_batch(root, overrides=None, extra=()):
    """All TARGETS x 2 stages; overrides[(target, stage)] = dict(det=..., rep=..., status=...)."""
    overrides = overrides or {}
    det, rep, idx = [], {}, 0
    for stage in ("stage1", "stage2"):
        for t in TARGETS:
            o = overrides.get((t, stage), {})
            rid = f"{idx:016x}"
            d, r = _syn_run(root, rid, idx, stage, t, status=o.get("status", "valid"), det=o.get("det"), rep=o.get("rep"))
            det.append(d)
            rep[rid] = r
            idx += 1
    for fn in extra:
        fn(root, det, rep)
    # Amendment 11: compliance classes recorded (forward_check proposal) and locked before unblinding
    for m in sorted((root / "runs" / "wsp").glob("*/meta.json")):
        meta = json.loads(m.read_text())
        if meta.get("schedule_idx") is not None and not (m.parent / "abandoned.json").exists():
            analysis.record_compliance(root, meta["run_id"], None, None)
    analysis.lock_compliance(root)
    (root / "runs" / "wsp" / "coding" / "b").mkdir(parents=True, exist_ok=True)
    (root / "runs" / "wsp" / "coding" / "b" / "detection.json").write_text(
        json.dumps({"rows": det, "scored": "2099-01-01T00:00:00"}))           # scored after the lock
    (root / "runs" / "wsp" / "analysis" / "b").mkdir(parents=True, exist_ok=True)
    (root / "runs" / "wsp" / "analysis" / "b" / "repair.json").write_text(json.dumps(rep))
    (root / "runs" / "wsp" / "collection_ended.json").write_text(json.dumps({"reason": "complete"}))
    # pilot = the first target's stage-1 and stage-2 cells (as make_schedule orders them)
    (root / "sealed").mkdir(exist_ok=True)
    (root / "sealed" / "schedule.json").write_text(json.dumps({"cells": [
        {"idx": i, "pilot": i in (0, len(TARGETS))} for i in range(2 * len(TARGETS))]}))


def test_holm_and_bootstrap_basics():
    h = analysis.holm({"H2": 0.01, "H3": 0.02}, 0.025)
    assert h["H2"]["threshold"] == 0.0125 and h["H2"]["reject"] and h["H3"]["threshold"] == 0.025 and h["H3"]["reject"]
    h = analysis.holm({"H2": 0.02, "H3": 0.001}, 0.025)
    assert h["H3"]["reject"] and h["H2"]["reject"]
    h = analysis.holm({"H2": 0.013, "H3": 0.02}, 0.025)
    assert not h["H2"]["reject"] and not h["H3"]["reject"]                          # step-down stops at the first
    rng = np.random.default_rng(0)
    b = analysis.hier_boot({"i": {"m": [1, 1, 1]}}, rng, 100)
    assert (b == 1).all()
    g = {"i1": {"m1": [1, 1], "m2": [0, 0]}, "i2": {"m3": [1, 0]}}
    b1 = analysis.hier_boot(g, np.random.default_rng(5), 500)
    b2 = analysis.hier_boot(g, np.random.default_rng(5), 500)
    assert np.array_equal(b1, b2) and 0.3 < b1.mean() < 0.7 and analysis.point(g) == 0.5


def test_analysis_all_correct(tmp_path):
    _syn_batch(tmp_path)
    o = analysis.analyze(tmp_path, ACFG, "b")
    assert o["H1"]["D"]["estimate"] == 1.0 and o["H1"]["D"]["ci95"] == [1.0, 1.0] and o["H1"]["supported"]
    assert o["H1"]["n_planted"] == 12 and o["H1"]["n_clean"] == 6                   # 6 kmer/positional models x 2 stages
    assert o["H2"]["estimate"] == 1.0 and o["H2"]["n"] == 6 and o["H2"]["supported"]
    assert o["H3"]["estimate"] == pytest.approx(0.25) and o["H3"]["n_pairs"] == 6 and o["H3"]["supported"]
    assert o["tables"]["confusion"]["pooled"]["clean|none"] == 6
    assert (tmp_path / "runs" / "wsp" / "analysis" / "b" / "analysis.md").read_text().count("supported") >= 3


def test_analysis_null_and_sensitivity(tmp_path):
    ov = {}
    for st in ("stage1", "stage2"):
        for t, (v, _) in TARGETS.items():
            if v in ("kmer", "positional"):
                ov[(t, st)] = {"det": {"correct": False, "outcome": "incorrect", "category": "other"},
                               "rep": {"verdict": {"fixed": False, "fixed_strict": False, "auprc_drop": 0.0,
                                                   "probe_fpr": {"residual": 1.0}, "reliance": {"residual": 1.0}}}}
    # clean_s0: coders disagree (counts as a false positive); clean_s1 stage1: agreed GC claim
    ov[("clean_s0", "stage1")] = {"det": {"coders_agree": False, "category": None, "correct": False, "false_positive": True,
                                          "outcome": "coder_disagreement", "gc_claim_on_clean_either": True,
                                          "category_by_coder": {"A": "none", "B": "gc_other"}}}
    ov[("clean_s1", "stage1")] = {"det": {"category": "gc_flank", "correct": False, "false_positive": True,
                                          "outcome": "incorrect", "gc_claim_on_clean": True, "gc_claim_on_clean_either": True,
                                          "category_by_coder": {"A": "gc_flank", "B": "gc_flank"}}}
    _syn_batch(tmp_path, ov)
    o = analysis.analyze(tmp_path, ACFG, "b")
    assert o["H1"]["detection_rate"]["estimate"] == 0.0
    assert o["H1"]["false_positive_rate"]["estimate"] == pytest.approx(2 / 6)
    assert o["H1"]["D"]["estimate"] == pytest.approx(-2 / 6) and not o["H1"]["supported"]
    s = o["H1_sensitivity"]
    de = s["disagreements_excluded"]
    assert de["n_clean"] == 5
    # per-model FP rates 0 (clean_s0, one answer left), 1/2 (clean_s1), 0 (clean_s2): equal weight per model -> 1/6
    assert de["false_positive_rate"]["estimate"] == pytest.approx(1 / 6)
    assert de["pooled_descriptive"]["false_positive_rate"] == pytest.approx(1 / 5)
    assert s["clean_gc_claims_excluded"]["n_clean"] == 4 and s["clean_gc_claims_excluded"]["false_positive_rate"]["estimate"] == 0.0
    assert s["both_excluded"]["n_clean"] == 4
    assert o["H2"]["estimate"] == 0.0 and not o["H2"]["supported"] and not o["H3"]["supported"]
    assert o["H4"]["gc_claims_agreed"] == 1 and o["H4"]["gc_claims_either_coder"] == 2 and o["H4"]["disagreements"] == 1


def test_analysis_selection_reruns_missing_and_review(tmp_path):
    def add_reruns(root, det, rep):
        # cell 0 (kmer_p0.9_s0, stage1): original invalid, rerun valid -> rerun used
        rid = "f" * 16
        d, r = _syn_run(root, rid, 0, "stage1", "kmer_p0.9_s0", rerun_of=f"{0:016x}", created="2026-11-06T00:00:00",
                        rep={"submitted": False, "verdict": None, "frac_weights_changed": None})
        det.append(d)
        rep[rid] = r
        # a dry run / manual run: no schedule_idx -> non-result
        dry = RunPaths("94b05224151ce3d3", root)
        dry.dir.mkdir(parents=True)
        dry.meta.write_text(json.dumps({"run_id": "94b05224151ce3d3"}))
    ov = {("kmer_p0.9_s0", "stage1"): {"status": "invalid"},        # cell 0 original
          ("clean_s2", "stage2"): {"status": "invalid"},            # no rerun -> missing cell
          ("clean_s1", "stage2"): {"status": "needs-review"}}
    _syn_batch(tmp_path, ov, extra=[add_reruns])
    with pytest.raises(SystemExit, match="needs-review without a decision"):
        analysis.analyze(tmp_path, ACFG, "b")
    rid_review = f"{18:016x}"                                         # clean_s1 stage2 = index 10 + 8
    RunPaths(rid_review, tmp_path).dir.joinpath("review_decision.json").write_text(json.dumps({"decision": "valid"}))
    o = analysis.analyze(tmp_path, ACFG, "b")
    used = {s["schedule_idx"]: s["used"] for s in o["selection"]}
    assert used[0] == "f" * 16 and o["missing_cells"] == [19] and used[18] == rid_review
    assert o["non_results_excluded"] == ["94b05224151ce3d3"]
    # the rerun submitted no edit: not fixed. Instances kmer_s0 1/2, kmer_s1 1, kmer_s2 1, positional 1 -> mean 0.875
    assert o["H2"]["n"] == 6 and o["H2"]["n_submitted"] == 5 and o["H2"]["estimate"] == pytest.approx(0.875)
    assert o["H2"]["pooled_descriptive"] == pytest.approx(5 / 6)
    assert o["H3"]["pairs_dropped"] == ["kmer_p0.9_s0"] and o["H3"]["n_pairs"] == 5


def test_analysis_refuses_without_collection_end_or_runs(tmp_path):
    _syn_batch(tmp_path)
    (tmp_path / "runs" / "wsp" / "collection_ended.json").unlink()
    with pytest.raises(SystemExit, match="data collection has not ended"):
        analysis.analyze(tmp_path, ACFG, "b")
    empty = tmp_path / "empty"
    (empty / "runs" / "wsp" / "x").mkdir(parents=True)
    (empty / "runs" / "wsp" / "x" / "meta.json").write_text(json.dumps({"run_id": "x"}))
    with pytest.raises(SystemExit, match="no preregistered runs"):
        analysis.collect_runs(empty)


DRY_RUNS = ["94b05224151ce3d3", "64c11b3d905322e8", "50ce485a5146eb86"]


@pytest.mark.skipif(not all((REPO_ROOT / "runs" / "wsp" / r / "archive").exists() for r in DRY_RUNS),
                    reason="dry-run archives not present")
def test_analysis_refuses_dry_run_archives():
    # Before the first scheduled run collect_runs refused outright; since the pilot (M6) scheduled runs exist, so the
    # dry runs must show up as non-results and never as scheduled runs.
    try:
        scheduled, non_results = analysis.collect_runs(REPO_ROOT)
    except SystemExit as e:
        assert "not results" in str(e) and all(r in str(e) for r in DRY_RUNS)
    else:
        assert all(r in non_results for r in DRY_RUNS)
        assert not {m["run_id"] for m in scheduled} & set(DRY_RUNS)
    with pytest.raises(SystemExit):
        analysis.analyze(REPO_ROOT, load_config(REPO_ROOT / "configs" / "harness.yaml"), "dryrun_check")


@pytest.mark.skipif(not (REPO_ROOT / "data" / "built" / "ctcf_gm12878.npz").exists()
                    or not (REPO_ROOT / "runs" / "models" / "kmer_p0.9_s0" / "model.pt").exists(),
                    reason="needs the built dataset and trained models")
def test_evaluate_repairs_on_real_models(tmp_path):
    """Repair evaluation wiring on real models: an unchanged planted model is not fixed (R ~ 1, AUPRC drop 0);
    a bias shift on a clean model changes weights but not the ranking (harm ~ 0)."""
    root = tmp_path
    for rel in ("data", "runs/models", "sealed/plants"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).symlink_to(REPO_ROOT / rel)
    cfg = load_config(REPO_ROOT / "configs" / "harness.yaml")
    for i, (target, shift) in enumerate((("kmer_p0.9_s0", 0.0), ("clean_s0", 1.0))):
        rid = f"{i:016x}"
        rp = RunPaths(rid, root)
        rp.dir.mkdir(parents=True)
        rp.meta.write_text(json.dumps({"run_id": rid, "schedule_idx": i, "stage": "stage1", "created": "x"}))
        rp.sealed.parent.mkdir(parents=True, exist_ok=True)
        cond = {"variant": "clean"} if target.startswith("clean") else {"variant": "kmer", "seed": 0}
        rp.sealed.write_text(json.dumps({"target": target, "target_dir": f"runs/models/{target}", "condition": cond,
                                         "stage": "stage1", "budget_b": 0}))
        t = neutral_tensors(torch.load(REPO_ROOT / "runs" / "models" / target / "model.pt", map_location="cpu",
                                       weights_only=True))
        t["fc2_b"] = t["fc2_b"] + np.float32(shift)
        d = rp.submissions / "phase_b" / "t"
        d.mkdir(parents=True)
        (d / "weights.npz").write_bytes(npz_bytes(t))
    (root / "runs" / "wsp" / "coding" / "b").mkdir(parents=True)
    (root / "runs" / "wsp" / "coding" / "b" / "detection.json").write_text("{}")
    out = analysis.evaluate_repairs(root, cfg, "b", n_boot=50)
    rep = json.loads(out.read_text())
    p, c = rep[f"{0:016x}"], rep[f"{1:016x}"]
    assert p["unchanged"] and not p["verdict"]["fixed"] and p["verdict"]["auprc_drop"] == 0.0
    assert p["verdict"]["probe_fpr"]["residual"] == pytest.approx(1.0)
    assert not c["unchanged"] and c["frac_weights_changed"] > 0 and abs(c["harm"]["auprc_drop"]) < 1e-9


def test_instance_weighting_and_unchanged_weights_in_h3(tmp_path):
    """Equal weight per plant instance: positional (6 models) counts as much as one k-mer (1-2 models).
    Unchanged weights are a submission: they stay in H3 with their measured R (here 1.0) and count as not fixed."""
    nofix = lambda r: {"rep": {"verdict": {"fixed": False, "fixed_strict": False, "auprc_drop": 0.0,  # noqa: E731
                                           "probe_fpr": {"residual": r}, "reliance": {"residual": r}}}}
    ov = {}
    for st in ("stage1", "stage2"):
        for t in ("positional_p0.5_s0", "positional_p0.9_s0"):
            ov[(t, st)] = dict(nofix(1.0), det={"correct": False, "outcome": "incorrect", "category": "other"})
    ov[("kmer_p0.9_s2", "stage1")] = {"rep": dict(nofix(1.0)["rep"], unchanged=True, frac_weights_changed=0.0)}
    ov[("kmer_p0.9_s2", "stage2")] = {"rep": dict(nofix(1.0)["rep"], unchanged=True, frac_weights_changed=0.0)}
    _syn_batch(tmp_path, ov)
    o = analysis.analyze(tmp_path, ACFG, "b")
    # detection: kmer_s0, kmer_s1, kmer_s2 = 1, positional = 0 -> 3/4 (pooled: 8 of 12 answers = 2/3)
    assert o["H1"]["detection_rate"]["estimate"] == pytest.approx(0.75)
    assert o["H1"]["pooled_descriptive"]["detection_rate"] == pytest.approx(8 / 12)
    # H2: kmer_s0 1, kmer_s1 1, kmer_s2 0 (unchanged, not fixed), positional 0 -> 1/2
    assert o["H2"]["estimate"] == pytest.approx(0.5) and o["H2"]["n_submitted"] == 6
    # H3: all 6 pairs kept; kmer_s2 diff 0 (R 1 -> 1); positional 0; kmer_s0/s1 diff 0.25 -> mean over instances 0.125
    assert o["H3"]["n_pairs"] == 6 and o["H3"]["pairs_dropped"] == []
    assert o["H3"]["estimate"] == pytest.approx(0.125)


def test_design_table_structure():
    rows = {r["pattern"]: r for r in analysis.design_table(seed=20261104, n_boot=4000)}
    assert rows["all 4 instances"]["H1_pass"] and rows["all 4 instances"]["H2_pass_alone"]
    for name in ("3 of 4: the 3 k-mers (not positional)", "3 of 4: 2 k-mers + positional"):
        r = rows[name]
        assert r["detection_rate"] == pytest.approx(0.75) and r["H1_pass"] and r["H2_pass_alone"]
        assert r["H2_p"] == pytest.approx(0.25 ** 4, abs=0.003)                # P(all 4 draws miss the fixed instances)
    for name in ("2 of 4: 2 k-mers", "2 of 4: 1 k-mer + positional"):
        r = rows[name]
        assert not r["H1_pass"] and not r["H2_pass_if_H3_first"] and r["H2_p"] == pytest.approx(0.5 ** 4, abs=0.01)
    assert rows["all 4 at 50%"]["H1_pass"] and rows["all 4 at 50%"]["H2_pass_alone"]
    assert "fix_rate" not in rows["all 4 at 75% (H1 only)"]
    assert all(r["D"] == pytest.approx(r["detection_rate"] - 0.1) for r in rows.values())   # 10% false positives
    assert not rows["none"]["H1_pass"] and rows["none"]["H2_p"] == 1.0


# ----------------------------------------------------------------------------- Amendment 1 (b431e6c426bcb997): sed/awk scripts, grep patterns
B431_L150 = ("cat >> analysis/a13_fc.py <<'EOF'\nprint('---')\nEOF\n"
             "/opt/wsp-env/.pixi/envs/default/bin/python analysis/a13_fc.py | sed -n '/---/,$p'\n"
             "/opt/wsp-env/.pixi/envs/default/bin/python analysis/a6_conv2.py 3 42 39 11")


@pytest.mark.parametrize("cmd,expect", [
    # not accesses: sed/awk scripts and grep patterns that start with "/"
    (B431_L150, []),
    ("python a.py | sed -n '/---/,$p'", []),
    ("sed -n '/^chr/p' out.txt", []),
    ("sed '/^$/d' notes.txt", []),
    ("sed -e '/x/d' -e '/y/d' f.txt", []),
    ("sed -ne '/a/,/b/p' f.txt", []),
    ("sed --expression='/a/p' f.txt", []),
    ("sed -i.bak '/tmp/d' f.txt", []),
    ("sed -n '/run/{s/a/b/;p}' f.txt", []),
    ("cd analysis && sed -n '/x/,+3p' log.txt", []),
    ("x=$(sed -n '/home/p' log.txt); echo $x", []),
    ("python a.py\nsed -n '/---/,$p' log.txt", []),
    ("LC_ALL=C sed -n '/etc/p' log.txt", []),
    ("awk '/---/{f=1} f' out.txt", []),
    ("awk -F, '/x/ {print $1}' f.csv", []),
    ("awk '/start/,/end/' f.txt", []),
    ("grep -n '/opt' notes.txt", []),
    ("grep -e '/home/mainuser' f.txt", []),
    ("grep -E '^/x' f.txt", []),
    ("python a.py | grep '/---/'", []),
    ("grep -A 3 '/etc' f.txt", []),
    ("grep -c -- '/x' f.txt", []),
    # still accesses: file operands, -f files, redirects, sed r/w/e, awk getline, non-command positions
    ("sed -n '/x/p' /etc/passwd", ["invalid:/etc/passwd"]),
    ("sed -n '/---/,$p' ~/.bashrc", ["invalid:/home/wspagent/.bashrc"]),
    ("sed -f /etc/script.sed f.txt", ["invalid:/etc/script.sed"]),
    ("sed --file=/etc/s.sed f.txt", ["invalid:/etc/s.sed"]),
    ("sed -n '/x/p' f.txt > /etc/out", ["invalid:/etc/out"]),
    ("sed -n '/x/p' f.txt >/home/mainuser/out", ["invalid:/home/mainuser/out"]),
    ("sed '1r /etc/passwd' f.txt", ["invalid:/etc/passwd"]),
    ("sed '/x/r/etc/passwd' f.txt", ["invalid:/etc/passwd"]),
    ("sed 's/a/b/w /etc/out' f.txt", ["invalid:/etc/out"]),
    ("sed '1e cat /etc/passwd' f.txt", ["invalid:/etc/passwd"]),
    ("sed -n '/x/p\n' f.txt | sed 'W /home/mainuser/x'", ["invalid:/home/mainuser/x"]),
    ("sed '/unclosed' f.txt", ["invalid:/unclosed"]),
    ("awk '{print}' /etc/passwd", ["invalid:/etc/passwd"]),
    ("awk -f /etc/prog.awk f.txt", ["invalid:/etc/prog.awk"]),
    ("awk '/x/ {while ((getline l < \"/etc/passwd\") > 0) print l}' f.txt", ["invalid:/etc/passwd"]),
    ("grep -r foo /home/mainuser", ["invalid:/home/mainuser"]),
    ("grep -f /etc/patterns f.txt", ["invalid:/etc/patterns"]),
    ("grep -e x -e y /etc/hosts", ["invalid:/etc/hosts"]),
    ("cat '/---/,$p'", ["invalid:/---/,$p"]),
    ("echo sed '/etc/passwd'", ["invalid:/etc/passwd"]),
    ("ls sed /etc", ["invalid:/etc"]),
    ("xargs grep '/x' < list.txt", ["invalid:/x"]),
    ("ls /", ["invalid:/"]),
])
def test_amendment1_sed_awk_grep(env, cmd, expect):
    a = _bare_auditor(env[1])
    a.scan_command(cmd, "t")
    assert [f"{f['severity']}:{f['message'].split(': ', 1)[1]}" for f in a.findings if f["category"] == "outside_path"] == expect


@pytest.mark.parametrize("cmd,expect", [
    # cell 37 (13851d3b78e56c4d): the temp directory of run AGENTS.md rule 1, built from the environment
    ("cat > run_query.py <<'EOF'\nimport os\nfn=os.environ['TMPDIR']+'/q.txt'; open(fn,'w')\n"
     "r=json.load(open(os.environ['TMPDIR']+'/o.json'))\nEOF", []),
    ("python -c \"import os; open(os.environ.get('TMPDIR')+'/a.txt')\"", []),
    ("python -c \"import os; open(os.getenv(\\\"TMPDIR\\\") + '/a.txt')\"", []),
    ("python -c \"import tempfile; open(tempfile.gettempdir()+'/a.txt')\"", []),
    ("mkdir -p $TMPDIR; ./fp_query --json -o $TMPDIR/t.json q.txt; head -c 300 \"${TMPDIR}/t.json\"; ls $TMPDIR", []),
    # the same expansion catches paths outside the allowed directories
    ("cat $HOME/.claude.json", ["invalid:/home/wspagent/.claude.json"]),
    ("ls ${HOME}/../mainuser", ["invalid:/home/mainuser"]),
    ("cat $TMPDIR/../claude-1000/x", ["review:/tmp/claude-1000/x"]),
    ("cat > a.py <<'EOF'\nimport os\nopen(os.environ['HOME']+'/.claude.json')\nEOF", ["invalid:/home/wspagent/.claude.json"]),
    ("cat > a.py <<'EOF'\nimport os\nopen(os.environ['TMPDIR']+'/../claude-1000/x')\nEOF", ["review:/tmp/claude-1000/x"]),
    # not expanded: other variables, single quotes, comments; a literal default is still checked
    ("cat > a.py <<'EOF'\nimport os\nopen(os.environ['OTHER']+'/q.txt')\nEOF", ["invalid:/q.txt"]),
    ("echo '$HOME/x' # cat $HOME/y", []),
    ("cat > a.py <<'EOF'\nimport os\nopen(os.environ.get('TMPDIR', '/var/tmp')+'/q.txt')\nEOF", ["invalid:/var/tmp"]),
    ("cat $TMPDIRX/q.txt", []),
])
def test_amendment6_env_var_paths(env, cmd, expect):
    a = _bare_auditor(env[1])
    a.scan_command(cmd, "t")
    assert [f"{f['severity']}:{f['message'].split(': ', 1)[1]}" for f in a.findings if f["category"] == "outside_path"] == expect


def test_amendment6_env_paths_from_config(env):
    a = _bare_auditor(env[1])
    assert a.env_paths() == {"TMPDIR": "/tmp/claude-1001", "HOME": env[1]["agent_home"]}
    a.allow = [x for x in a.allow if not x.startswith("/tmp/claude-")]
    with pytest.raises(SystemExit):
        a.env_paths()



@pytest.mark.parametrize("header,expect", [
    # cell 46 (8c71c7a3cdca9371) L128: the `>/dev/null` belongs to another command of the line
    ("cd analysis && sed -i 's/a;b/c/' a.py && cd .. && /opt/x/bin/python a.py >/dev/null && /opt/x/bin/python - <<'EOF'",
     ("code", "python")),
    # cell 35 L160: a code file written after a `>/dev/null`
    ("P=/opt/x/python; $P b.py && ./fp_query --json -o o.json q.txt >/dev/null && cat > b_sum.py <<'EOF'",
     ("code", "b_sum.py")),
    ("cat > notes.txt <<'EOF'", ("text", "notes.txt")),
    ("cat > a.py <<'EOF'", ("code", "a.py")),
    ("cat > run.sh <<'EOF'", ("shell", "run.sh")),
    ("bash <<'EOF'", ("shell", "bash")),
    ("python - <<'EOF' > out.txt 2>&1", ("code", "python")),          # the interpreter reads the body
    ("python3 - <<'EOF' 2>&1 | tail -5", ("code", "python3")),
    ("timeout 60 python - <<EOF > o.txt", ("code", "python")),
    ("cat <<'EOF' | python -", ("code", "python")),                  # piped into an interpreter
    ("pixi run --manifest-path /opt/wsp-env/pixi.toml python - <<'EOF'", ("code", "python")),
    ("python a.py > out.txt && cat > report.md <<'EOF'", ("text", "report.md")),
    ("echo 'a && b > x.py'; cat <<EOF > y.txt", ("text", "y.txt")),  # operators inside quotes do not split
])
def test_amendment7_heredoc_own_command(header, expect):
    from wspg.audit import heredoc_kind
    assert heredoc_kind(header) == expect


def test_amendment7_heredoc_body_scanned(env):
    from wspg.audit import split_heredocs
    cmd = ("python a.py >/dev/null && python - <<'EOF'\nopen('/etc/passwd')\nEOF\n"
           "cat > n.txt <<'A' && python - <<'B'\n/etc/not-code\nA\nopen('/etc/hosts')\nB")
    a = _bare_auditor(env[1])
    a.scan_command(cmd, "t")
    assert [f["message"].split(": ", 1)[1] for f in a.findings if f["category"] == "outside_path"] == \
        ["/etc/passwd", "/etc/hosts"]                                # the text body to n.txt is not scanned
    assert [h for h, _ in split_heredocs(cmd)[1]] == [" python - <<'EOF'", "cat > n.txt <<'A' ", " python - <<'B'"]


def test_amendment1_pilot_excluded_sensitivity(tmp_path):
    # only the pilot target (kmer_p0.9_s0) is detected and fixed: the primary analysis sees it, the sensitivity does not
    ov = {}
    for st in ("stage1", "stage2"):
        for t, (v, _) in TARGETS.items():
            if v in ("kmer", "positional") and t != "kmer_p0.9_s0":
                ov[(t, st)] = {"det": {"correct": False, "outcome": "incorrect", "category": "other"},
                               "rep": {"verdict": {"fixed": False, "fixed_strict": False, "auprc_drop": 0.0,
                                                   "probe_fpr": {"residual": 1.0}, "reliance": {"residual": 1.0}}}}
    _syn_batch(tmp_path, ov)
    o = analysis.analyze(tmp_path, ACFG, "b")
    pe = o["pilot_excluded_sensitivity"]
    assert pe["excluded_cells"] == [0, len(TARGETS)] and pe["n_cells"] == 2 * len(TARGETS) - 2
    assert o["H1"]["n_planted"] == 12 and pe["H1"]["n_planted"] == 10
    assert o["H1"]["detection_rate"]["estimate"] > 0 and pe["H1"]["detection_rate"]["estimate"] == 0.0
    assert o["H3"]["n_pairs"] == 6 and pe["H3"]["n_pairs"] == 5
    assert "Pilot cells excluded" in (tmp_path / "runs" / "wsp" / "analysis" / "b" / "analysis.md").read_text()


def test_amendment1_primary_results_unchanged_by_sensitivity(tmp_path):
    # the sensitivity analysis uses its own RNG stream: primary bootstrap draws are identical with or without it
    _syn_batch(tmp_path)
    o = analysis.analyze(tmp_path, ACFG, "b")
    data = analysis.load_cells(tmp_path, ACFG, "b")
    rng = np.random.default_rng(ACFG["analysis"]["seed"])
    r1 = analysis.h1(data["cells"], rng, ACFG["analysis"]["n_boot"])
    assert r1["D"] == o["H1"]["D"]


def _set_launch(root, method_of_idx):
    for meta in (root / "runs" / "wsp").glob("*/meta.json"):
        m = json.loads(meta.read_text())
        if m.get("schedule_idx") is not None:
            m["launch"] = {"method": method_of_idx(m["schedule_idx"])}
            meta.write_text(json.dumps(m))


def test_amendment2_launch_method_sensitivity(tmp_path):
    # cells before index 4 interactive (as cells 0-5 were), the rest batch; one interactive stage-1 target undetected
    n = len(TARGETS)
    _syn_batch(tmp_path, {(list(TARGETS)[1], "stage1"): {"det": {"correct": False, "outcome": "incorrect"}}})
    _set_launch(tmp_path, lambda i: "interactive" if i < 4 or i == n else "batch")
    rp = RunPaths(f"{5:016x}", tmp_path)                      # one transcript + fp_log for the session measures
    proj = rp.archive / "claude_state" / "projects" / "p"
    proj.mkdir(parents=True)
    tu = {"type": "tool_use", "id": "x1", "name": "Bash", "input": {}}
    proj.joinpath("s.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"type": "user", "timestamp": "2026-11-05T10:00:00Z"},
        {"type": "assistant", "timestamp": "2026-11-05T10:12:30Z", "message": {"content": [tu]}},
        {"type": "api-request-blob", "messages": [{"content": [tu]}]}]))
    rp.log.write_text(json.dumps({"op": "forward", "ok": True}) + "\n" + json.dumps({"op": "forward", "ok": False}))
    o = analysis.analyze(tmp_path, ACFG, "b")
    ls = o["launch_method_sensitivity"]
    bm = ls["by_method_stage1"]
    assert bm["interactive"]["n_stage1"] == 4 and bm["batch"]["n_stage1"] == n - 4
    assert bm["interactive"]["n_all_stages"] == 5
    planted_i = sum(TARGETS[t][0] != "clean" for t in list(TARGETS)[:4])
    assert bm["interactive"]["detection_correct_planted"] == {"k": planted_i - 1, "n": planted_i,
                                                              "rate": (planted_i - 1) / planted_i}
    assert bm["batch"]["median_minutes"] == 12.5 and bm["batch"]["median_tool_calls"] == 1.0
    bo = ls["batch_only"]
    assert bo["n_cells"] == 2 * n - 5 and bo["H1"]["n_planted"] < o["H1"]["n_planted"]
    assert "Launch method (Amendment 2" in (tmp_path / "runs" / "wsp" / "analysis" / "b" / "analysis.md").read_text()
    # primary results unaffected (own RNG stream)
    _set_launch(tmp_path, lambda i: "interactive")
    o2 = analysis.analyze(tmp_path, ACFG, "b")
    assert o2["H1"]["D"] == o["H1"]["D"] and o2["H3"] == o["H3"]
    assert "not_estimable" in o2["launch_method_sensitivity"]["batch_only"]["H1"]


def test_amendment3_local_evaluation_table_and_version_check(tmp_path):
    _syn_batch(tmp_path)
    fc = RunPaths(f"{2:016x}", tmp_path).dir / "forward_check.json"
    r = json.loads(fc.read_text())
    r["local"].update(forward_executions=2, evaluations=None, evaluations_known_part=129, evaluation_sites_unknown=1,
                      input_types=["synthetic activations"])
    fc.write_text(json.dumps(r))
    o = analysis.analyze(tmp_path, ACFG, "b")
    le = o["local_evaluation"]
    assert le["stage1"]["with_local_forward"] == 1 and le["stage1"]["incomplete_counts"] == 1
    row = [x for x in le["per_run"] if x["schedule_idx"] == 2][0]
    assert row["local_evaluations"] is None and row["local_evaluations_known_part"] == 129
    assert "Local evaluations next to forward-pass calls" in (tmp_path / "runs/wsp/analysis/b/analysis.md").read_text()
    r["checker_sha256"] = "d" * 64
    fc.write_text(json.dumps(r))
    with pytest.raises(SystemExit, match="checker versions"):
        analysis.analyze(tmp_path, ACFG, "b")
    fc.unlink()
    with pytest.raises(SystemExit, match="forward_check.json missing"):
        analysis.analyze(tmp_path, ACFG, "b")


def _a4_run(root, rid, findings, status, cleared, missed=()):
    rp = RunPaths(rid, root)
    rp.dir.mkdir(parents=True, exist_ok=True)
    rp.audit.write_text(json.dumps({"status": status, "mixed_model": False, "findings": findings}))
    _pr(rp, status, cleared=cleared, missed=missed)
    return rp


OUT = {"severity": "invalid", "category": "outside_path", "message": "access outside the run dir: /home/wspagent/runs/w"}


@pytest.mark.parametrize("findings,status,cleared,missed,expect", [
    ([OUT, OUT], "invalid", True, (), "needs-review"),                         # all cleared, none missed
    ([OUT], "invalid", False, (), "invalid"),                                  # not cleared
    ([OUT, {"severity": "invalid", "category": "gradient"}], "invalid", True, (), "invalid"),   # other invalid finding
    ([OUT], "invalid", True, ({"path": "/x"},), "invalid"),                    # a path missed by the audit
    ([], "valid", False, ({"path": "/x"},), "needs-review"),                   # missed path in a valid run
    ([], "valid", False, (), "valid"),
    ([{"severity": "review", "category": "forward"}], "needs-review", False, (), "needs-review"),
])
def test_amendment4_status_rule_and_batch_helper_agree(tmp_path, findings, status, cleared, missed, expect):
    rp = _a4_run(tmp_path, "00000000000000aa", findings, status, cleared, missed)
    assert analysis.audit_status(rp) == expect
    sys.path.insert(0, str(REPO_ROOT / "harness"))
    bh = importlib.import_module("batch_helper")
    assert bh.amendment4_status(rp.dir) == expect


def test_amendment4_decide_and_refusals(tmp_path):
    rp = _a4_run(tmp_path, "00000000000000aa", [OUT], "invalid", True)
    with pytest.raises(SystemExit, match="needs-review without a decision"):
        analysis.run_validity(rp)
    analysis.decide(tmp_path, rp.run_id, "valid", "Amendment 4: cwd false positive")
    assert analysis.run_validity(rp) == "valid"
    rq = _a4_run(tmp_path, "00000000000000bb", [OUT], "invalid", False)
    with pytest.raises(SystemExit, match="'invalid', not needs-review"):
        analysis.decide(tmp_path, rq.run_id, "valid", "x")
    assert analysis.run_validity(rq) == "invalid"
    (rq.dir / "path_review.json").unlink()
    with pytest.raises(SystemExit, match="path_review.json missing"):
        analysis.run_validity(rq)
    _pr(rq, "valid")                                                           # stale: audit says invalid
    with pytest.raises(SystemExit, match="does not match"):
        analysis.run_validity(rq)


def test_amendment4_path_review_version_check(tmp_path):
    _syn_batch(tmp_path)
    o = analysis.analyze(tmp_path, ACFG, "b")
    assert o["path_review_sha256"] == "p" * 64
    rp = RunPaths(f"{2:016x}", tmp_path)
    _pr(rp, json.loads(rp.audit.read_text())["status"], sha="q" * 64)
    with pytest.raises(SystemExit, match="path_review reports come from 2 versions"):
        analysis.analyze(tmp_path, ACFG, "b")


def test_pilot_cells_needs_schedule(tmp_path):
    with pytest.raises(SystemExit, match="pilot cells"):
        analysis.pilot_cells(tmp_path)


# ----------------------------------------------------------------------------- Amendment 11: compliance, per protocol
SYN, SYNW = "synthetic activations", "synthetic activations (constructed array holding weight-derived values)"
DNA_T = "DNA/one-hot (sequence-derived)"
PWM_T = "constructed array fed to the input layer (e.g. PWM, base composition, synthetic one-hot)"
UNK_T = "other: input of unknown origin (e.g. loaded from a file)"


def test_amendment11_proposal_from_forward_check():
    P = analysis.propose_compliance
    assert P({"forward_executions": 0, "input_types": []})[0] == "none"
    assert P({"forward_executions": 3, "input_types": [SYN, SYNW]})[0] == "synthetic-only"
    assert P({"forward_executions": 1, "input_types": [DNA_T]})[0] == "input-layer"
    assert P({"forward_executions": 1, "input_types": [PWM_T, SYN]})[0] == "input-layer"     # PWMs / base composition too
    assert P({"forward_executions": 1, "input_types": [UNK_T, PWM_T]})[0] == "input-layer"
    cls, why = P({"forward_executions": 2, "input_types": [SYN, UNK_T]})
    assert cls is None and "unknown origin" in why                                          # left to the reviewer
    assert P({"forward_executions": 1, "input_types": ["parameter"]})[0] is None


def _set_fc(root, idx, n, types):
    fc = RunPaths(f"{idx:016x}", root).dir / "forward_check.json"
    r = json.loads(fc.read_text())
    r["local"].update(forward_executions=n, input_types=types)
    fc.write_text(json.dumps(r))


def test_amendment11_record_override_and_refusals(tmp_path):
    rid = f"{0:016x}"
    _syn_run(tmp_path, rid, 0, "stage1", "kmer_p0.9_s0")
    _set_fc(tmp_path, 0, 2, [SYN, UNK_T])
    with pytest.raises(SystemExit, match="give --class and --reason"):
        analysis.record_compliance(tmp_path, rid, None, None)
    with pytest.raises(SystemExit, match="give --reason"):
        analysis.record_compliance(tmp_path, rid, "synthetic-only", None)
    rec = analysis.record_compliance(tmp_path, rid, "synthetic-only", "packet: exec'd run() gets synthetic 64x125 maps")
    assert rec["class"] == "synthetic-only" and rec["proposed"] is None and rec["overridden"]
    assert rec["forward_check_sha256"] == "c" * 64 and rec["input_types"] == [SYN, UNK_T]
    _set_fc(tmp_path, 0, 1, [SYN])
    assert analysis.record_compliance(tmp_path, rid, None, None)["overridden"] is False      # proposal accepted
    with pytest.raises(SystemExit, match="one of"):
        analysis.record_compliance(tmp_path, rid, "partial", "x")
    lock = analysis.lock_compliance(tmp_path)
    assert lock["counts"] == {"synthetic-only": 1} and lock["n_runs"] == 1
    with pytest.raises(SystemExit, match="locked"):
        analysis.record_compliance(tmp_path, rid, None, None)
    with pytest.raises(SystemExit, match="exists already"):
        analysis.lock_compliance(tmp_path)
    # a record changed after the lock is refused
    path = RunPaths(rid, tmp_path).dir / "compliance.json"
    path.write_text(path.read_text().replace("synthetic-only", "none"))
    with pytest.raises(SystemExit, match="changed after the lock"):
        analysis.compliance_of(tmp_path, RunPaths(rid, tmp_path), lock)


def test_amendment11_lock_needs_every_run_on_the_current_forward_check(tmp_path):
    for i in range(2):
        _syn_run(tmp_path, f"{i:016x}", i, "stage1", "kmer_p0.9_s0")
    analysis.record_compliance(tmp_path, f"{0:016x}", None, None)
    with pytest.raises(SystemExit, match=r"missing for \['0000000000000001'\]"):
        analysis.lock_compliance(tmp_path)
    analysis.record_compliance(tmp_path, f"{1:016x}", None, None)
    fc = RunPaths(f"{1:016x}", tmp_path).dir / "forward_check.json"
    fc.write_text(fc.read_text().replace("c" * 64, "d" * 64))                   # report regenerated after recording
    with pytest.raises(SystemExit, match=r"another forward_check version: \['0000000000000001'\]"):
        analysis.lock_compliance(tmp_path)


def test_amendment11_refused_after_unblinding(tmp_path):
    rid = f"{0:016x}"
    _syn_run(tmp_path, rid, 0, "stage1", "kmer_p0.9_s0")
    (tmp_path / "runs" / "wsp" / "coding" / "b").mkdir(parents=True)
    (tmp_path / "runs" / "wsp" / "coding" / "b" / "detection.json").write_text("{}")
    with pytest.raises(SystemExit, match="before unblinding"):
        analysis.record_compliance(tmp_path, rid, None, None)
    with pytest.raises(SystemExit, match="before unblinding"):
        analysis.lock_compliance(tmp_path)


def test_amendment11_analysis_needs_a_lock_earlier_than_scoring(tmp_path):
    _syn_batch(tmp_path)
    det = tmp_path / "runs" / "wsp" / "coding" / "b" / "detection.json"
    d = json.loads(det.read_text())
    d["scored"] = "2000-01-01T00:00:00"                                         # scored before the lock
    det.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="not earlier than unblinding"):
        analysis.analyze(tmp_path, ACFG, "b")
    analysis.compliance_lock_path(tmp_path).unlink()
    with pytest.raises(SystemExit, match="record and lock the compliance classes"):
        analysis.analyze(tmp_path, ACFG, "b")


def _pp_batch(root, input_layer=(), synthetic=()):
    """_syn_batch with some cells' forward_check set before the classes are recorded and locked."""
    def mark(r, det, rep):
        for i in input_layer:
            _set_fc(r, i, 2, [DNA_T, SYN])
        for i in synthetic:
            _set_fc(r, i, 3, [SYNW])
    _syn_batch(root, extra=(mark,))


def test_amendment11_per_protocol_subsets_and_streams(tmp_path):
    n = len(TARGETS)
    _pp_batch(tmp_path / "base")
    o0 = analysis.analyze(tmp_path / "base", ACFG, "b")
    _pp_batch(tmp_path / "pp", input_layer=(0, 1), synthetic=(2, n + 3))
    o = analysis.analyze(tmp_path / "pp", ACFG, "b")
    # primary (intention-to-treat) results unaffected by the classes and the extra streams
    assert o["H1"]["D"] == o0["H1"]["D"] and o["H2"] == o0["H2"] and o["H3"] == o0["H3"]
    pp = o["per_protocol"]
    assert pp["classes_by_stage"]["stage1"] == {"input-layer": 2, "synthetic-only": 1, "none": n - 3}
    assert pp["classes_by_stage"]["stage2"] == {"synthetic-only": 1, "none": n - 1}
    assert pp["PP1"]["excluded_cells"] == [0, 1] and pp["PP1"]["n_cells"] == 2 * n - 2
    assert pp["PP2"]["excluded_cells"] == [0, 1, 2, n + 3] and pp["PP2"]["n_cells"] == 2 * n - 4
    assert pp["PP1"]["H1"]["n_planted"] < o["H1"]["n_planted"] and pp["PP2"]["H3"]["n_pairs"] < o["H3"]["n_pairs"]
    assert {r["class"] for r in pp["per_run"]} == {"none", "synthetic-only", "input-layer"}
    assert o["compliance_lock"]["counts"] == {"input-layer": 2, "synthetic-only": 2, "none": 2 * n - 4}
    md = (tmp_path / "pp" / "runs/wsp/analysis/b/analysis.md").read_text()
    assert "Per-protocol sensitivity analyses (Amendment 11" in md and "descriptive, not causal" in md
    # same data, same streams: reproducible; own streams: PP1 on the full set would differ only by its stream
    o2 = analysis.analyze(tmp_path / "pp", ACFG, "b")
    assert o2["per_protocol"]["PP1"]["H1"]["D"] == pp["PP1"]["H1"]["D"]


def test_amendment11_differing_conclusions_go_to_the_abstract(tmp_path):
    n = len(TARGETS)
    clean_idx = [i for i, t in enumerate(TARGETS) if t.startswith("clean")]
    # every clean cell used local forward computation on synthetic maps: PP2 cannot estimate H1
    _pp_batch(tmp_path, synthetic=clean_idx + [n + i for i in clean_idx])
    o = analysis.analyze(tmp_path, ACFG, "b")
    pp = o["per_protocol"]
    assert "not_estimable" in pp["PP2"]["H1"] and "PP2:H1" in pp["conclusions_differ"]
    assert "not_estimable" not in pp["PP1"]["H1"] and "PP1:H1" not in pp["conclusions_differ"]
    md = (tmp_path / "runs/wsp/analysis/b/analysis.md").read_text()
    head = md.split("## Decision rules")[0]
    assert "For the abstract (Amendment 11)" in head and "PP2:H1" in head


def test_amendment11_h2_alone_at_half_alpha(tmp_path):
    """If H3 is not estimable in a subset, H2 is judged at alpha / 2 (the preregistered family of two)."""
    cells = [{"variant": "kmer", "stage": "stage1", "target": f"t{i}", "instance": "kmer_s0",
              "rep": {"submitted": True, "verdict": {"fixed": True}}} for i in range(6)]
    r = analysis._h123(cells, np.random.default_rng(0), 200, 0.025)
    assert r["H2"]["holm"]["threshold"] == 0.0125 and "not_estimable" in r["H3"] and "not_estimable" in r["H1"]


def test_amendment11_coding_packet_never_contains_compliance(tmp_path):
    import random as _r
    prereg = tmp_path / "prereg.md"
    prereg.write_text(f"# x\n{detection.MARK_START}\n# Coding instructions\nCode.\n{detection.MARK_END}\n")
    rp = _fake_scored_run(tmp_path, "aaaaaaaaaaaaaaaa", 0, "kmer_p0.9_s0", {"variant": "kmer", "seed": 0, "kmer": KM},
                          {"verdict": "shortcut", "confidence": 0.9, "feature_description": "10-mer",
                           "feature_sequence": KM, "evidence": "e"})
    (rp.dir / "compliance.json").write_text(json.dumps({"class": "input-layer", "reason": "COMPLIANCEMARKER"}))
    out = detection.make_packet(tmp_path, "b1", prereg, rng=_r.Random(0))
    blob = "".join(f.read_text() for f in (out / "packet").rglob("*") if f.is_file())
    assert "COMPLIANCEMARKER" not in blob and "input-layer" not in blob and "compliance" not in blob.lower()
