"""harness/path_review.py: relative paths re-resolved against the shell's working directory (review procedure)."""
from __future__ import annotations

import importlib
import json
import sys

import numpy as np

from wspg.audit import Auditor
from wspg.utils import REPO_ROOT, load_config
from wspg.wsp import RunPaths

sys.path.insert(0, str(REPO_ROOT / "harness"))
pr_mod = importlib.import_module("path_review")
rp_mod = importlib.import_module("review_packet")

RID = "0123456789abcdef"
CFG = load_config(REPO_ROOT / "configs" / "harness.yaml")
LIVE = f"{CFG['agent_runs']}/{RID}"
SECRET = "SECRETVERDICT"
SUB = "import numpy as np\nW = np.load('../weights.npz')\nprint(W.files)\n"


def _run(tmp_path, files, calls):
    """calls: (command, recorded cwd after the call (relative, "" = run dir), is_error)."""
    d = tmp_path / "runs" / "wsp" / RID
    rd = d / "archive" / "rundir"
    (rd / "analysis").mkdir(parents=True)
    np.savez(rd / "weights.npz", conv1_w=np.zeros((2, 4, 3), np.float32))
    for name, text in files.items():
        (rd / name).parent.mkdir(parents=True, exist_ok=True)
        (rd / name).write_text(text)
    proj = d / "archive" / "claude_state" / "projects" / "p"
    proj.mkdir(parents=True)
    entries = [{"type": "user", "cwd": LIVE, "message": {"content": "Read AGENTS.md and complete the task."}}]
    for i, (cmd, after, err) in enumerate(calls):
        cwd = f"{LIVE}/{after}" if after else LIVE
        entries.append({"type": "assistant", "cwd": cwd, "message": {"content": [
            {"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {"command": cmd}}]}})
        entries.append({"type": "user", "cwd": cwd, "toolUseResult": SECRET, "message": {"content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": SECRET, "is_error": err}]}})
    (proj / "s.jsonl").write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    # the audit's own scans produce the findings (same `where` labels and resolved paths as audit.py)
    a = Auditor.__new__(Auditor)
    a.p, a.cfg, a.live_dir = RunPaths(RID, tmp_path), CFG, LIVE
    a.allow, a.review_prefixes = list(CFG["audit"]["allow_prefixes"]), list(CFG["audit"].get("review_prefixes", []))
    a.findings, a.stats = [], {}
    a.meta = {"claude_model": "m", "effort": "high", "sha256": {}}
    a.scan_rundir()
    for i, (cmd, _, _) in enumerate(calls):
        a.scan_tool_use("Bash", {"command": cmd}, f"archive/claude_state/projects/p/s.jsonl#L{2 + 2 * i}")
    sev = {f["severity"] for f in a.findings}
    status = "invalid" if "invalid" in sev else "needs-review" if "review" in sev else "valid"
    (d / "audit.json").write_text(json.dumps({"status": status, "findings": a.findings}))
    return d


def _review(tmp_path, files, calls):
    _run(tmp_path, files, calls)
    return pr_mod.PathReview(RID, tmp_path).report()


def _by(r, needle):
    return [o for o in r["outside_path_findings"] if needle in o["where"]]


def test_cd_into_subdir_clears_relative_paths(tmp_path):
    calls = [(f"cd analysis && cat > a6.py <<'EOF'\n{SUB.rstrip()}\nEOF\n/opt/wsp-env/.pixi/envs/default/bin/python a6.py", "analysis", False),
             ("cd .. && ls", "", False)]
    r = _review(tmp_path, {"analysis/a6.py": SUB}, calls)
    assert r["invalid_findings"] == 3 and r["all_invalid_cleared"]
    arch, doc, cd = _by(r, "rundir/analysis/a6.py:2")[0], _by(r, "(heredoc -> a6.py):2")[0], _by(r, "#L4")[0]
    assert arch["audit_path"] == f"{CFG['agent_runs']}/weights.npz" and arch["status"] == "cleared"
    assert arch["cwds"] == ["analysis/"] and arch["paths"] == ["weights.npz"] and arch["raw"] == ["../weights.npz"]
    assert doc["status"] == "cleared" and cd["status"] == "cleared" and cd["raw"] == [".."]
    assert r["commands_disagreeing"] == 0 and [c["cds"] for c in r["commands_with_cd"]] == [["analysis"], [".."]]
    assert pr_mod.summary_line(r).startswith(f"path_review {RID}: 3 outside_path finding(s): 3 cleared")


def test_script_run_from_run_dir_stays_outside(tmp_path):
    r = _review(tmp_path, {"analysis/a6.py": SUB}, [("python analysis/a6.py", "", False)])
    f = _by(r, "rundir/analysis/a6.py:2")[0]
    assert f["status"] == "confirmed" and f["cwds"] == ["run dir"] and not r["all_invalid_cleared"]


def test_never_executed_script_keeps_audit_resolution(tmp_path):
    r = _review(tmp_path, {"analysis/a6.py": SUB}, [("ls analysis", "", False)])
    assert _by(r, "rundir/analysis/a6.py:2")[0]["status"] == "not executed" and not r["all_invalid_cleared"]


def test_failed_call_keeps_cwd_and_is_uncertain(tmp_path):
    calls = [("cd analysis; python a6.py", "", True),         # failed: Claude Code keeps the old cwd
             ("python analysis/a6.py", "", False)]
    r = _review(tmp_path, {"analysis/a6.py": SUB}, calls)
    c = r["commands_with_cd"][0]
    assert c["failed"] and c["agree"] and not c["certain"] and c["after"] == "run dir"
    assert c["and_chain_paths"] == 0
    assert _by(r, "rundir/analysis/a6.py:2")[0]["status"] == "undetermined"      # `;`: may run after a failed cd


def test_amendment7_failed_call_and_chain(tmp_path):
    """A path joined to every cd before it by && runs only if those cds succeeded (Amendment 7)."""
    calls = [("cd analysis && python a6.py", "", True),       # failed; a6.py ran (if at all) in analysis/
             ("ls", "", False)]
    r = _review(tmp_path, {"analysis/a6.py": SUB}, calls)
    c = r["commands_with_cd"][0]
    assert c["failed"] and not c["certain"] and c["after"] == "run dir"
    f = _by(r, "rundir/analysis/a6.py:2")[0]
    assert f["status"] == "cleared" and f["cwds"] == ["analysis/"]


def test_amendment7_cd_dotdot_in_failed_call(tmp_path):
    """Cell 46 (8c71c7a3cdca9371) L128: `cd analysis && ... && cd .. && python ... && python - <<EOF` failed."""
    cmd = ("cd analysis && sed -i 's/^x/y/' a6.py && cd .. && python analysis/a6.py >/dev/null && python - <<'EOF'\n"
           "import pickle\nprint(open('analysis/p.pkl', 'rb'))\nEOF")
    r = _review(tmp_path, {"analysis/a6.py": SUB}, [(cmd, "", True)])
    d = _by(r, "#L2")[0]
    assert d["audit_path"] == CFG["agent_runs"] and d["status"] == "cleared" and d["paths"] == ["run dir"]
    assert _by(r, "rundir/analysis/a6.py:2")[0]["status"] == "confirmed"      # run from the run dir: really outside
    assert r["commands_with_cd"][0]["and_chain_paths"] >= 2


def test_amendment7_operators_that_break_the_chain(tmp_path):
    for k, sep in enumerate(("; ", "\n", " || true && ", " & ")):
        r = _review(tmp_path / str(k), {}, [(f"cd analysis && pwd{sep}cat ../weights.npz", "", True)])
        assert _by(r, "#L2")[0]["status"] == "undetermined", sep
    r = _review(tmp_path / "ok", {}, [("ls; cd analysis && cat ../weights.npz | head", "", True)])
    assert _by(r, "#L2")[0]["status"] == "cleared"            # the `;` comes before the cd; `|` keeps the chain


def test_amendment10_cd_to_current_dir_in_failed_call(tmp_path):
    """Cell 47 (eeb389ffa72d6489) L106: `cd <run dir>; cat > analysis/x.py <<EOF ... EOF\\ncd analysis && python x.py
    && cd ..` failed. The first cd goes to the directory the shell is already in, so it cannot change the cwd whether
    it succeeds or fails; the `;` after it does not make later paths uncertain (Amendment 10)."""
    cmd = (f"cd {LIVE}; cat > analysis/a6.py <<'EOF'\n{SUB.rstrip()}\nEOF\n"
           "cd analysis && /opt/wsp-env/.pixi/envs/default/bin/python a6.py && cd ..")
    r = _review(tmp_path, {"analysis/a6.py": SUB}, [(cmd, "", True)])
    d = _by(r, "#L2")[0]
    assert d["raw"] == [".."] and d["status"] == "cleared" and d["paths"] == ["run dir"]
    assert _by(r, "rundir/analysis/a6.py:2")[0]["status"] == "cleared"         # ran in analysis/ (after `cd analysis &&`)
    c = r["commands_with_cd"][0]
    assert c["noop_cds"] == 1 and c["failed"] and not c["certain"]               # `cd analysis` still makes it uncertain
    assert r["all_invalid_cleared"]


def test_amendment10_only_a_cd_to_the_current_dir(tmp_path):
    # a no-op cd followed by `;` and a path: resolved (here really outside: from the run dir, .. is outside)
    r = _review(tmp_path / "a", {}, [(f"cd {LIVE}; cat ../weights.npz", "", True)])
    assert _by(r, "#L2")[0]["status"] == "confirmed"
    c = r["commands_with_cd"][0]
    assert c["certain"] and c["noop_cds"] == 1 and c["and_chain_paths"] == 0     # no cd that moves: not a failed cd
    r = _review(tmp_path / "b", {}, [("cd .; cat ../weights.npz", "", True)])
    assert _by(r, "#L2")[0]["status"] == "confirmed"
    # a cd to another directory still exposes what follows a `;` (Amendment 7 unchanged)
    r = _review(tmp_path / "c", {}, [(f"cd {LIVE}/analysis; cat ../weights.npz", "", True)])
    assert _by(r, "#L2")[0]["status"] == "undetermined"
    # a cd back to the start directory after a cd that moved is not a no-op: the first cd may have failed
    r = _review(tmp_path / "d", {}, [(f"cd analysis; cd {LIVE}/analysis; cat ../weights.npz", "", True)])
    assert _by(r, "#L2")[0]["status"] == "undetermined" and r["commands_with_cd"][0]["noop_cds"] == 0
    # successful calls are unaffected
    r = _review(tmp_path / "e", {}, [(f"cd {LIVE}; cd analysis; cat ../weights.npz", "analysis", False)])
    assert _by(r, "#L2")[0]["status"] == "cleared"


def test_amendment10_cd_inside_child_shell_still_counts(tmp_path):
    cmd = "bash <<'EOF'\ncd analysis; cat ../weights.npz\nEOF"
    r = _review(tmp_path, {}, [(cmd, "", True)])
    found = _by(r, "#L2")
    assert found and all(o["status"] == "undetermined" for o in found), found


def test_recorded_cwd_mismatch_is_undetermined(tmp_path):
    calls = [(f"cd analysis && cat > a6.py <<'EOF'\n{SUB.rstrip()}\nEOF\npython a6.py", "", False)]   # recorded: no cd
    r = _review(tmp_path, {"analysis/a6.py": SUB}, calls)
    assert r["commands_disagreeing"] == 1 and _by(r, "rundir/analysis/a6.py:2")[0]["status"] == "undetermined"
    assert not r["all_invalid_cleared"]


def test_subshell_multiline_quotes_and_loaded_scripts(tmp_path):
    loader = "exec(open('a6.py').read())\n"
    calls = [("(cd analysis && python loader.py)\npython -c \"\nimport numpy\nprint(1)\n\" # cd elsewhere", "", False)]
    r = _review(tmp_path, {"analysis/a6.py": SUB, "analysis/loader.py": loader}, calls)
    f = _by(r, "rundir/analysis/a6.py:2")[0]
    assert f["status"] == "cleared" and f["cwds"] == ["analysis/"]          # loaded by loader.py, run in analysis/
    assert r["commands_disagreeing"] == 0 and r["commands_uncertain"] == 0


def test_path_missed_by_the_audit_is_listed(tmp_path):
    peek = f"open('../{RID}/weights.npz')\n"                       # audit: <run>/weights.npz; from <runs>/: outside
    r = _review(tmp_path, {"peek.py": peek}, [(f"cd .. && python {RID}/peek.py", "", False)])
    assert _by(r, "#L2")[0]["status"] == "confirmed"                # the `cd ..` itself really leaves the run dir
    m = [x for x in r["missed_by_audit"] if "peek.py" in x["where"]]
    assert m and m[0]["path"] == f"/home/wspagent/{RID}/weights.npz" and m[0]["class"] == "outside"


def test_packet_section_places_heredocs_and_hides_content(tmp_path):
    calls = [(f"cd analysis && cat > a6.py <<'EOF'\n{SUB.rstrip()}\nEOF\npython a6.py", "analysis", False),
             (f"cd .. && cat > NOTES.md <<'EOF'\n{SECRET}\nEOF", "", False)]
    _run(tmp_path, {"analysis/a6.py": SUB, "NOTES.md": SECRET}, calls)
    text = rp_mod.Packet(RID, tmp_path).render()
    assert "## Working directory of path findings" in text and "**cleared**" in text
    assert "[heredoc body for analysis/a6.py: identical to the archived file]" in text
    assert "### a6.py" not in text and "### analysis/a6.py" in text
    assert SECRET not in text
    r = pr_mod.PathReview(RID, tmp_path).report()
    assert SECRET not in json.dumps(r)
