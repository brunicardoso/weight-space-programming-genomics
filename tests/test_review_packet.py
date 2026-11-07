"""harness/review_packet.py: only flagged scripts, their load chain and their commands; nothing from submissions."""
from __future__ import annotations

import importlib
import json
import sys

import pytest

from wspg.utils import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "harness"))
rp_mod = importlib.import_module("review_packet")

RID = "0123456789abcdef"
LIVE = f"/home/wspagent/runs/{RID}"
SECRET = "SECRETVERDICT"

IMPULSE2 = "import numpy as np\nw = np.load('weights.npz')\ndef tail(h):\n    return h\n"
TRACE = "exec(open('impulse2.py').read().split('res=[]')[0])\nprint(tail(1))\n"
TRACE3 = "exec(open('trace.py').read())\n"
HELPER = "def helper(x):\n    return x\n"
USES_HELPER = "from analysis.helper import helper\nimport numpy as np\nprint(helper(np.ones(3)))\n"
FLAGGED = "# relu(w2) composite\nimport numpy as np\nfrom analysis import uses_helper\nB='ACGT'\n"
UNRELATED = "import numpy as np\nprint(np.load('weights.npz').files)\n"
MAKE_B = f"exec(open('impulse2.py').read())\nimport numpy as np\nnp.savez('phase_b/weights.npz')  # {SECRET}\n"
EDIT_IN_SUBDIR = f"exec(open('impulse2.py').read())  # {SECRET}\n"


def _bash(i, cmd):
    return {"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {"command": cmd}}


def _run(tmp_path, findings, commands, extra_entries=(), files=None):
    d = tmp_path / "runs" / "wsp" / RID
    rd = d / "archive" / "rundir"
    (rd / "analysis").mkdir(parents=True)
    (rd / "edits").mkdir()
    files = files or {
        "impulse2.py": IMPULSE2, "trace.py": TRACE, "trace3.py": TRACE3, "unrelated.py": UNRELATED,
        "make_b.py": MAKE_B, "edits/edit.py": EDIT_IN_SUBDIR,
        "analysis/helper.py": HELPER, "analysis/uses_helper.py": USES_HELPER, "analysis/flagged.py": FLAGGED,
        "NOTES.md": SECRET, "phase_a.json": json.dumps({"verdict": SECRET}), "report.md": SECRET,
        "NEXT_PHASE.md": SECRET, "edits/report.md": SECRET,
    }
    for name, text in files.items():
        (rd / name).write_text(text)
    (d / "submissions").mkdir()
    (d / "submissions" / "x.json").write_text(SECRET)
    (d / "audit.json").write_text(json.dumps({"status": "needs-review", "findings": findings}))
    proj = d / "archive" / "claude_state" / "projects" / "p"
    proj.mkdir(parents=True)
    entries = [{"type": "user", "message": {"content": "Read AGENTS.md and complete the task."}}]
    for i, c in enumerate(commands):
        blk = _bash(i, c) if isinstance(c, str) else {"type": "tool_use", "id": f"t{i}", **c}
        entries.append({"type": "assistant", "message": {"content": [blk]}})
        entries.append({"type": "api-request-blob", "messages": [{"content": [blk]}]})        # logged twice
        entries.append({"type": "user", "toolUseResult": SECRET,
                        "message": {"content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": SECRET}]}})
    entries += list(extra_entries)
    (proj / "s.jsonl").write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return d


def _loc(d, idx):
    """Transcript loc of the idx-th command (assistant entry): line 2 + 3*idx."""
    return f"archive/claude_state/projects/p/s.jsonl#L{2 + 3 * idx}"


CMDS = [
    f"cat > impulse2.py <<'EOF'\n{IMPULSE2.rstrip()}\nEOF\npython impulse2.py",                          # 0
    f"cat >> NOTES.md <<'EOF'\n{SECRET} impulse2.py looked fine\nEOF\n"
    f"cat > trace.py <<'EOF'\n{TRACE.rstrip()}\nEOF\npython trace.py 4 52",                                # 1
    f"cat > trace3.py <<'EOF'\n{TRACE3.rstrip()}\nEOF\npython trace3.py",                                   # 2
    "python unrelated.py",                                                                                  # 3
    f"echo '{{\"verdict\": \"{SECRET}\"}}' > phase_a.json && ./wsp_submit phase_a phase_a.json NOTES.md",   # 4
    f"cat > report.md <<'EOF'\n{SECRET}\nEOF",                                                              # 5
    f"python - <<'EOF'\nexec(open('impulse2.py').read())\nprint(tail(3))\nEOF",                             # 6 inline loader
    f"python make_b.py && ./wsp_submit phase_b edits/",                                                     # 7
    "sed -i 's/h$/h + 0/' impulse2.py; python impulse2.py > out.txt",                                       # 8
    f"cat > seqs.txt <<'EOF'\nACGTACGTAC impulse2.py {SECRET}\nEOF\n./fp_query seqs.txt",                   # 9
    f"python -c \"import json; print(json.load(open('phase_a.json'))['verdict'])\"",                        # 10
    f"cat edits/report.md; echo {SECRET} >> scratch.txt",                                                   # 11
]
AWAY = {"type": "system", "subtype": "away_summary", "content": f"{SECRET} the session concluded ..."}
ATTACH = {"type": "attachment", "attachment": {"type": "edited_text_file", "filename": "NOTES.md", "snippet": SECRET}}


def _packet(tmp_path, findings, commands=CMDS, **kw):
    d = _run(tmp_path, findings, commands, extra_entries=[AWAY, ATTACH])
    pk = rp_mod.Packet(RID, tmp_path, **{k: v for k, v in kw.items() if k in ("include_info", "extra_scripts", "release")})
    return pk, pk.render(), d


def test_packet_follows_exec_chain_and_strips_submissions(tmp_path):
    f = [{"severity": "review", "category": "forward", "message": "possible forward reimplementation",
          "where": "archive/rundir/impulse2.py"}]
    pk, text, _ = _packet(tmp_path, f)
    assert SECRET not in text
    b = pk.build()
    # flagged script, both loaders (trace, trace3 via trace), and make_b (loads it) are in; unrelated is not
    assert {"impulse2.py", "trace.py", "trace3.py", "make_b.py"} <= set(b["scripts"])
    assert "unrelated.py" not in b["scripts"]
    assert "edits/edit.py" in b["scripts"] and "edits/edit.py" in pk.withheld          # submission dir
    assert "make_b.py" in pk.withheld                                                   # mentions phase_b
    assert "def tail(h):" in text and "exec(open('trace.py').read())" in text
    # commands: creators, runs, the sed -i edit and the inline loader are present; unrelated run is not
    for i in (0, 1, 2, 6, 8):
        assert _loc(None, i) in text, i
    assert _loc(None, 3) not in text
    assert "[non-code heredoc removed" in text and "[line removed" in text
    assert "heredoc body for impulse2.py: DIFFERS" not in text or "sed -i" in text
    assert "python trace.py 4 52" in text


def test_packet_seeds_from_transcript_heredoc_and_imports(tmp_path):
    cmds = CMDS + [f"cat > analysis/flagged.py <<'EOF'\n{FLAGGED.rstrip()}\nEOF\npython analysis/flagged.py 4 52"]
    f = [{"severity": "review", "category": "forward", "message": "layer primitive (relu) next to sequence-like input",
          "where": f"{_loc(None, len(cmds) - 1)} (heredoc -> analysis/flagged.py)"}]
    pk, text, _ = _packet(tmp_path, f, cmds)
    b = pk.build()
    assert b["seeds"] == ["analysis/flagged.py"]
    # flagged imports uses_helper (from analysis import uses_helper), which imports analysis.helper
    assert {"analysis/flagged.py", "analysis/uses_helper.py", "analysis/helper.py"} == set(b["scripts"])
    assert "heredoc body for analysis/flagged.py: identical to the archived file" in text
    assert SECRET not in text


def test_packet_inline_loader_and_info_seeds(tmp_path):
    f = [{"severity": "info", "category": "forward", "message": "layer primitive on weights only (relu)",
          "where": "archive/rundir/trace3.py"},
         {"severity": "info", "category": "environment", "message": "account email", "where": "transcript"}]
    pk, text, _ = _packet(tmp_path, f)
    assert pk.build()["scripts"] == []                         # info findings are not seeds by default
    pk2 = rp_mod.Packet(RID, tmp_path, include_info=True)
    b = pk2.build()
    assert {"trace3.py", "trace.py", "impulse2.py"} <= set(b["scripts"])
    assert any(u["loc"] == _loc(None, 6) for u, _ in b["commands"])     # python - <<EOF exec(impulse2) is a loader
    assert SECRET not in pk2.render()


def test_packet_deleted_script_and_release(tmp_path):
    cmds = CMDS + [f"cat > gone.py <<'EOF'\nexec(open('impulse2.py').read())\nprint(tail(0))\nEOF\npython gone.py; rm gone.py"]
    f = [{"severity": "review", "category": "forward", "message": "x", "where": "archive/rundir/impulse2.py"}]
    pk, text, _ = _packet(tmp_path, f, cmds)
    assert "gone.py" in pk.build()["scripts"] and "not in archived run dir" in text
    pk2 = rp_mod.Packet(RID, tmp_path, release=["make_b.py"])
    t2 = pk2.render()
    assert "make_b.py" not in pk2.withheld and "Released by the operator: make_b.py" in t2
    assert SECRET in t2                                        # released on purpose: the operator's call


def test_packet_write_and_edit_tools(tmp_path):
    cmds = CMDS + [
        {"name": "Write", "input": {"file_path": f"{LIVE}/impulse2.py", "content": IMPULSE2}},
        {"name": "Write", "input": {"file_path": f"{LIVE}/NOTES.md", "content": SECRET}},
        {"name": "Edit", "input": {"file_path": f"{LIVE}/trace.py", "old_string": "tail(1)", "new_string": "tail(2)"}},
    ]
    f = [{"severity": "review", "category": "forward", "message": "x", "where": "archive/rundir/impulse2.py"}]
    pk, text, _ = _packet(tmp_path, f, cmds)
    assert SECRET not in text
    assert '"new_string": "tail(2)"' in text
    assert f"{_loc(None, len(cmds) - 2)}" not in text           # Write NOTES.md names no packet script


def test_cli_writes_packet(tmp_path):
    f = [{"severity": "review", "category": "forward", "message": "x", "where": "archive/rundir/impulse2.py"}]
    d = _run(tmp_path, f, CMDS, extra_entries=[AWAY])
    assert rp_mod.main(["--run", RID, "--root", str(tmp_path)]) == 0
    out = (d / "review_packet.md").read_text()
    assert "impulse2.py" in out and SECRET not in out


def test_bad_run_id(tmp_path):
    with pytest.raises(ValueError):
        rp_mod.Packet("not-a-run", tmp_path)


def test_imports_after_semicolon_and_inside_functions_are_load_edges():
    code = "import numpy as np,sys\nsys.path.insert(0,'analysis'); from a11_probe import head\n" \
           "def f():\n    import helper_mod as h\n    return h\n"
    mods = rp_mod._ast_imports(code)
    assert {"a11_probe", "a11_probe.head", "helper_mod", "numpy", "sys"} <= mods
    assert rp_mod._ast_imports("def broken(:\n") == set()
