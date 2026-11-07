"""harness/forward_check.py: AST detection of forward computation outside fp_query (synthetic runs, no model)."""
from __future__ import annotations

import importlib
import json
import sys

import numpy as np

from wspg.utils import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "harness"))
fc = importlib.import_module("forward_check")

RID = "0123456789abcdef"
LIVE = f"/home/wspagent/runs/{RID}"
SECRET = "SECRETLITERAL"

HEAD = '''"""Pure weight algebra on abstract feature vectors; no sequence is evaluated (relu, max_pool1d in words only)."""
import numpy as np
W = np.load('weights.npz')
w2, b2, w3, b3 = W['conv2_w'], W['conv2_b'], W['conv3_w'], W['conv3_b']
f1, fb1, f2, fb2 = W['fc1_w'], W['fc1_b'], W['fc2_w'], W['fc2_b']
def conv(x, w, b):
    C, L = x.shape; k = w.shape[2]; p = (k - 1) // 2
    xp = np.pad(x, ((0, 0), (p, p)))
    out = np.zeros((w.shape[0], L))
    for j in range(k): out += np.einsum('oi,il->ol', w[:, :, j], xp[:, j:j + L])
    return out + b[:, None]
def head(h1):
    h2 = np.maximum(conv(h1, w2, b2), 0); h2 = h2[:, :124].reshape(64, 31, 4).max(2)
    h3 = np.maximum(conv(h2, w3, b3), 0).max(1)
    z = np.maximum(f1 @ h3 + fb1, 0)
    return (f2 @ z + fb2)[0]
if __name__ == '__main__':
    base = head(np.zeros((64, 125)))
    for a in [1.0, 2.0]:
        r = []
        for f in range(64):
            h = np.zeros((64, 125)); h[f, 62] = a
            r.append(head(h) - base)
        print(max(r), np.array(r).max())
'''
USES_HEAD = '''import numpy as np, sys
sys.path.insert(0, 'analysis'); from a11 import head
base = head(np.zeros((64, 125)))
M = np.zeros((64, 3))
for f in range(64):
    for i, p in enumerate([0, 62, 124]):
        h = np.zeros((64, 125)); h[f, p] = 1.5; M[f, i] = head(h) - base
print(M.max(0))
'''
ALGEBRA = '''# path = fc2 @ relu(fc1) ... max_pool1d(conv2(x)): written out in a comment only
import numpy as np
W = np.load("weights.npz")
P = W['fc2_w'] @ W['fc1_w']
Q = np.einsum('o,oik->ik', P[0], W['conv3_w'])
R = np.maximum(P, 0).max()
v = np.abs(W['conv1_w']).sum((1, 2))
np.save('analysis/v1.npy', v)
print(Q.shape, R)
'''
LOADS_SAVED = '''import numpy as np
W = np.load('weights.npz'); w1 = W['conv1_w']
v1 = np.load('analysis/v1.npy')
s = np.einsum('f,fbp->b', v1, w1)
print(s.max())
'''
DNA_ONEHOT = f'''import numpy as np
W = np.load('weights.npz'); w1 = W['conv1_w']
seq = "{SECRET[:0]}ACGTTGCAACGTAGGCTACGATCGATCG"
oh = np.zeros((4, len(seq)))
for i, c in enumerate(seq):
    oh["ACGT".index(c), i] = 1
act = np.array([[np.sum(w1[f] * oh[:, j:j + 19]) for j in range(len(seq) - 18)] for f in range(64)])
print(act.max(1), "{SECRET}")
'''
PWM = '''import numpy as np
W = np.load('weights.npz'); c1 = W['conv1_w']
lo = np.array([[1., 0., 2.] * 6 + [1.]] * 4)
def best(a, b):
    m = -9
    for s in range(-3, 4):
        x, y = a[:, 3:], b[:, :16]
        m = max(m, (x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))
    return m
print([best(c1[f], lo) for f in range(64)])
'''
TORCH_MODEL = '''import torch, numpy as np
import torch.nn as nn
W = np.load('weights.npz')
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(4, 64, 19, padding=9)
        self.r = nn.ReLU()
    def forward(self, x):
        return self.r(self.c1(x)).max(-1).values
net = Net()
net.c1.weight.data = torch.tensor(W['conv1_w'])
x = torch.zeros(1, 4, 500); x[0, 0, :] = 1
print(net(x))
'''
PHASE_B = "import numpy as np\nW = np.load('weights.npz')\nprint(np.maximum(W['conv1_w'] @ np.ones(19), 0).max())\n"


def _bash(i, cmd):
    return {"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {"command": cmd}}


def _run(tmp_path, files, commands):
    d = tmp_path / "runs" / "wsp" / RID
    rd = d / "archive" / "rundir"
    (rd / "analysis").mkdir(parents=True)
    (rd / "phase_b").mkdir()
    np.savez(rd / "weights.npz", conv1_w=np.zeros((64, 4, 19), np.float32), conv2_w=np.zeros((64, 64, 7), np.float32),
             fc2_w=np.zeros((1, 64), np.float32), fc2_b=np.zeros(1, np.float32))
    for name, text in files.items():
        (rd / name).write_text(text)
    proj = d / "archive" / "claude_state" / "projects" / "p"
    proj.mkdir(parents=True)
    entries = [{"type": "user", "message": {"content": "Read AGENTS.md and complete the task."}}]
    for i, c in enumerate(commands):
        blk = _bash(i, c) if isinstance(c, str) else {"type": "tool_use", "id": f"t{i}", **c}
        entries.append({"type": "assistant", "message": {"content": [blk]}})
        entries.append({"type": "user", "toolUseResult": SECRET,
                        "message": {"content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": SECRET}]}})
    (proj / "s.jsonl").write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return d


def _report(tmp_path, files, commands):
    _run(tmp_path, files, commands)
    return fc.build_report(fc.Checker(RID, tmp_path))


def _execs(r):
    return {e["loc"].rsplit("#", 1)[-1]: e for e in r["executions"]}


def test_forward_on_synthetic_activations_and_reuse_by_import(tmp_path):
    files = {"analysis/a11.py": HEAD, "analysis/a12.py": USES_HEAD, "analysis/a2.py": ALGEBRA}
    cmds = [f"cd {LIVE} && python analysis/a11.py", "python analysis/a12.py", "python analysis/a2.py"]
    r = _report(tmp_path, files, cmds)
    ex = _execs(r)
    a11, a12, a2 = ex["L2"], ex["L4"], ex["L6"]
    assert a11["forward"] and a11["types"] == ["synthetic activations"]
    assert a11["evaluations"] == {"head": 129}                       # 1 + 2 x 64
    assert a12["forward"] and a12["imports"] == ["analysis/a11.py"]
    assert a12["evaluations"] == {"head": 193}                       # 1 + 64 x 3; max over logits is not pooling
    assert a12["module_level_steps"] == 0
    assert not a2["forward"]                                         # weight algebra; relu/pool only in comments
    d = r["scripts"]["analysis/a11.py"]
    assert list(d["forward_functions"]) == ["head"]
    assert d["forward_functions"]["head"]["layers"][:2] == ["conv2_w", "conv3_w"]
    assert r["per_script"]["analysis/a11.py"]["imported_by_runs"] == 1
    assert fc.summary_line(r).startswith(f"forward_check {RID}: FORWARD; 2 of 3 executions")


def test_dna_onehot_pwm_and_saved_weights(tmp_path):
    files = {"analysis/a2.py": ALGEBRA, "analysis/l.py": LOADS_SAVED, "d.py": DNA_ONEHOT, "pwm.py": PWM}
    cmds = ["python analysis/a2.py", "python analysis/l.py", "python d.py", "python pwm.py"]
    ex = _execs(_report(tmp_path, files, cmds))
    assert not ex["L4"]["forward"]                                   # v1.npy was saved from weights: algebra
    assert ex["L6"]["forward"] and ex["L6"]["types"] == [fc.DNA_TYPE]
    assert ex["L8"]["forward"] and ex["L8"]["types"] == [fc.INPUT_SPACE_TYPE]   # PWM, not a sequence


def test_torch_module_and_transcript_only_and_edits(tmp_path):
    v1 = "import numpy as np\nW = np.load('weights.npz')\nprint(W['conv1_w'].sum())\n"
    edit = {"name": "Edit", "input": {"file_path": f"{LIVE}/e.py", "old_string": "print(W['conv1_w'].sum())",
                                      "new_string": "x = np.zeros((4, 50)); x[0] = 1\n"
                                                    "print(np.maximum(np.einsum('fbk,bk->f', W['conv1_w'], x[:, :19]), 0))"}}
    cmds = [f"cat > t.py <<'EOF'\n{TORCH_MODEL.rstrip()}\nEOF\npython t.py && rm t.py",     # transcript only
            {"name": "Write", "input": {"file_path": f"{LIVE}/e.py", "content": v1}},
            "python e.py", edit, "python e.py"]
    r = _report(tmp_path, {}, cmds)
    ex = _execs(r)
    assert ex["L2"]["forward"] and ex["L2"]["types"] == [fc.INPUT_SPACE_TYPE]
    assert r["scripts"]["t.py"]["source"].startswith("transcript only")
    assert not ex["L6"]["forward"] and ex["L10"]["forward"]          # the version run at each point counts
    assert r["scripts"]["e.py"]["versions"] == 2


def test_inline_code_and_exec_loader(tmp_path):
    cmds = ["python - <<'EOF'\nexec(open('analysis/a11.py').read())\nprint(head(np.ones((64, 125))))\nEOF",
            "python -c \"from analysis.a11 import head; import numpy as np; print(head(np.eye(64, 125)))\""]
    ex = _execs(_report(tmp_path, {"analysis/a11.py": HEAD}, cmds))
    assert ex["L2"]["kind"] == "inline" and ex["L2"]["forward"]
    assert ex["L2"]["evaluations"]["head"] == 130                    # exec as __main__: 129 + 1
    assert ex["L4"]["forward"] and ex["L4"]["evaluations"] == {"head": 1}   # import: no __main__ block


def test_submission_dir_not_analysed_and_no_literals_in_output(tmp_path):
    files = {"phase_b/b.py": PHASE_B, "d.py": DNA_ONEHOT}
    r = _report(tmp_path, files, ["python phase_b/b.py", "python d.py"])
    assert "phase_b/b.py" not in r["scripts"] and r["skipped_in_submission_dirs"] == 1
    assert [e["target"] for e in r["executions"]] == ["d.py"]
    md = fc.render_md(r)
    out = md + json.dumps(r, default=fc._jsonable)
    assert SECRET not in out and "ACGTTGCA" not in out


def test_loop_counts_with_negative_and_dependent_bounds(tmp_path):
    pairs = HEAD.split("if __name__")[0] + """if __name__ == '__main__':
    base = head(np.zeros((64, 125)))
    for f in range(64):
        for g in range(f, 64):
            for d in range(-6, 7):
                h = np.zeros((64, 125)); h[f, 62] = 1; h[g, 62 + d] = 1
                print(head(h) - base)
"""
    r = _report(tmp_path, {"analysis/p.py": pairs}, ["python analysis/p.py"])
    e = r["executions"][0]
    assert e["evaluations"] == {"head": 1 + 2080 * 13}               # sum over f of (64 - f), times 13 offsets
    assert r["local"]["evaluations"] == 1 + 2080 * 13 and r["local"]["forward_executions"] == 1
    assert len(r["checker_sha256"]) == 64
