"""Prepare one blind WSP run (runs as the main user; never runs sudo, only prints the commands).

Preregistered batch (M6):
  pixi run python harness/make_run.py --next                      next cell of sealed/schedule.json
  pixi run python harness/make_run.py --rerun <run_id> [--reason] rerun an invalid run once (same cell, fresh session)
  pixi run python harness/make_run.py --abandon <run_id> --reason "..."   free the cell of a run that never started

Manual runs (toy / dry runs):

  pixi run python harness/make_run.py --phases A  --claude-model claude-opus-5-5 --effort high
  pixi run python harness/make_run.py --phases AB --budget-b 25 --claude-model claude-opus-5-5 --effort high [--pool clean]
  pixi run python harness/make_run.py --target toy_ca --phases AB --budget-a 2 --budget-b 2 --claude-model ... --effort ...

- target: drawn at random (default) among eligible models that have the fewest runs with the same phases/budget-b
  (A: accepted planted + clean_s0..s9; AB: accepted planted only), or named explicitly. The choice goes to
  sealed/runs/<id>.json only and is never printed.
- phase-A budget: assigned per target, the least-used value of configs/harness.yaml:budgets_a (ties at random), for A
  and AB runs alike, so the budget is independent of the condition. --budget-a is accepted for toy targets only.
- writes runs/wsp/<id>/{meta.json, staged/, phase_b.md}; staged/ holds exactly what goes into the blind run directory.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import secrets
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from wspg.seq import decode, revcomp_idx, encode_str
from wspg.utils import REPO_ROOT, git_hash, load_config
from wspg.wsp import (FIXED_PROMPT, RunPaths, neutral_tensors, npz_bytes, render, sha256_bytes, sha256_file,
                      tensor_table, write_json_atomic)

HARNESS = REPO_ROOT / "harness"
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))
TEMPLATE_DIR = HARNESS / "run_template"
CLIS = ("fp_query", "wsp_submit")
CONDITION_KEYS = ("variant", "p", "seed", "kmer")


# ----------------------------------------------------------------------------- targets
def eligible(cfg: dict, root: Path) -> dict[str, list[str]]:
    res = json.loads((root / cfg["verify_json"]).read_text())["results"]
    planted = sorted(k for k, v in res.items() if not k.startswith("_") and v["verdict"]["accepted"])
    return {"planted": planted, "clean": list(cfg["clean_targets"]), "toy": list(cfg["toy_targets"])}


def ledger(root: Path) -> list[dict]:
    d = root / "sealed" / "runs"
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.exists() else []


def choose_target(cfg, root, target, phases, budget_b, rng) -> str:
    el = eligible(cfg, root)
    allowed = el["planted"] + el["clean"] + el["toy"]
    if target != "random":
        if target not in allowed:
            raise SystemExit(f"target {target!r} is not eligible for phases={phases}")
        return target
    pool = el["planted"] + el["clean"]
    done = Counter(r["target"] for r in ledger(root) if r["phases"] == phases and r.get("budget_b") == budget_b)
    m = min(done.get(t, 0) for t in pool)
    return rng.choice(sorted(t for t in pool if done.get(t, 0) == m))


def choose_budget_a(cfg, root, target, budget_a, is_toy, rng, scheduled: bool = False) -> int:
    if is_toy:
        return int(budget_a or 0)
    if scheduled:
        return int(budget_a)
    if budget_a is not None:
        raise SystemExit("--budget-a is only accepted for toy targets; it is assigned independently of the condition")
    used = Counter(r["budget_a"] for r in ledger(root) if r["target"] == target)
    m = min(used.get(b, 0) for b in cfg["budgets_a"])
    return int(rng.choice([b for b in cfg["budgets_a"] if used.get(b, 0) == m]))


def model_dir(cfg, root, target) -> Path:
    for d in cfg["model_dirs"]:
        p = root / d / target
        if (p / "model.pt").exists():
            return p
    raise SystemExit(f"no model.pt for {target}")


def condition_of(target: str, mdir: Path, el: dict, root: Path) -> dict:
    if target in el["clean"]:
        return {"variant": "clean"}
    if target in el["toy"]:
        return json.loads((mdir / "run.json").read_text())["config"]["toy"]
    spec = json.loads((root / "sealed" / "plants" / f"{target}.json").read_text())
    return {k: spec[k] for k in CONDITION_KEYS if k in spec}


# ----------------------------------------------------------------------------- files
def build_pool(cfg: dict, root: Path) -> list[str]:
    d = np.load(root / cfg["dataset"])
    seqs = d["seqs"][d["split"] == cfg["pool_split"]]
    seqs = seqs[(seqs < 4).all(axis=1)]                     # fp_query accepts A/C/G/T only
    order = np.random.default_rng(cfg["pool_seed"]).permutation(len(seqs))
    return [decode(s) for s in seqs[order]]


def forbidden_strings(cfg: dict, target: str, cond: dict) -> list[str]:
    bad = list(cfg["forbidden_in_template"]) + [target]
    if cond.get("kmer"):
        k = cond["kmer"]
        bad += [k, decode(revcomp_idx(encode_str(k)))]
    return bad


def check_text(text: str, bad: list[str], what: str) -> None:
    low = text.lower()
    hits = [b for b in bad if b.lower() in low]
    if hits:
        raise SystemExit(f"refusing to write {what}: contains forbidden strings {hits}")


def claude_version(cfg: dict) -> str:
    try:
        return Path(os.readlink(cfg["claude_link"])).name
    except OSError:
        try:
            return subprocess.check_output(["claude", "--version"], text=True).strip() + " (main user's install)"
        except (OSError, subprocess.CalledProcessError):
            return "unknown"


def agent_runs_empty(cfg: dict) -> bool:
    try:
        return not os.listdir(cfg["agent_runs"])
    except PermissionError:
        raise SystemExit(f"cannot list {cfg['agent_runs']} (is the main user in group {cfg['agent_user']}?)")


def settings_hashes(cfg: dict) -> dict:
    raw = (HARNESS / "wspagent_settings.json").read_bytes()
    return {"literal": sha256_bytes(raw), "substituted": sha256_bytes(raw.replace(b"MAINUSER", cfg["main_user"].encode()))}


# ----------------------------------------------------------------------------- main logic
def prepare_run(cfg: dict, *, root: Path = REPO_ROOT, target: str = "random", phases: str = "A",
                budget_a: int | None = None, budget_b: int | None = None, claude_model: str, effort: str,
                pool: str = "none", seed: int | None = None, check_agent_dir: bool = True,
                cell: dict | None = None, rerun_of: str | None = None) -> RunPaths:
    if check_agent_dir and not agent_runs_empty(cfg):
        raise SystemExit(f"{cfg['agent_runs']} is not empty: archive the previous run first (one run at a time)")
    if phases == "AB":
        if budget_b is None:
            raise SystemExit("--budget-b is required for --phases AB")
    elif budget_b is not None:
        raise SystemExit("--budget-b only applies to --phases AB")
    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    el = eligible(cfg, root)
    is_toy = target in el["toy"]
    if phases == "AB" and budget_b not in cfg["budgets_b"] and not is_toy:
        raise SystemExit(f"--budget-b must be one of {cfg['budgets_b']}")
    tgt = choose_target(cfg, root, target, phases, budget_b, rng)
    is_toy = tgt in el["toy"]
    b_a = choose_budget_a(cfg, root, tgt, budget_a, is_toy, rng, scheduled=cell is not None)
    mdir = model_dir(cfg, root, tgt)
    cond = condition_of(tgt, mdir, el, root)

    while True:
        rp = RunPaths(secrets.token_hex(8), root)
        if not rp.dir.exists() and not rp.sealed.exists():
            break

    # weights
    sd = torch.load(mdir / "model.pt", map_location="cpu", weights_only=True)
    tensors = neutral_tensors(sd)
    wbytes = npz_bytes(tensors)

    # templates
    bad = forbidden_strings(cfg, tgt, cond)
    phases_text = (TEMPLATE_DIR / "phases.md").read_text().rstrip("\n")
    agents_md = render((TEMPLATE_DIR / "AGENTS.md").read_text(),
                       {"RUN_ID": rp.run_id, "BUDGET": str(b_a), "PHASES": phases_text, "TENSOR_TABLE": tensor_table(tensors)})
    check_text(agents_md, bad, "AGENTS.md")
    phase_b_md = None
    if phases == "AB":
        phase_b_md = render((TEMPLATE_DIR / "phase_b.md").read_text(), {"RUN_ID": rp.run_id, "BUDGET_B": str(budget_b)})
        check_text(phase_b_md, bad, "phase_b.md")

    # write (sealed first, so a crash never leaves an unattributed run)
    created = time.strftime("%Y-%m-%dT%H:%M:%S")
    rp.sealed.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(rp.sealed.parent.parent, 0o700)
    os.chmod(rp.sealed.parent, 0o700)
    write_json_atomic(rp.sealed, {"run_id": rp.run_id, "target": tgt, "target_dir": str(mdir.relative_to(root)),
                                  "condition": cond, "phases": phases, "budget_a": b_a, "budget_b": budget_b,
                                  "created": created, **_cell_fields(cell, rerun_of, sealed=True)})
    os.chmod(rp.sealed, 0o600)
    rp.staged.mkdir(parents=True)
    (rp.staged / "AGENTS.md").write_text(agents_md)
    (rp.staged / "weights.npz").write_bytes(wbytes)
    for c in CLIS:
        shutil.copy2(HARNESS / c, rp.staged / c)
        os.chmod(rp.staged / c, 0o755)
    pool_info = {"pool": pool}
    if pool == "clean":
        lines = build_pool(cfg, root)
        (rp.staged / "pool.txt").write_text("\n".join(lines) + "\n")
        pool_info.update(pool_split=cfg["pool_split"], pool_seed=cfg["pool_seed"], pool_n=len(lines),
                         pool_sha256=sha256_file(rp.staged / "pool.txt"), pool_note="clean built dataset, no injection, no labels")
    if phase_b_md is not None:
        rp.phase_b.write_text(phase_b_md)
    meta = {"run_id": rp.run_id, "created": created, "git_hash": git_hash(), "claude_model": claude_model, "effort": effort,
            "claude_version": claude_version(cfg), "phases": phases, "budget_a": b_a, "budget_b": budget_b, **pool_info,
            "fixed_prompt": FIXED_PROMPT, "draw": "explicit" if target != "random" else "random",
            "seed": seed, "sha256": {f: sha256_file(rp.staged / f) for f in sorted(os.listdir(rp.staged))},
            "settings_expected_sha256": settings_hashes(cfg), **_cell_fields(cell, rerun_of, sealed=False)}
    write_json_atomic(rp.meta, meta)
    pf = rp.dir / "preflight.sh"
    pf.write_text(preflight_script(cfg, rp.run_id, sorted(set(meta["settings_expected_sha256"].values()))))
    pf.chmod(0o700)
    for name, text in (("install.sh", install_script(cfg, rp)), ("finish.sh", finish_script(cfg, rp))):
        f = rp.dir / name
        f.write_text(text)
        f.chmod(0o700)
    return rp


def _cell_fields(cell: dict | None, rerun_of: str | None, sealed: bool) -> dict:
    """Schedule bookkeeping. meta.json gets stage/index only; the pilot flag (it names a k-mer target) stays sealed."""
    if cell is None:
        return {"rerun_of": rerun_of} if rerun_of else {}
    out = {"schedule_idx": cell["idx"], "stage": cell["stage"], "rerun_of": rerun_of}
    if sealed:
        out["pilot"] = cell["pilot"]
    return out


# Run by preflight.sh with /opt/wsp-env's python: lists every configured MCP server / plugin source it can find.
MCP_CHECK_PY = r"""
import json, os, sys
home, runs = sys.argv[1], sys.argv[2]
bad = []
def names(d):
    return sorted(d) if isinstance(d, dict) and d else []
def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        bad.append(f"{path}: unreadable ({type(e).__name__})")
        return {}
cj = os.path.join(home, ".claude.json")
if os.path.exists(cj):
    d = load(cj)
    if names(d.get("mcpServers")):
        bad.append(f"{cj}: mcpServers {names(d['mcpServers'])}")
    for proj, v in (d.get("projects") or {}).items():
        if isinstance(v, dict) and names(v.get("mcpServers")):
            bad.append(f"{cj}: project {proj}: mcpServers {names(v['mcpServers'])}")
for d in ("/", "/home", home, runs, os.path.join(home, ".claude")):
    m = os.path.join(d, ".mcp.json")
    if os.path.exists(m):
        bad.append(f"{m} exists")
m = "/etc/claude-code/managed-mcp.json"
if os.path.exists(m) and names(load(m).get("mcpServers")):
    bad.append(f"{m}: mcpServers configured")
ip = os.path.join(home, ".claude", "plugins", "installed_plugins.json")
if os.path.exists(ip):
    d = load(ip)
    pl = d.get("plugins", d) if isinstance(d, dict) else d
    if pl:
        bad.append(f"{ip}: installed plugins {sorted(pl) if isinstance(pl, dict) else pl}")
for b in bad:
    print(b)
sys.exit(1 if bad else 0)
"""


def preflight_script(cfg: dict, run_id: str, expected_sha: list[str], settings: str | None = None,
                     require_root: bool = True, mcp_timeout: int = 60) -> str:
    """POSIX sh run by the user as root before installing a run. Exits 1 if any check FAILs.
    `settings` / `require_root` exist only so tests can exercise the settings checks without root."""
    u, h, main = cfg["agent_user"], cfg["agent_home"], cfg["main_user"]
    s = settings or f"{h}/.claude/settings.json"
    root_guard = ('[ "$(id -u)" = 0 ] || { echo "preflight: run as root (sudo sh $0)"; exit 2; }'
                  if require_root else "# root guard disabled (test)")
    return f"""#!/bin/sh
# Pre-flight isolation checks for run {run_id}. Run as root:  sudo sh $0
# Exit status 0 = all checks passed, 1 = at least one FAIL (do not install the run), 2 = not root.
set -u
{root_guard}
fail=0
pass() {{ echo "PASS  $1"; }}
bad()  {{ echo "FAIL  $1"; fail=1; }}
warn() {{ echo "WARN  $1"; }}
S={shlex.quote(s)}

# 1. wspagent cannot read the main user's home
if runuser -u {u} -- ls /home/{main} >/dev/null 2>&1; then bad "{u} can list /home/{main}"; else pass "{u} cannot list /home/{main}"; fi

# 2. one run at a time
if [ -n "$(ls -A {cfg['agent_runs']} 2>/dev/null)" ]; then bad "{cfg['agent_runs']} is not empty"; else pass "{cfg['agent_runs']} is empty"; fi

# 3. settings.json: present, owned by root, not writable by group/other, readable by {u}, expected content
if [ ! -f "$S" ]; then
  bad "$S missing"
else
  own=$(stat -c %U "$S")
  if [ "$own" = root ]; then pass "settings.json owned by root"; else bad "settings.json owned by $own, must be root"; fi
  if [ -n "$(find "$S" -perm /022)" ]; then bad "settings.json is group/other-writable (mode $(stat -c %a "$S"))"; else pass "settings.json not group/other-writable (mode $(stat -c %a "$S"))"; fi
  if runuser -u {u} -- test -r "$S" 2>/dev/null; then pass "settings.json readable by {u}"; else bad "settings.json not readable by {u} (Claude Code would start without it)"; fi
  if lsattr "$S" 2>/dev/null | cut -d' ' -f1 | grep -q i; then pass "settings.json is immutable (chattr +i)"; else bad "settings.json is not immutable: sudo chattr +i $S"; fi
  h=$(sha256sum "$S" | cut -d' ' -f1)
  case "$h" in
    {'|'.join(expected_sha)}) pass "settings.json sha256 matches the repo";;
    *) bad "settings.json sha256 $h matches no expected hash ({', '.join(x[:12] for x in expected_sha)})";;
  esac
fi

# 3b. no skills, no MCP servers, no plugins (the session must not reach the claude.ai account's connectors or skills)
if [ -n "$(ls -A {h}/.claude/skills 2>/dev/null)" ]; then bad "{h}/.claude/skills is not empty: $(ls -A {h}/.claude/skills | tr '\n' ' ')"; else pass "no skills in {h}/.claude/skills"; fi
if mcp_out=$({cfg['env_python']} -I -c {shlex.quote(MCP_CHECK_PY)} {h} {cfg['agent_runs']} 2>&1); then
  pass "no MCP servers or plugins configured in files"
else
  bad "MCP servers / plugins configured:"; printf '%s\\n' "$mcp_out" | sed 's/^/      /'
fi
# Run as {u} with a login environment, from {h}, stdin from /dev/null. timeout is claude's direct parent and
# output goes to a file (not a $(...) pipe), so a hanging claude or a leftover child cannot block the preflight.
mcp_tmp=$(mktemp)
timeout -k 5 {mcp_timeout + 15} runuser -l {u} -c 'cd {h} && exec timeout -k 5 {mcp_timeout} {cfg['claude_link']} mcp list' </dev/null >"$mcp_tmp" 2>&1; rc=$?
mcp_list=$(cat "$mcp_tmp"); rm -f "$mcp_tmp"
if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
  bad "claude mcp list (as {u}) timed out after {mcp_timeout} s (rc=$rc):"; printf '%s\\n' "$mcp_list" | sed 's/^/      /'
elif [ "$rc" = 0 ] && printf '%s\\n' "$mcp_list" | grep -q '^No MCP servers configured' \\
     && ! printf '%s\\n' "$mcp_list" | grep -v '^No MCP servers configured' | grep -q '[^[:space:]]'; then
  pass "claude mcp list (as {u}): No MCP servers configured"
else
  bad "claude mcp list (as {u}) printed something else or failed (rc=$rc):"; printf '%s\\n' "$mcp_list" | sed 's/^/      /'
fi

# 3c. the analysis environment works for {u}
if timeout 120 runuser -u {u} -- {cfg['env_python']} -I -c "import torch, numpy" >/dev/null 2>&1; then
  pass "{cfg['env_python']} imports torch and numpy as {u}"
else
  bad "{cfg['env_python']} cannot import torch/numpy as {u}"
fi

# 4. no instruction files, memory or transcripts left in ~/.claude
for f in CLAUDE.md AGENTS.md CLAUDE.local.md memory; do
  if [ -e {h}/.claude/$f ]; then bad "{h}/.claude/$f exists"; fi
done
if [ -n "$(ls -A {h}/.claude/projects 2>/dev/null)" ]; then bad "{h}/.claude/projects is not empty (old transcripts)"; else pass "no old transcripts"; fi
for f in /home/CLAUDE.md /home/AGENTS.md {h}/CLAUDE.md {h}/AGENTS.md {cfg['agent_runs']}/CLAUDE.md {cfg['agent_runs']}/AGENTS.md; do
  if [ -e "$f" ]; then bad "$f exists (would be loaded as instructions)"; fi
done

# 5. WSL interop off (config and running kernel) and no Windows drive visible to {u}
if grep -A3 '^\\[interop\\]' /etc/wsl.conf 2>/dev/null | grep -qiE '^[[:space:]]*enabled[[:space:]]*=[[:space:]]*false'; then
  pass "WSL interop disabled in /etc/wsl.conf"
else
  bad "WSL interop not disabled in /etc/wsl.conf ([interop] enabled=false, then wsl --shutdown)"
fi
if ls /proc/sys/fs/binfmt_misc/ 2>/dev/null | grep -q '^WSLInterop'; then
  bad "WSL interop is active in the running system (binfmt WSLInterop registered; wsl --shutdown after editing /etc/wsl.conf)"
else
  pass "WSL interop not active"
fi
if runuser -u {u} -- ls /mnt/c/Users >/dev/null 2>&1; then bad "{u} can list /mnt/c/Users"; else pass "{u} cannot list /mnt/c/Users"; fi

if [ "$fail" = 0 ]; then echo "PREFLIGHT PASSED"; else echo "PREFLIGHT FAILED"; exit 1; fi
"""


def install_script(cfg: dict, rp: RunPaths) -> str:
    """Root script: socket dir, immutable settings, preflight, install. Installs nothing unless preflight passes."""
    u, h, main = cfg["agent_user"], cfg["agent_home"], cfg["main_user"]
    st, dst = rp.staged, f"{cfg['agent_runs']}/{rp.run_id}"
    files_ro = " ".join(str(st / f) for f in ("AGENTS.md", "weights.npz", "pool.txt") if (st / f).exists())
    files_x = " ".join(str(st / f) for f in CLIS)
    return f"""#!/bin/sh
# Preflight + install for run {rp.run_id}. Run as root:  sudo sh $0
# Stops at the first error; preflight.sh exits 1 on any FAIL, so nothing is installed unless every check passes.
set -eu
[ "$(id -u)" = 0 ] || {{ echo "install: run as root (sudo sh $0)"; exit 2; }}
[ -d {Path(cfg['socket']).parent} ] || install -d -o {main} -g {u} -m 750 {Path(cfg['socket']).parent}
chattr +i {h}/.claude/settings.json
sh {rp.dir}/preflight.sh
install -d -o {u} -g {u} -m 750 {dst}
install -o {u} -g {u} -m 444 {files_ro} {dst}/
install -o root -g {u} -m 555 {files_x} {dst}/
echo "INSTALLED run {rp.run_id} at {dst}"
"""


def finish_script(cfg: dict, rp: RunPaths) -> str:
    """Root script: stop fp_server, archive the run out of wspagent's reach, then audit it as the main user."""
    import archive_run                                       # harness/ is on sys.path when make_run runs
    u, main = cfg["agent_user"], cfg["main_user"]
    pixi = shutil.which("pixi") or "pixi"
    body = archive_run.commands(cfg, rp, sudo=False, stop=False, audit=False)
    return f"""#!/bin/sh
# Stop the server, archive run {rp.run_id}, audit it. Run as root after the blind session has exited:  sudo sh $0
set -u
[ "$(id -u)" = 0 ] || {{ echo "finish: run as root (sudo sh $0)"; exit 2; }}
[ -d {rp.archive}/rundir ] && {{ echo "finish: run {rp.run_id} is already archived"; exit 1; }}
if pgrep -u {u} -f '(^|/)claude( |$)' >/dev/null; then echo "finish: a claude process of {u} is still running; exit the session first"; exit 1; fi
pkill -f fp_server.py || true
i=0; while [ -S {cfg['socket']} ] && [ $i -lt 20 ]; do sleep 0.5; i=$((i+1)); done
[ -S {cfg['socket']} ] && {{ echo "finish: fp_server did not stop ({cfg['socket']} still exists)"; exit 1; }}
echo "fp_server stopped"
set -e
{body}
set +e
[ -z "$(ls -A {cfg['agent_runs']})" ] || echo "WARNING: {cfg['agent_runs']} is not empty"
[ -z "$(ls -A {cfg['agent_home']}/.claude/projects 2>/dev/null)" ] || echo "WARNING: {cfg['agent_home']}/.claude/projects is not empty"
runuser -l {main} -c 'cd {REPO_ROOT} && {pixi} run python -m wspg.audit --run {rp.run_id}'
"""


def print_instructions(cfg: dict, rp: RunPaths, meta: dict) -> None:
    rid, st, dst = rp.run_id, rp.staged, f"{cfg['agent_runs']}/{rp.run_id}"
    u, h = cfg["agent_user"], cfg["agent_home"]
    distro = os.environ.get("WSL_DISTRO_NAME", "<distro>")
    print(f"""
Run {rid} prepared: phases={meta['phases']} budget A={meta['budget_a']} budget B={meta['budget_b']} pool={meta['pool']}
model={meta['claude_model']} effort={meta['effort']} claude={meta['claude_version']}
(target and condition are in sealed/ only)

# --- 1. preflight + install (root; installs nothing unless every check passes) --------------
sudo sh {rp.dir}/install.sh
#   settings.json must be root-owned, readable by {u} and immutable. To update it intentionally:
#   sudo chattr -i {h}/.claude/settings.json && \\
#   sudo install -o root -g {u} -m 640 {REPO_ROOT}/harness/wspagent_settings.json {h}/.claude/settings.json && \\
#   sudo chattr +i {h}/.claude/settings.json

# --- 2. start the forward-pass server (main user, separate terminal; keep it running) ---------
cd {REPO_ROOT} && pixi run python harness/fp_server.py --run {rid}

# --- 3. launch the blind session in Windows Terminal (not sudo -iu) ---------------------------
wsl -d {distro} -u {u}
cd ~/runs/{rid} && claude --model {meta['claude_model']} --effort {meta['effort']} "{FIXED_PROMPT}"
#   the prompt is passed on the command line: don't type anything in the session.

# --- 4. after the session has exited: stop server, archive, audit (root) ----------------------
sudo sh {rp.dir}/finish.sh
#   (manual equivalent: pkill -f fp_server.py, then harness/archive_run.py --run {rid})
""")


# ----------------------------------------------------------------------------- preregistered schedule
def load_schedule(root: Path) -> dict:
    path = root / "sealed" / "schedule.json"
    if not path.exists():
        raise SystemExit(f"no schedule at {path}: run harness/make_schedule.py first")
    return json.loads(path.read_text())


def _active(rec: dict) -> bool:
    return not rec.get("abandoned")


def unfinished_runs(root: Path) -> list[str]:
    """Non-abandoned runs that are not archived yet (one run at a time)."""
    return [r["run_id"] for r in ledger(root) if _active(r) and not (RunPaths(r["run_id"], root).archive / "rundir").exists()]


def next_cell(root: Path) -> dict | None:
    taken = {r["schedule_idx"] for r in ledger(root) if _active(r) and r.get("schedule_idx") is not None}
    return next((c for c in load_schedule(root)["cells"] if c["idx"] not in taken), None)


def _guard_unfinished(root: Path) -> None:
    from wspg.detection import collection_ended
    if collection_ended(root):
        raise SystemExit("data collection has ended (coding started): no further subject sessions (preregistration §5.1)")
    pending = unfinished_runs(root)
    if pending:
        raise SystemExit(f"run(s) {pending} not archived yet: finish them (finish.sh) or abandon a run that never started")


def make_next(cfg: dict, root: Path = REPO_ROOT, check_agent_dir: bool = True) -> RunPaths | None:
    _guard_unfinished(root)
    cell = next_cell(root)
    if cell is None:
        return None
    return prepare_run(cfg, root=root, target=cell["target"], phases="AB", budget_a=cell["budget_a"],
                       budget_b=cell["budget_b"], claude_model=cfg["claude_model"], effort=cfg["effort"],
                       pool="none", check_agent_dir=check_agent_dir, cell=cell)


def make_rerun(cfg: dict, run_id: str, reason: str | None, root: Path = REPO_ROOT, check_agent_dir: bool = True) -> RunPaths:
    """Rerun a scheduled run once: allowed if its audit is invalid or mixed-model, or with a logged reason (needs-review
    decided invalid before unblinding)."""
    _guard_unfinished(root)
    old = RunPaths(run_id, root)
    rec = json.loads(old.sealed.read_text())
    if rec.get("schedule_idx") is None:
        raise SystemExit(f"{run_id} is not a scheduled run")
    if not old.audit.exists():
        raise SystemExit(f"{run_id} has no audit yet")
    aud = json.loads(old.audit.read_text())
    from wspg.analysis import audit_status                   # Amendment 4: status after path_review
    status = "invalid" if aud.get("mixed_model") else audit_status(old)
    dec = old.dir / "review_decision.json"
    if dec.exists() and json.loads(dec.read_text()).get("decision") == "valid":
        raise SystemExit(f"{run_id} was decided valid (review_decision.json); a valid run is not rerun")
    if status != "invalid" and not reason:
        raise SystemExit(f"{run_id} audit status is {status!r} (after Amendment 4); rerun only invalid / mixed-model "
                         f"runs, or give --reason for a needs-review run decided invalid")
    same = [r for r in ledger(root) if _active(r) and r.get("schedule_idx") == rec["schedule_idx"]]
    if len(same) > 1:
        raise SystemExit(f"cell {rec['schedule_idx']} was already rerun ({[r['run_id'] for r in same]}); one rerun per cell")
    cell = next(c for c in load_schedule(root)["cells"] if c["idx"] == rec["schedule_idx"])
    rp = prepare_run(cfg, root=root, target=cell["target"], phases="AB", budget_a=cell["budget_a"],
                     budget_b=cell["budget_b"], claude_model=cfg["claude_model"], effort=cfg["effort"], pool="none",
                     check_agent_dir=check_agent_dir, cell=cell, rerun_of=run_id)
    if reason:
        write_json_atomic(rp.dir / "rerun_reason.json", {"rerun_of": run_id, "reason": reason,
                                                          "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return rp


def abandon(cfg: dict, run_id: str, reason: str, root: Path = REPO_ROOT) -> None:
    """Free the cell of a run whose session never started (no server request, not installed, not archived)."""
    rp = RunPaths(run_id, root)
    if rp.log.exists():
        raise SystemExit(f"{run_id} has server requests (fp_log.jsonl): the session started; finish and audit it instead")
    if (rp.archive / "rundir").exists():
        raise SystemExit(f"{run_id} is archived; it cannot be abandoned")
    live = Path(cfg["agent_runs"]) / run_id
    if live.exists():
        raise SystemExit(f"{live} is installed. If no session was started there: sudo rm -rf {live} ; then abandon again. "
                         f"If a session was started, run finish.sh instead.")
    rec = json.loads(rp.sealed.read_text())
    rec.update(abandoned=True, abandon_reason=reason, abandoned_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    write_json_atomic(rp.sealed, rec)
    os.chmod(rp.sealed, 0o600)
    write_json_atomic(rp.dir / "abandoned.json", {"reason": reason, "time": rec["abandoned_at"]})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "harness.yaml"))
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--next", action="store_true", help="next cell of the sealed schedule (preregistered batch)")
    mode.add_argument("--rerun", metavar="RUN_ID", help="rerun an invalid scheduled run once")
    mode.add_argument("--abandon", metavar="RUN_ID", help="free the cell of a run that never started")
    ap.add_argument("--reason", help="required for --abandon; for --rerun of a needs-review run decided invalid")
    ap.add_argument("--target", default="random", help="manual runs only")
    ap.add_argument("--phases", choices=["A", "AB"], help="manual runs only")
    ap.add_argument("--budget-a", type=int, help="manual runs, toy targets only")
    ap.add_argument("--budget-b", type=int, help="manual runs only")
    ap.add_argument("--claude-model", help="manual runs only (scheduled runs use configs/harness.yaml)")
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], help="manual runs only")
    ap.add_argument("--pool", choices=["none", "clean"], default="none", help="manual runs only")
    ap.add_argument("--seed", type=int, default=None, help="manual runs: seed for the target/budget draw")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.abandon:
        if not a.reason:
            ap.error("--abandon needs --reason")
        abandon(cfg, a.abandon, a.reason)
        print(f"run {a.abandon} abandoned; its cell is free again")
        return
    if a.next or a.rerun:
        if a.phases or a.budget_a is not None or a.budget_b is not None or a.claude_model or a.effort or a.target != "random":
            ap.error("--next / --rerun take model, effort, target and budgets from the config and the schedule")
        rp = make_next(cfg) if a.next else make_rerun(cfg, a.rerun, a.reason)
        if rp is None:
            print("schedule complete: every cell has a run")
            return
    else:
        if not (a.phases and a.claude_model and a.effort):
            ap.error("manual runs need --phases, --claude-model and --effort (or use --next)")
        rp = prepare_run(cfg, target=a.target, phases=a.phases, budget_a=a.budget_a, budget_b=a.budget_b,
                         claude_model=a.claude_model, effort=a.effort, pool=a.pool, seed=a.seed)
    print_instructions(cfg, rp, json.loads(rp.meta.read_text()))


if __name__ == "__main__":
    sys.exit(main())
