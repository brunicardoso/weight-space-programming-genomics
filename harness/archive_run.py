"""Print the commands that move a finished blind run out of wspagent's reach (never runs sudo itself).

  pixi run python harness/archive_run.py --run <run_id>

Moves (not copies) everything a later wspagent session could read: the run directory, the Claude Code transcript
project dir, per-session state under ~/.claude (file history, plans, shell snapshots, ...), the prompt history,
~/.bash_history, the CLI cache for this run (~/.cache/claude-cli-nodejs/<project>, incl. MCP logs) and
wspagent-owned /tmp entries. Copies settings.json and .claude.json for the audit. Destination: runs/wsp/<id>/archive/.
The only thing this script does itself is create that destination directory (main user, mode 700).
"""
from __future__ import annotations

import argparse
import json
import re
import shlex

from wspg.utils import REPO_ROOT, load_config
from wspg.wsp import RunPaths

# Per-session state under ~/.claude that a later session could read (moved wholesale: one run at a time).
SESSION_DIRS = ("file-history", "plans", "shell-snapshots", "session-env", "debug", "tasks", "todos",
                "paste-cache", "uploads", "image-cache", "sessions", "dev-mods", "backups")


def project_dir_name(path: str) -> str:
    """Claude Code's transcript dir name: every non-alphanumeric character of the cwd becomes '-'."""
    return re.sub(r"[^A-Za-z0-9]", "-", path)


def commands(cfg: dict, rp: RunPaths, sudo: bool = True, stop: bool = True, audit: bool = True) -> str:
    """The archive steps. sudo=True: commands to paste (each prefixed with sudo); sudo=False: lines for a root script
    (finish.sh), which handles stopping the server and the audit itself (stop=False, audit=False)."""
    S = "sudo " if sudo else ""
    u, h, main = cfg["agent_user"], cfg["agent_home"], cfg["main_user"]
    src = f"{cfg['agent_runs']}/{rp.run_id}"
    a = rp.archive
    q = shlex.quote
    proj = project_dir_name(src)
    loop = (f"cd {h}/.claude && for d in {' '.join(SESSION_DIRS)}; do "
            f"if [ -d \"$d\" ]; then mkdir -p {q(str(a))}/claude_state/\"$d\" && "
            f"find \"$d\" -mindepth 1 -maxdepth 1 -exec mv -t {q(str(a))}/claude_state/\"$d\" {{}} +; fi; done")
    out = []
    if stop:
        out.append("""
# --- 0. exit the blind Claude session, then stop the forward-pass server (SIGTERM; it removes its socket)
pkill -f fp_server.py""")
    out.append(f"""
# --- 1. run directory ------------------------------------------------------------------------
{S}mkdir -p {a}
{S}mv {src} {a}/rundir

# --- 2. transcript + per-session Claude Code state (moved, so later sessions cannot read them) ---
{S}mkdir -p {a}/claude_state/projects
{S}sh -c 'if [ -d {h}/.claude/projects/{proj} ]; then mv {h}/.claude/projects/{proj} {a}/claude_state/projects/; else echo "no transcript dir {proj}"; ls -A {h}/.claude/projects; fi'
{S}sh -c {q(loop)}
{S}sh -c 'if [ -f {h}/.claude/history.jsonl ]; then mv {h}/.claude/history.jsonl {a}/claude_state/; fi'
{S}cp {h}/.claude/settings.json {a}/wspagent_settings.json
{S}cp {h}/.claude.json {a}/claude_state/claude.json

# --- 2b. shell history and the CLI cache for this run (MCP server logs etc.) ----------------------
{S}mkdir -p {a}/leftovers
{S}sh -c 'if [ -f {h}/.bash_history ]; then mv {h}/.bash_history {a}/leftovers/; fi'
{S}sh -c 'if [ -d {h}/.cache/claude-cli-nodejs/{proj} ]; then mv {h}/.cache/claude-cli-nodejs/{proj} {a}/leftovers/; fi'

# --- 3. wspagent-owned /tmp entries (Claude scratchpad, $TMPDIR files) --------------------------
{S}mkdir -p {a}/tmp
{S}find /tmp -mindepth 1 -maxdepth 1 -user {u} -exec mv -t {a}/tmp {{}} +

# --- 4. ownership + verification ---------------------------------------------------------------
{S}chown -R {main}:{main} {a}
chmod -R u+rwX,go-rwx {a}
{S}ls -A {cfg['agent_runs']} {h}/.claude/projects    # both must be empty
{S}ls -A {h}/.claude                                 # only settings.json and empty/state-free dirs should remain""")
    if audit:
        out.append(f"""
# --- 5. audit --------------------------------------------------------------------------------
cd {REPO_ROOT} && pixi run python -m wspg.audit --run {rp.run_id}""")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "harness.yaml"))
    ap.add_argument("--seed", type=int, default=0, help="unused; script convention")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rp = RunPaths(args.run)
    if not rp.meta.exists():
        raise SystemExit(f"unknown run {args.run}: {rp.meta} not found")
    if (rp.archive / "rundir").exists():
        raise SystemExit(f"run {args.run} is already archived at {rp.archive}")
    rp.archive.mkdir(parents=True, exist_ok=True)
    rp.archive.chmod(0o700)
    meta = json.loads(rp.meta.read_text())
    print(f"Archive run {rp.run_id} (phases={meta['phases']}) into {rp.archive}")
    if (rp.dir / "finish.sh").exists():
        print(f"\n# All steps below in one root script (stop server, archive, audit):\nsudo sh {rp.dir / 'finish.sh'}\n"
              f"\n# ... or step by step:")
    print(commands(cfg, rp))


if __name__ == "__main__":
    main()
