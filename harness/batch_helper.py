"""Detection helpers for harness/batch.sh (stdlib only; run as root with `python3 -I`). Not a §9 file.

  python3 -I harness/batch_helper.py screen <pane.txt> [dir]  -> trust_no | trust_yes | mnt_c | menu | none
  python3 -I harness/batch_helper.py exitscreen <pane.txt>     -> exit_tasks | menu | none (after /exit)
  python3 -I harness/batch_helper.py exitsnap <pane.txt>       -> the exit screen's frame, task lines replaced by a count
  python3 -I harness/batch_helper.py state <run_dir>          -> "phase=<A|B|done|none> b_accepted=<0|1>"
  python3 -I harness/batch_helper.py progress <path> ...      -> signature (file count, sizes, mtimes); no content read
  python3 -I harness/batch_helper.py runid <make_run.out>     -> the 16-hex run id make_run prepared
  python3 -I harness/batch_helper.py audit <run_dir>          -> "<status> invalid=<n> review=<n> info=<n>"
  python3 -I harness/batch_helper.py batchstatus <run_dir>    -> "continue ..." | "pending ..." | "stop ..." (batch_status)
  python3 -I harness/batch_helper.py meta <run_dir> <key>     -> one field of meta.json
  python3 -I harness/batch_helper.py envref <ref.txt> <KEY>   -> value of KEY in the interactive reference (exit 1 if unset)
  python3 -I harness/batch_helper.py envdiff <ref.txt> <env>  -> differences between the reference and a process env

Only fp_server's state.json and the phase/op/ok fields of fp_log.jsonl are read; never submissions or transcripts' text.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Start-up screens. A selection menu is recognised by Claude Code's "Enter to confirm" hint or a cursor before a
# numbered option; the main UI (prompt echo, input box) has neither, so it is never mistaken for a menu.
CURSOR_LINE = re.compile(r"^\s*❯\s*(\S.*?)\s*$", re.M)
CURSOR_NUMBERED = re.compile(r"^\s*[❯›>]\s*\d+[.)]\s", re.M)
TRUST_YES, TRUST_NO = "Yes, I trust this folder", "No, exit"

# The /mnt/c settings screen (preregistration §4.2: "Claude Code's /mnt/c settings screen, if shown, is answered with
# option 3"). Cause (confirmed by the operator, 2026-11-05): wspagent cannot read /mnt/c/Program Files (EACCES), so
# Claude Code cannot check its Windows managed-settings locations; C:\Program Files\ClaudeCode does not exist, so no
# settings are skipped. Matched narrowly: the "Settings Error" screen listing exactly these two paths, each with
# EACCES, and option 3 "Continue without these settings". Any other settings error (another path, another errno, an
# extra entry) is an unrecognised menu and stops the batch. Text is compared without whitespace (the pane wraps lines).
MNT_C_PATHS = ("/mnt/c/Program Files/ClaudeCode/managed-settings.d", "/mnt/c/Program Files/ClaudeCode/managed-settings.json")
MNT_C_ERRORS = ("Managed settings drop-in directory could not be read: EACCES: permission denied, scandir "
                "'/mnt/c/Program Files/ClaudeCode/managed-settings.d'",
                "Settings file could not be read: EACCES: permission denied, stat "
                "'/mnt/c/Program Files/ClaudeCode/managed-settings.json'")
MNT_C_OPTION_3 = "3. Continue without these settings"


def _nows(s: str) -> str:
    return re.sub(r"\s+", "", s)


def is_mnt_c_screen(text: str) -> bool:
    flat = _nows(text)
    i = flat.find("SettingsError")
    if i < 0:
        return False
    body = flat[i:]
    need = [*MNT_C_PATHS, *MNT_C_ERRORS, MNT_C_OPTION_3, "Enter to confirm"]
    if not all(_nows(x) in body for x in need):
        return False
    if text.count("└") != 2 or body.count("EACCES") != 2 or "❯" not in text:
        return False
    rest = body
    for x in sorted(map(_nows, [*MNT_C_ERRORS, *MNT_C_PATHS]), key=len, reverse=True):
        rest = rest.replace(x, "")
    return "/" not in rest            # no other path anywhere on the screen after "Settings Error"


def classify_screen(text: str, folder: str | None = None) -> str:
    """trust_no / trust_yes: Claude Code's folder-trust question for `folder`, cursor on "No, exit" / on "Yes, I trust
    this folder"; mnt_c: the /mnt/c managed-settings screen exactly as described above; menu: any other selection
    screen (also a trust question for another folder or with the cursor elsewhere, or another settings error);
    none: no selection screen."""
    flat = "".join(line.strip() for line in text.splitlines())
    cursor = [m.group(1) for m in CURSOR_LINE.finditer(text)]
    is_menu = "Enter to confirm" in text or bool(CURSOR_NUMBERED.search(text))
    if TRUST_YES in text and TRUST_NO in text:
        if "Accessing workspace" not in text or (folder and folder not in flat) or len(cursor) != 1:
            return "menu"
        return {TRUST_NO: "trust_no", TRUST_YES: "trust_yes"}.get(cursor[0], "menu")
    if is_mnt_c_screen(text):
        return "mnt_c"
    return "menu" if is_menu else "none"


# The exit screen Claude Code (2.1.318) shows after /exit while background work of the session is running: title, subtitle,
# one line per task, then a select with exactly these options (cell 53, dd9da3385fa32d20, answered "1" by the operator).
# Only the frame is matched and logged; the task lines (the session's own commands) and the screen above are not logged.
EXIT_TITLE, EXIT_SUBTITLE = "Background work is running", "The following will stop when you exit:"
EXIT_OPTIONS = [("1", "Exit and stop tasks"), ("2", "Move to background and exit"), ("3", "Stay")]
OPTION_LINE = re.compile(r"^\s*(❯)?\s*(\d+)\.\s+(.*?)\s*$")


def _exit_frame(text: str):
    """(title index, subtitle index, option line indexes) when the screen is exactly the exit screen, else None."""
    lines = [ln.strip().strip("│").strip() for ln in text.splitlines()]          # a side border, if drawn, is ignored
    titles = [i for i, ln in enumerate(lines) if ln == EXIT_TITLE]
    if len(titles) != 1:
        return None
    t = titles[0]
    subs = [i for i in range(t + 1, len(lines)) if lines[i] == EXIT_SUBTITLE]
    if len(subs) != 1:
        return None
    opts = [(i, OPTION_LINE.match(lines[i])) for i in range(subs[0] + 1, len(lines)) if OPTION_LINE.match(lines[i])]
    if [(m.group(2), m.group(3)) for _, m in opts] != EXIT_OPTIONS:
        return None
    if [i for _, m in opts for i in [m.group(1)] if i] != ["❯"] or opts[-1][0] - opts[0][0] != 2:
        return None                                            # one cursor; the three options on consecutive lines
    return t, subs[0], [i for i, _ in opts]


def classify_exit_screen(text: str) -> str:
    """After /exit: exit_tasks (the background-work exit screen above, exactly) | menu (any other selection screen) |
    none."""
    if _exit_frame(text):
        return "exit_tasks"
    is_menu = "Enter to confirm" in text or bool(CURSOR_NUMBERED.search(text)) or EXIT_TITLE in text
    return "menu" if is_menu else "none"


def exit_snapshot(text: str) -> str:
    """The exit screen's frame for the log: title, subtitle, a count in place of the task lines, the options and the
    line after them (key hint), if any. Nothing above the title is returned."""
    t, s, opts = _exit_frame(text)
    lines = [ln.strip().strip("│").strip() for ln in text.splitlines()]           # as matched: borders removed
    tasks = sum(1 for ln in lines[s + 1:opts[0]] if ln)
    tail = [ln for ln in lines[opts[-1] + 1:opts[-1] + 3] if ln and not set(ln) <= set("─╰╯")][:1]
    return "\n".join([lines[t], lines[s], f"[{tasks} task line(s) not logged]", *(lines[i] for i in opts), *tail])


def run_state(run_dir: Path) -> tuple[str, bool]:
    st = run_dir / "state.json"
    phase = "none"
    if st.exists():
        try:
            phase = str(json.loads(st.read_text()).get("phase", "none"))
        except (json.JSONDecodeError, OSError):
            phase = "none"
    b_ok = False
    log = run_dir / "fp_log.jsonl"
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(e, dict) and e.get("op") == "submit" and e.get("phase") == "B" and e.get("ok") is True:
                b_ok = True
    return phase, b_ok


def progress_signature(paths) -> str:
    items = []
    for p in map(Path, paths):
        files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file()) if p.is_dir() else []
        for f in files:
            try:
                s = f.stat()
            except OSError:
                continue
            items.append(f"{f}:{s.st_size}:{int(s.st_mtime)}")
    return f"{len(items)}|" + "|".join(items)


def run_id(text: str) -> str | None:
    m = re.findall(r"^Run ([0-9a-f]{16}) prepared", text, re.M)
    return m[-1] if len(m) == 1 else None


def amendment4_status(run_dir: Path) -> str:
    """Audit status after Amendment 4 (mirrors wspg.analysis.audit_status; plain JSON, no repo imports)."""
    a = json.loads((run_dir / "audit.json").read_text())
    try:
        pr = json.loads((run_dir / "path_review.json").read_text())
    except (OSError, ValueError):
        return f"{a.get('status')} (no path_review.json)"
    if pr.get("run_id") != run_dir.name or pr.get("audit_status") != a.get("status"):
        return f"{a.get('status')} (path_review.json does not match the audit)"
    if a.get("status") == "invalid":
        inv = [f for f in a.get("findings", []) if f.get("severity") == "invalid"]
        if inv and all(f.get("category") == "outside_path" for f in inv) and pr.get("all_invalid_cleared") \
                and not pr.get("missed_by_audit"):
            return "needs-review"
        return "invalid"
    if a.get("status") == "valid" and pr.get("missed_by_audit"):
        return "needs-review"
    return str(a.get("status"))


PENDING_OK = ("cleared", "undetermined")


def batch_status(run_dir: Path) -> str:
    """What batch.sh does after an audit (operational rule of 2026-11-06; it changes no run's status, which is decided
    after the batch under the frozen rules and Amendment 4). One line, starting with:
      continue  audit valid / needs-review, or invalid made needs-review by Amendment 4 ("continue needs-review (A4)")
      pending   audit invalid, every invalid finding is an outside_path finding that path_review marks cleared or
                undetermined, and path_review found no outside path the audit missed: "invalid, pending review"
      stop      anything else: another invalid category, a finding path_review confirms (or marks not executed /
                review list), an outside path the audit missed, or no usable path_review.json
    Reads only the statuses, categories and counts of audit.json and path_review.json (no paths, no session text)."""
    a = json.loads((run_dir / "audit.json").read_text())
    if a.get("status") != "invalid":
        return f"continue {a.get('status')}"
    s4 = amendment4_status(run_dir)
    if s4 == "needs-review":
        return "continue needs-review (A4)"
    if s4 != "invalid":
        return f"stop {s4}"                                  # no path_review.json, or one that does not match
    pr = json.loads((run_dir / "path_review.json").read_text())
    inv = [f for f in a.get("findings", []) if f.get("severity") == "invalid"]
    other = sorted({str(f.get("category")) for f in inv if f.get("category") != "outside_path"})
    if other:
        return "stop invalid finding(s) of another category: " + ", ".join(other)
    recs = [o for o in pr.get("outside_path_findings", []) if o.get("severity") == "invalid"]
    if [o.get("where") for o in recs] != [f.get("where") for f in inv]:
        return "stop path_review.json does not list the audit's invalid outside_path findings"
    bad = sorted({str(o.get("status")) for o in recs} - set(PENDING_OK))
    if bad:
        return "stop outside_path finding(s) that path_review marks " + ", ".join(bad)
    missed = [m for m in pr.get("missed_by_audit", []) if m.get("class") != "review list"]
    if missed:
        return f"stop {len(missed)} path(s) outside the run dir that the audit missed (path_review)"
    n = {s: sum(o.get("status") == s for o in recs) for s in PENDING_OK}
    tail = f"; {len(pr.get('missed_by_audit', []))} review-list path(s) missed by the audit" if pr.get("missed_by_audit") else ""
    return f"pending {len(recs)} invalid outside_path finding(s): {n['cleared']} cleared, {n['undetermined']} undetermined{tail}"


def audit_line(run_dir: Path) -> str:
    a = json.loads((run_dir / "audit.json").read_text())
    c = a.get("counts", {})
    return f"{a.get('status')} invalid={c.get('invalid', 0)} review={c.get('review', 0)} info={c.get('info', 0)}"


# Interactive reference: `env` of a wspagent login in Windows Terminal plus STTY_SIZE/TTY_OWNER/PARENT lines
# (see the operations log, "batch.sh"). Per-session values are reported but expected to differ.
PROBE_KEYS = {"STTY_SIZE", "TTY_OWNER", "PARENT"}
PER_SESSION = {"PWD", "OLDPWD", "_", "WT_SESSION"}
SAFE_VALUE = re.compile(r"^[A-Za-z0-9_.:/+-]*$")


def parse_env(text: str) -> dict[str, str]:
    """KEY=VALUE lines (newline- or NUL-separated); lines that are not KEY=VALUE (multi-line values) are skipped."""
    out = {}
    for line in re.split(r"[\0\n]", text):
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def env_diff(ref: dict[str, str], cur: dict[str, str]) -> list[str]:
    ref = {k: v for k, v in ref.items() if k not in PROBE_KEYS}
    lines = []
    for k in sorted(set(ref) | set(cur)):
        if k in PER_SESSION:
            continue
        if k not in cur:
            lines.append(f"only interactive: {k}={ref[k]}")
        elif k not in ref:
            lines.append(f"only batch: {k}={cur[k]}")
        elif ref[k] != cur[k]:
            lines.append(f"differs: {k}: interactive={ref[k]} batch={cur[k]}")
    return lines or ["none (apart from per-session values: " + ", ".join(sorted(PER_SESSION)) + ")"]


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd, args = argv[0], argv[1:]
    if cmd == "screen":
        print(classify_screen(Path(args[0]).read_text(errors="replace"), args[1] if len(args) > 1 else None))
    elif cmd == "exitscreen":
        print(classify_exit_screen(Path(args[0]).read_text(errors="replace")))
    elif cmd == "exitsnap":
        print(exit_snapshot(Path(args[0]).read_text(errors="replace")))
    elif cmd == "state":
        phase, b_ok = run_state(Path(args[0]))
        print(f"phase={phase} b_accepted={int(b_ok)}")
    elif cmd == "progress":
        print(progress_signature(args))
    elif cmd == "runid":
        rid = run_id(Path(args[0]).read_text(errors="replace"))
        if not rid:
            return 1
        print(rid)
    elif cmd == "audit":
        print(audit_line(Path(args[0])))
    elif cmd == "a4status":
        print(amendment4_status(Path(args[0])))
    elif cmd == "batchstatus":
        print(batch_status(Path(args[0])))
    elif cmd == "envref":
        env = parse_env(Path(args[0]).read_text(errors="replace"))
        if args[1] not in env:
            return 1
        if not SAFE_VALUE.match(env[args[1]].replace(" ", "")):
            print(f"unsafe value for {args[1]}", file=sys.stderr)
            return 2
        print(env[args[1]])
    elif cmd == "envdiff":
        ref = parse_env(Path(args[0]).read_text(errors="replace"))
        cur = parse_env(Path(args[1]).read_bytes().decode(errors="replace"))
        print("\n".join(env_diff(ref, cur)))
    elif cmd == "meta":
        print(json.loads((Path(args[0]) / "meta.json").read_text())[args[1]])
    else:
        print(f"unknown command {cmd}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    os.umask(0o022)
    sys.exit(main(sys.argv[1:]))
