"""Compare how a run's session was launched with an interactive reference run (main user). Not a §9 file.

  pixi run python harness/launch_check.py --run <run_id> [--ref 50ce485a5146eb86]

Reads only start-up records of the archived transcripts (entries before the first assistant message: user prompt
metadata, attachments such as environment, sandbox instructions, model, tool and agent listings, instructions,
session context) plus every user entry's kind, and the audit's settings and model fields. It never prints assistant
text, tool results or system entries. Run ids, session ids, uuids, dates and timestamps are normalised before comparing.
Checks: exactly one human prompt, equal to the fixed prompt, with the same prompt metadata as the reference; no other
user message or local command (e.g. /exit) in the transcript; the same start-up record sequence and contents; the same
settings hash, model, effort and Claude Code version.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from wspg.utils import REPO_ROOT
from wspg.wsp import FIXED_PROMPT, RunPaths

PROMPT_KEYS = ("entrypoint", "origin", "promptSource", "turnOrigin", "turnPosition", "userType", "permissionMode",
               "isMeta", "version")
SKIP_COMPARE = {"date", "prompt_snapshot"}      # differ by construction (today's date; full request incl. timestamps)
VOLATILE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|\d{4}-\d{2}-\d{2}T[\d:.]+Z?|"
                      r"\b\d{4}-\d{2}-\d{2}\b|\b1\d{12}\b")


def transcript(rp: RunPaths) -> list[dict]:
    files = sorted((rp.archive / "claude_state" / "projects").rglob("*.jsonl"))
    main = [f for f in files if "subagents" not in f.parts]
    if len(main) != 1:
        raise SystemExit(f"{rp.run_id}: expected one main transcript, found {len(main)}")
    out = []
    for line in main[0].read_text(errors="replace").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(e, dict):
            out.append(e)
    return out


def _text(content) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list) and not any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return None


def user_entries(entries: list[dict]) -> list[dict]:
    """Every user entry that is not a tool result: kind (prompt/command/meta/reminder) and metadata; text only for
    the fixed-prompt comparison and for local commands (their name)."""
    out = []
    for e in entries:
        if e.get("type") != "user":
            continue
        t = _text((e.get("message") or {}).get("content"))
        if t is None:
            continue
        s = t.strip()
        if e.get("isMeta") or s.startswith("<system-reminder>"):
            kind = "meta"
        elif "<command-name>" in s or s.startswith(("<local-command", "/")):
            kind = "command"
        elif not s:
            continue
        else:
            kind = "prompt"
        rec = {"kind": kind, "meta": {k: e.get(k) for k in PROMPT_KEYS}}
        if kind == "prompt":
            rec["is_fixed_prompt"] = s == FIXED_PROMPT
            rec["text"] = s if s == FIXED_PROMPT else f"<{len(s)} chars, not the fixed prompt>"
        elif kind == "command":
            m = re.search(r"<command-name>([^<]*)</command-name>", s)
            rec["text"] = m.group(1) if m else s.split()[0][:40]
        out.append(rec)
    return out


def startup(entries: list[dict], run_id: str) -> list[tuple[str, str]]:
    """(record type, normalised content) for entries before the first assistant message; user text is not included."""
    out = []
    for e in entries:
        t = e.get("type")
        if t == "assistant":
            break
        if t == "attachment" and isinstance(e.get("attachment"), dict):
            a = e["attachment"]
            label = f"attachment:{a.get('type')}"
            body = "" if a.get("type") in SKIP_COMPARE else json.dumps(a, sort_keys=True)
        elif t in ("user", "system"):
            label, body = str(t), ""
        else:
            label = str(t)
            body = json.dumps({k: v for k, v in e.items() if k in ("mode", "permissionMode")}, sort_keys=True)
        out.append((label, VOLATILE.sub("<v>", body.replace(run_id, "<run>"))))
    return out


def compare(run_id: str, ref_id: str, root: Path = REPO_ROOT) -> list[tuple[bool, str]]:
    rp, rr = RunPaths(run_id, root), RunPaths(ref_id, root)
    new, ref = transcript(rp), transcript(rr)
    res: list[tuple[bool, str]] = []
    nu, ru = user_entries(new), user_entries(ref)
    prompts = [u for u in nu if u["kind"] == "prompt"]
    res.append((len(prompts) == 1 and prompts[0]["is_fixed_prompt"],
                f"one human prompt, equal to the fixed prompt: {[p['text'] for p in prompts]}"))
    rp_meta = next((u["meta"] for u in ru if u["kind"] == "prompt"), None)
    if prompts:
        diff = {k: (prompts[0]["meta"][k], rp_meta[k]) for k in PROMPT_KEYS if rp_meta and prompts[0]["meta"][k] != rp_meta[k]}
        res.append((not diff, f"prompt metadata equal to the reference {'' if not diff else diff}".rstrip()))
    others = [u for u in nu if u["kind"] != "prompt" and u not in ru]
    res.append((not others, f"no other user messages or local commands: {[(u['kind'], u.get('text')) for u in others]}"))
    ns, rs = startup(new, run_id), startup(ref, ref_id)
    res.append(([s[0] for s in ns] == [s[0] for s in rs], "same start-up record sequence"
                + ("" if [s[0] for s in ns] == [s[0] for s in rs] else f": {[s[0] for s in ns]} vs {[s[0] for s in rs]}")))
    rmap: dict[str, list[str]] = {}
    for label, body in rs:
        rmap.setdefault(label, []).append(body)
    seen: dict[str, int] = {}
    for label, body in ns:
        k = seen.get(label, 0)
        seen[label] = k + 1
        refs = rmap.get(label, [])
        if body and k < len(refs) and refs[k] != body:
            hint = " (rendered AGENTS.md: expected when budgets or phases differ)" if label == "attachment:instructions" else ""
            res.append((False, f"start-up record {label} #{k + 1} differs from the reference{hint}"))
    if all(ok for ok, msg in res if msg.startswith("start-up record ")):
        res.append((True, "start-up records equal after normalising run/session ids and dates "
                          f"(not compared: {', '.join(sorted(SKIP_COMPARE))})"))
    na, ra = json.loads(rp.audit.read_text()), json.loads(rr.audit.read_text())
    for key in ("settings_sha256", "models", "efforts", "prompts"):
        a, b = na["stats"].get(key), ra["stats"].get(key)
        res.append((a == b, f"audit {key} equal: {a}" + ("" if a == b else f" vs reference {b}")))
    vm, rm = json.loads(rp.meta.read_text()), json.loads(rr.meta.read_text())
    res.append((vm.get("claude_version") == rm.get("claude_version"),
                f"Claude Code version {vm.get('claude_version')} (reference {rm.get('claude_version')})"))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ref", default="50ce485a5146eb86", help="interactive reference run (default: toy dry run 3)")
    ap.add_argument("--root", default=str(REPO_ROOT))
    a = ap.parse_args(argv)
    res = compare(a.run, a.ref, Path(a.root))
    for ok, msg in res:
        print(("PASS " if ok else "DIFF ") + msg)
    bad = sum(not ok for ok, _ in res)
    print(f"launch check {a.run} vs {a.ref}: {'match' if not bad else f'{bad} difference(s)'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
