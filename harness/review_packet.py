"""Build a blind review packet for a needs-review run (main user, before unblinding). Not a §9 file.

  pixi run python harness/review_packet.py --run <run_id> [--include-info] [--script PATH ...] [--release PATH ...]

Reads runs/wsp/<id>/audit.json, archive/rundir and the archived transcripts, and writes runs/wsp/<id>/review_packet.md
with ONLY:
  - the scripts named by the audit findings, plus every script connected to them by loading (exec/open, import,
    runpy/importlib, a path or module name in the code), followed transitively in both directions, including inline
    code in commands (python -c, `python - <<EOF`) and scripts created in the transcript but no longer in the run dir;
  - the commands (tool calls) that create, run, edit or load those scripts;
  - harness/path_review.py's re-resolution of outside_path findings against the shell's working directory, and
    heredoc-created scripts placed in the directory they were written to.

Never included: tool results, user/system/attachment entries (so no away_summary), NOTES.md, report*.md, NEXT_PHASE.md,
phase_a*.json, phase_b/ and any directory or file handed to wsp_submit, the main-user submissions/ dir, and the body of
any non-code heredoc. Command lines that mention one of those names are replaced by a marker; scripts that mention them
are withheld (listed by name with the reason) unless the operator releases them with --release. Reviews read only the packet.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

from wspg.audit import CODE_SUFFIXES, HEREDOC, heredoc_header, heredoc_kind
from wspg.utils import REPO_ROOT, git_hash
from wspg.wsp import RunPaths

BASE_FORBIDDEN = [r"NOTES\.md", r"NEXT_PHASE\.md", r"report[\w.-]*\.md", r"phase_a[\w.-]*\.json", r"\bphase_b\b",
                  r"\bwsp_submit\b", r"\bsubmissions/", r"away_summary"]
SKIP_ENTRY_TYPES = {"user", "system", "attachment", "summary"}
PY_INLINE = re.compile(r"python[\d.]*\s+(?:-\w+\s+)*-c\b")
ECHO_TO_FILE = re.compile(r"\b(echo|printf|tee)\b.*(>|\btee\b)")
IMPORT = re.compile(r"^[ \t]*(?:from[ \t]+([\w.]+)[ \t]+import[ \t]+(\([^)]*\)|[\w.,* \t]+)|import[ \t]+([\w.,* \t]+))",
                    re.M)


# ----------------------------------------------------------------------------- inputs
def tool_uses(archive: Path, run_dir: Path) -> list[dict]:
    """Every tool call in the transcripts, once, in file/line order: {loc, name, input}. Only tool_use blocks are read."""
    out: dict[str, dict] = {}
    files = sorted((archive / "claude_state" / "projects").rglob("*.jsonl"))
    for f in files:
        rel = str(f.relative_to(run_dir))
        for ln, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(e, dict) or e.get("type") in SKIP_ENTRY_TYPES:
                continue
            for k, b in enumerate(_tool_use_blocks(e)):
                key = b.get("id") or f"{rel}#L{ln}#{k}"
                if key not in out or (e.get("type") == "assistant" and not out[key]["assistant"]):
                    out[key] = {"loc": f"{rel}#L{ln}", "name": str(b.get("name", "")), "input": b.get("input") or {},
                                "assistant": e.get("type") == "assistant", "order": (files.index(f), ln, k)}
    return sorted(out.values(), key=lambda u: u["order"])


def _tool_use_blocks(o):
    if isinstance(o, dict):
        if o.get("type") == "tool_use":
            yield o
            return
        for v in o.values():
            yield from _tool_use_blocks(v)
    elif isinstance(o, list):
        for v in o:
            yield from _tool_use_blocks(v)


def heredoc_segments(cmd: str):
    """Yield ('line', text) for shell lines and ('doc', header, body, closing) for heredocs, in order."""
    lines = cmd.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        docs = list(HEREDOC.finditer(line))
        yield ("line", line)
        for m in docs:
            strip_tabs, delim = m.group(1) == "-", m.group(3)
            body, closing = [], None
            while i < len(lines):
                cur = lines[i].lstrip("\t") if strip_tabs else lines[i]
                i += 1
                if cur == delim:
                    closing = lines[i - 1]
                    break
                body.append(lines[i - 1])
            yield ("doc", heredoc_header(line, m.start()), "\n".join(body), closing)   # Amendment 7: own command


# ----------------------------------------------------------------------------- packet
class Packet:
    def __init__(self, run_id: str, root: Path = REPO_ROOT, include_info: bool = False,
                 extra_scripts=(), release=()):
        self.p = RunPaths(run_id, root)
        self.run_id = run_id
        self.rundir = self.p.archive / "rundir"
        self.audit = json.loads(self.p.audit.read_text())
        self.include_info = include_info
        self.release = {self._norm(r) for r in release}
        self.uses = tool_uses(self.p.archive, self.p.dir)
        from path_review import PathReview                 # late import: path_review imports this module
        self.paths = PathReview(run_id, root)               # working directory of every call, script run and heredoc
        self.forbidden = self._forbidden()
        self.scripts = self._scripts()
        self.withheld = {s: why for s in self.scripts if (why := self._withhold_reason(s)) and s not in self.release}
        self.extra = [self._norm(s) for s in extra_scripts]

    # -- names and paths
    def _norm(self, path: str) -> str:
        p = str(path).strip().strip("'\"")
        p = re.sub(rf"^/home/[^/]+/runs/{self.run_id}/", "", p)
        p = re.sub(r"^archive/rundir/", "", p)
        p = re.sub(r":\d+$", "", p)
        while p.startswith("./"):
            p = p[2:]
        return p

    def _resolve(self, path: str) -> str | None:
        p = self._norm(path)
        if p in self.scripts:
            return p
        hits = [s for s in self.scripts if Path(s).name == Path(p).name]
        return hits[0] if len(hits) == 1 else None

    def _forbidden(self) -> re.Pattern:
        pats = list(BASE_FORBIDDEN)
        for u in self.uses:
            cmd = u["input"].get("command") if u["name"] == "Bash" else None
            for m in re.finditer(r"wsp_submit\s+phase_[ab]\s+([^\n;&|]+)", cmd or ""):
                for tok in m.group(1).split():
                    name = self._norm(tok).rstrip("/")
                    if name and name not in (".", "..") and not name.startswith("-"):
                        pats.append(r"(?<![\w.-])" + re.escape(name) + r"(?![\w-])")
        return re.compile("|".join(pats))

    def _submission_dirs(self) -> list[str]:
        dirs = {"phase_b"}
        for u in self.uses:
            for m in re.finditer(r"wsp_submit\s+phase_b\s+(\S+)", u["input"].get("command") or "" if u["name"] == "Bash" else ""):
                dirs.add(self._norm(m.group(1)).rstrip("/"))
        return sorted(dirs)

    # -- scripts
    def _scripts(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        if self.rundir.exists():
            for f in sorted(self.rundir.rglob("*")):
                if f.is_file() and f.suffix in CODE_SUFFIXES:
                    text = f.read_text(errors="replace")
                    if f.suffix == ".ipynb":
                        try:
                            text = "\n".join("".join(c.get("source", "")) for c in json.loads(text).get("cells", []))
                        except json.JSONDecodeError:
                            pass
                    out[str(f.relative_to(self.rundir))] = {"text": text, "archived": True, "source": "run dir"}
        # scripts created in the transcript but absent from the archived run dir (deleted or moved)
        for u in self.uses:
            for target, body in self._code_writes(u):
                rel = self._norm(target)
                if Path(rel).suffix in CODE_SUFFIXES and (rel not in out or not out[rel]["archived"]):
                    out[rel] = {"text": body, "archived": False, "source": f"transcript {u['loc']} (not in archived run dir)"}
        return out

    def _code_writes(self, u: dict) -> list[tuple[str, str]]:
        inp = u["input"]
        if u["name"] == "Write" and inp.get("file_path"):
            return [(inp["file_path"], str(inp.get("content", "")))]
        if u["name"] == "Bash":
            out = []
            for seg in heredoc_segments(str(inp.get("command") or "")):
                if seg[0] == "doc":
                    kind, target = heredoc_kind(seg[1])
                    if kind in ("code", "shell") and Path(target).suffix in CODE_SUFFIXES:
                        # the file lands relative to the shell's working directory at that point (path_review)
                        out.append((self.paths.writes.get((u["loc"], target), target), seg[2]))
            return out
        return []

    def _withhold_reason(self, rel: str) -> str | None:
        top = rel.split("/")[0]
        if top in self._submission_dirs():
            return f"inside the submission directory {top}/"
        m = self.forbidden.search(self.scripts[rel]["text"])
        return f"mentions {m.group(0)!r}" if m else None

    def refs(self, text: str, self_rel: str | None = None) -> set[str]:
        """Scripts referenced by code text: a path or file name, a module import, or a quoted module name."""
        found = set()
        mods = _ast_imports(text)
        for m in IMPORT.finditer(text):
            if m.group(1):
                base = m.group(1)
                mods.add(base)
                mods.update(f"{base}.{n.strip()}" for n in re.split(r"[,\s()]+", m.group(2)) if n.strip() not in ("", "*"))
            else:
                mods.update(n.strip().split(" as ")[0] for n in m.group(3).split(","))
        for s in self.scripts:
            if s == self_rel:
                continue
            name, stem = Path(s).name, Path(s).stem
            dotted = s[: -len(Path(s).suffix)].replace("/", ".")
            if re.search(r"(?<![\w.-])" + re.escape(name) + r"(?!\w)", text) \
                    or any(mod in (stem, dotted) or mod.endswith("." + stem) for mod in mods) \
                    or re.search(r"['\"]" + re.escape(stem) + r"['\"]", text):
                found.add(s)
        return found

    # -- commands
    def command_parts(self, u: dict) -> tuple[str, str]:
        """(code-relevant text for mentions, inline code for load edges). Non-code heredoc bodies are excluded."""
        inp = u["input"]
        if u["name"] != "Bash":
            body = "\n".join(str(inp.get(k, "")) for k in ("file_path", "path", "notebook_path", "content",
                                                          "old_string", "new_string", "new_source"))
            return body, ""
        mention, inline = [], []
        for seg in heredoc_segments(str(inp.get("command") or "")):
            if seg[0] == "line":
                mention.append(seg[1])
                if PY_INLINE.search(seg[1]):
                    inline.append(seg[1])
            else:
                kind, target = heredoc_kind(seg[1])
                if kind in ("code", "shell"):
                    mention.append(seg[2])
                    if Path(target).suffix not in CODE_SUFFIXES:      # fed straight to an interpreter
                        inline.append(seg[2])
        return "\n".join(mention), "\n".join(inline)

    def build(self) -> dict:
        seeds_s, seeds_c, unresolved = set(), set(), []
        sevs = {"invalid", "review"} | ({"info"} if self.include_info else set())
        findings = [f for f in self.audit.get("findings", []) if f.get("severity") in sevs and f.get("category") != "environment"]
        for f in findings:
            where = str(f.get("where", ""))
            m = re.search(r"\((?:heredoc -> |Write |Edit |MultiEdit )([^)]+)\)", where)
            if where.startswith("archive/rundir/"):
                (seeds_s.add(r) if (r := self._resolve(where)) else unresolved.append(where))
            elif ".jsonl#L" in where:
                seeds_c.add(where.split(" ")[0])
                if m and (r := self._resolve(m.group(1))):
                    seeds_s.add(r)
        for s in self.extra:
            (seeds_s.add(r) if (r := self._resolve(s)) else unresolved.append(s))

        # load graph: scripts <-> scripts they reference; commands <-> scripts their inline code references
        adj: dict[str, set] = {s: set() for s in self.scripts}
        for s, info in self.scripts.items():
            for t in self.refs(info["text"], s):
                adj[s].add(t)
                adj[t].add(s)
        cmd_nodes = {}
        for i, u in enumerate(self.uses):
            node = f"cmd:{i}"
            cmd_nodes[node] = u
            adj[node] = set()
            for t in self.refs(self.command_parts(u)[1]):
                adj[node].add(t)
                adj[t].add(node)
        start = set(seeds_s) | {n for n, u in cmd_nodes.items() if u["loc"] in seeds_c}
        comp, q = set(start), deque(start)
        while q:
            for nb in adj[q.popleft()]:
                if nb not in comp:
                    comp.add(nb)
                    q.append(nb)
        scripts = sorted(s for s in comp if s in self.scripts)
        cmds = []
        for node, u in cmd_nodes.items():
            mentioned = self.refs(self.command_parts(u)[0]) & set(scripts)
            if node in comp or mentioned:
                cmds.append((u, sorted(mentioned)))
        return {"findings": findings, "seeds": sorted(seeds_s), "seed_cmds": sorted(seeds_c), "unresolved": unresolved,
                "scripts": scripts, "commands": cmds,
                "edges": sorted({(a, b) for a in scripts for b in adj[a] if b in self.scripts and b in scripts and a < b})}

    def render_command(self, u: dict, packet: set[str]) -> str:
        inp = u["input"]
        if u["name"] in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
            fp = str(inp.get("file_path") or inp.get("notebook_path") or "")
            rel = self._resolve(fp) or self._norm(fp)
            if rel in packet and rel not in self.withheld and not self.forbidden.search(json.dumps(inp)):
                body = {k: v for k, v in inp.items() if k in ("file_path", "content", "old_string", "new_string",
                                                              "replace_all", "edits", "new_source")}
                return json.dumps(body, indent=1)
            return f"# [{u['name']} {rel}: body withheld or not in packet]"
        if u["name"] != "Bash":
            keep = {k: inp[k] for k in ("file_path", "path", "pattern") if k in inp}
            s = json.dumps(keep)
            return f"# [{u['name']} call: input withheld]" if self.forbidden.search(s) else s
        out = []
        for seg in heredoc_segments(str(inp.get("command") or "")):
            if seg[0] == "line":
                out.append(self._line(seg[1], packet))
                continue
            header, body, closing = seg[1], seg[2], seg[3]
            kind, target = heredoc_kind(header)
            n = body.count("\n") + 1 if body else 0
            if kind == "text":
                out.append(f"# [non-code heredoc removed: {n} lines -> {target or 'stdin'}]")
            elif Path(target).suffix in CODE_SUFFIXES:
                at = self.paths.writes.get((u["loc"], target), target)
                rel = self._resolve(at) or self._norm(at)
                if rel in self.withheld or self.forbidden.search(body):
                    out.append(f"# [code heredoc for {rel} withheld: {self.withheld.get(rel, 'mentions a submission name')}]")
                elif rel not in packet:
                    out.append(f"# [code heredoc for {rel} (not in packet) omitted: {n} lines]")
                else:
                    arch = self.scripts.get(rel, {})
                    same = arch.get("archived") and arch["text"].rstrip("\n") == body.rstrip("\n")
                    out.append(f"# [heredoc body for {rel}: {'identical to' if same else 'DIFFERS from'} the archived file]")
                    out.append(body)
            elif self.forbidden.search(body):
                out.append(f"# [inline {kind} heredoc withheld: mentions a submission name, {n} lines]")
            elif kind == "shell":
                out.extend(self._line(x, packet) for x in body.split("\n"))
            else:
                out.append(body)
            if closing is not None:
                out.append(closing)
        return "\n".join(out)

    def _line(self, line: str, packet: set[str]) -> str:
        m = self.forbidden.search(line)
        if m or ECHO_TO_FILE.search(line):
            also = sorted(self.refs(line) & packet)
            why = f"mentions {m.group(0)!r}" if m else "echo/printf/tee to a file"
            return f"# [line removed: {why}" + (f"; also names packet script(s) {', '.join(also)}" if also else "") + "]"
        return line

    def render(self) -> str:
        b = self.build()
        packet = set(b["scripts"])
        L = [f"# Review packet, run {self.run_id}",
             "",
             f"Generated {datetime.now().isoformat(timespec='seconds')} by harness/review_packet.py at git "
             f"{git_hash()[:12]}. Contains only code and the commands that create, run, edit or load it; no tool "
             "results, no system/away_summary entries, no NOTES/report/phase_a/phase_b/submission content.",
             "", "## Audit findings used as seeds",
             f"Audit status: {self.audit.get('status')}; audit git {str(self.audit.get('audit_git_hash', ''))[:12]}.", ""]
        L += [f"- {f.get('severity')} / {f.get('category')}: {f.get('message')} — `{f.get('where')}`" for f in b["findings"]]
        if b["unresolved"]:
            L += ["", "**Unresolved seeds (no matching script):** " + ", ".join(f"`{x}`" for x in b["unresolved"])]
        pr = self.paths.report()
        if pr["outside_path_findings"] or pr["missed_by_audit"] or pr["commands_disagreeing"]:
            from path_review import render_md as paths_md
            L += ["", "## Working directory of path findings (harness/path_review.py; the audit is unchanged)",
                  "Relative paths re-resolved against the directory the shell was in (replayed `cd`, checked against the "
                  "cwd Claude Code recorded for each call). Status `cleared`: inside the run dir from every directory "
                  "the code ran in.", "", paths_md(pr).rstrip("\n")]
        L += ["", "## Load graph",
              f"Seed scripts: {', '.join(b['seeds']) or 'none'}. Seed commands: {', '.join(b['seed_cmds']) or 'none'}.",
              f"Scripts in the packet ({len(b['scripts'])} of {len(self.scripts)} in the run): {', '.join(b['scripts']) or 'none'}.",
              "References between packet scripts (either direction; path, module or quoted name in the code):"]
        L += [f"- {a} -- {c}" for a, c in b["edges"]] or ["- none"]
        withheld = [s for s in b["scripts"] if s in self.withheld]
        if withheld:
            L += ["", "**Withheld scripts (in the chain, body not shown; operator may release with --release):**"]
            L += [f"- {s}: {self.withheld[s]}" for s in withheld]
        if self.release:
            L += ["", f"Released by the operator: {', '.join(sorted(self.release))}."]
        L += ["", "## Scripts"]
        for s in b["scripts"]:
            info = self.scripts[s]
            users = [u["loc"].rsplit("#", 1)[-1] for u, m in b["commands"] if s in m]
            L += ["", f"### {s}", f"Source: {info['source']}. Named in commands: {', '.join(users) or 'none'}."]
            if s in self.withheld:
                L.append(f"_Withheld: {self.withheld[s]}._")
                continue
            numbered = "\n".join(f"{i:4d}  {x}" for i, x in enumerate(info["text"].split("\n"), 1))
            L += [_fence(numbered, "python")]
        L += ["", "## Commands (chronological; tool results never included)"]
        for u, mentioned in b["commands"]:
            L += ["", f"### {u['loc']} — {u['name']}" + (f" (names {', '.join(mentioned)})" if mentioned else " (loads packet code inline)"),
                  _fence(self.render_command(u, packet), "bash" if u["name"] == "Bash" else "json")]
        return "\n".join(L) + "\n"


def _ast_imports(text: str) -> set[str]:
    """Module names imported anywhere in parseable code (also after `;` or inside functions); empty if unparseable."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return set()
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module)
            mods.update(f"{n.module}.{a.name}" for a in n.names if a.name != "*")
    return mods


def _fence(text: str, lang: str) -> str:
    ticks = "`" * max(3, max((len(m) for m in re.findall(r"`+", text)), default=0) + 1)
    return f"{ticks}{lang}\n{text}\n{ticks}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", default=str(REPO_ROOT))
    ap.add_argument("--include-info", action="store_true", help="also seed from info findings")
    ap.add_argument("--script", action="append", default=[], help="extra seed script (path in the run dir)")
    ap.add_argument("--release", action="append", default=[], help="show a withheld script (operator decision)")
    ap.add_argument("--out", help="default: runs/wsp/<id>/review_packet.md")
    a = ap.parse_args(argv)
    pk = Packet(a.run, Path(a.root), a.include_info, a.script, a.release)
    out = Path(a.out) if a.out else pk.p.dir / "review_packet.md"
    out.write_text(pk.render())
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
