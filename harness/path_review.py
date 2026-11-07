"""Working-directory re-resolution of the audit's path findings (review procedure, main user). Not a §9 file: the audit
(src/wspg/audit.py) is unchanged and its report is not rewritten.

  pixi run python harness/path_review.py --run <run_id> [--json PATH] [--summary]

Why: the audit resolves every relative path against the run directory. The session's shell keeps its working directory
between Bash calls, so after `cd analysis`, a script that opens '../weights.npz' reads <run dir>/weights.npz, which the
audit reports as an access to <runs>/weights.npz. This tool

  1. replays the transcript's tool calls in order and follows `cd` in command position (a `( ... )` subshell is scoped;
     a call that ends outside the run dir is reset to it, as Claude Code does), and checks every call against the `cwd`
     Claude Code records on the tool_result entry (the directory after the call) and on the previous one (before it);
  2. records the working directory of every script execution (python <file>, python -m <module>, bash <file>, ./<file>)
     and of the scripts those load (a quoted .py path in the code, imports), transitively;
  3. re-runs the audit's own path checks (Auditor.scan_command_paths / scan_code: same regexes, same allow and review
     lists) with relative paths resolved against those directories;
  4. matches each audit outside_path finding to its re-resolution: `cleared` (inside the run dir or on the allow list
     from every directory it ran in), `confirmed` (still outside), `review list`, `not executed` (a script never run in
     the transcript: left as the audit has it), or `undetermined` (no match, or the replayed directory is not certain or
     disagrees with the recorded one); and lists paths outside the run dir that the audit resolved as inside.

Reads only tool_use inputs and the type/cwd/tool_use_id fields of transcript entries; never tool results, user text or
system entries. Path tokens that mention a submission name are withheld from the output.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import posixpath
import re
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_packet import BASE_FORBIDDEN, heredoc_segments  # noqa: E402

import wspg.audit  # noqa: E402
from wspg.audit import CODE_SUFFIXES, SKIP_FILES, Auditor, expand_shell_env, heredoc_kind  # noqa: E402
from wspg.utils import REPO_ROOT, load_config  # noqa: E402
from wspg.wsp import RunPaths, write_json_atomic  # noqa: E402

OPS = {"&&", "||", ";", "|", "&", "|&", ";;", "\n"}
def tool_sha256() -> str:
    """Version of this review: path_review.py together with the audit checks it reuses (Amendment 6) and the heredoc
    splitting of review_packet.py (Amendment 7)."""
    files = [Path(__file__), Path(wspg.audit.__file__), Path(sys.modules[heredoc_segments.__module__].__file__)]
    return hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()


REDIR = {">", ">>", "<", "<<", "<<<", ">&", "&>", "<&", ">|", "&>>", "<<-"}
PREFIX_WORDS = {"!", "{", "}", "then", "do", "else", "elif", "if", "while", "until", "time", "fi", "done", "nohup",
                "timeout", "command", "builtin"}
INTERP = re.compile(r"(?:.*/)?python[\d.]*")
SHELLS = re.compile(r"(?:.*/)?(?:bash|sh|zsh|dash)")
PY_LIT = re.compile(r"""['"]([^'"\s]+\.(?:py|sh))['"]""")
SYS_PATH = re.compile(r"""sys\.path\.(?:insert\(\s*\d+\s*,|append\()\s*['"]([^'"]+)['"]""")
OK_CLASSES = ("run dir", "allowed")


class Resolver(Auditor):
    """The audit's path checks with a settable working directory. Only paths are recorded; other findings are dropped."""

    def __init__(self, cfg: dict, run_id: str):           # noqa: super().__init__ needs meta; only path state is used
        self.cfg = cfg
        self.live_dir = f"{cfg['agent_runs']}/{run_id}"
        self.allow = list(cfg["audit"]["allow_prefixes"])
        self.review_prefixes = list(cfg["audit"].get("review_prefixes", []))
        self.findings, self.stats = [], {}
        self.cwd = self.live_dir
        self.basis = "command"
        self.records: list[dict] = []

    def add(self, *a, **k) -> None:
        pass

    def resolve_from(self, p: str, base: str) -> str:
        saved, self.live_dir = self.live_dir, base
        try:
            return Auditor._norm(self, p)
        finally:
            self.live_dir = saved

    def _norm(self, p: str) -> str:
        return self.resolve_from(p, self.cwd)

    def classify(self, p: str) -> str:
        if p == self.live_dir or p.startswith(self.live_dir + "/"):
            return "run dir"
        if any(p == a or p.startswith(a.rstrip("/") + "/") for a in self.allow):
            return "allowed"
        if any(p == a or p.startswith(a.rstrip("/") + "/") for a in self.review_prefixes):
            return "review list"
        return "outside"

    def check_path(self, raw: str, where: str, context: str = "") -> None:
        self.records.append({"where": where, "raw": raw, "cwd": self.cwd, "basis": self.basis,
                             "audit_path": Auditor._norm(self, raw), "path": self._norm(raw)})


class PathReview:
    def __init__(self, run_id: str, root: Path = REPO_ROOT, cfg: dict | None = None):
        if not re.fullmatch(r"[0-9a-f]{16}", run_id):
            raise SystemExit(f"bad run id {run_id!r}")
        self.run_id = run_id
        self.p = RunPaths(run_id, root)
        self.cfg = cfg or load_config(REPO_ROOT / "configs" / "harness.yaml")
        self.r = Resolver(self.cfg, run_id)
        self.live = self.r.live_dir
        self.audit = json.loads(self.p.audit.read_text()) if self.p.audit.exists() else {"findings": []}
        self.calls, self.starts = self._transcript()
        self.forbidden = self._forbidden()
        self.units: list[dict] = []            # code units the audit scans: label, abs path (None: inline), text, kind
        self.versions: dict[str, str] = {}     # abs path -> last written text (transcript order)
        self.execs: list[dict] = []            # {abs, cwd, loc, certain}
        self.writes: dict[tuple[str, str], str] = {}   # (loc, raw heredoc/Write target) -> abs path
        self.commands: list[dict] = []
        self.notes: list[str] = []
        self._archived_units()
        self._replay()
        self._scan_units()

    # ------------------------------------------------------------------ inputs
    def _transcript(self):
        files = sorted((self.p.archive / "claude_state" / "projects").rglob("*.jsonl"))
        calls, seen, starts, result_cwd, result_err = [], set(), {}, {}, {}
        for f in files:
            rel = str(f.relative_to(self.p.dir))
            for ln, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(e, dict):
                    continue
                cwd = e.get("cwd") if isinstance(e.get("cwd"), str) else None
                if cwd and rel not in starts:
                    starts[rel] = cwd
                msg = e.get("message") if isinstance(e.get("message"), dict) else {}
                blocks = msg.get("content") if isinstance(msg.get("content"), list) else []
                if e.get("type") == "user":
                    for b in blocks:
                        if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id"):
                            if cwd:
                                result_cwd.setdefault(b["tool_use_id"], cwd)
                            result_err.setdefault(b["tool_use_id"], bool(b.get("is_error")))
                    continue
                if e.get("type") != "assistant":
                    continue
                for k, b in enumerate(blocks):
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        key = b.get("id") or f"{rel}#L{ln}#{k}"
                        if key in seen:
                            continue
                        seen.add(key)
                        calls.append({"file": rel, "loc": f"{rel}#L{ln}", "id": b.get("id"), "name": str(b.get("name", "")),
                                      "input": b.get("input") if isinstance(b.get("input"), dict) else {}})
        for c in calls:
            c["cwd_after"] = result_cwd.get(c["id"])
            c["failed"] = result_err.get(c["id"], False)
        return calls, starts

    def _forbidden(self) -> re.Pattern:
        pats = list(BASE_FORBIDDEN) + [r"[ACGTacgt]{6,}"]
        for c in self.calls:
            if c["name"] == "Bash":
                for m in re.finditer(r"wsp_submit\s+phase_[ab]\s+([^\n;&|]+)", str(c["input"].get("command") or "")):
                    for tok in m.group(1).split():
                        name = self.rel(self.r.resolve_from(tok.strip("'\""), self.live)).rstrip("/")
                        if name and name not in (".", "..") and not name.startswith("-"):
                            pats.append(r"(?<![\w.-])" + re.escape(name) + r"(?![\w-])")
        return re.compile("|".join(pats))

    def _archived_units(self) -> None:
        rd = self.p.archive / "rundir"
        if not rd.exists():
            return
        for f in sorted(rd.rglob("*")):
            if not f.is_file() or (f.name in SKIP_FILES and f.parent == rd):
                continue
            try:
                head = f.read_bytes()[:2]
            except OSError:
                continue
            if f.suffix in CODE_SUFFIXES or (f.suffix == "" and head == b"#!"):
                text = f.read_text(errors="replace")
                if f.suffix == ".ipynb":
                    try:
                        text = "\n".join("".join(c.get("source", "")) for c in json.loads(text).get("cells", []))
                    except json.JSONDecodeError:
                        pass
                shell = f.suffix in (".sh", ".bash") or (f.suffix == "" and head == b"#!")
                self.units.append({"label": str(f.relative_to(self.p.dir)), "abs": f"{self.live}/{f.relative_to(rd)}",
                                   "text": text, "kind": "shell" if shell else "code", "archived": True})

    # ------------------------------------------------------------------ helpers
    def rel(self, p: str | None) -> str:
        if p is None:
            return "?"
        if p == self.live:
            return "run dir"
        if p.startswith(self.live + "/"):
            return p[len(self.live) + 1:]
        return p

    def reldir(self, p: str | None) -> str:
        r = self.rel(p)
        return r if r in ("run dir", "?") or r.startswith("/") else r + "/"

    def show(self, s: str) -> str:
        m = self.forbidden.search(s)
        return f"[withheld: mentions {m.group(0)!r}]" if m else s

    def _inside(self, p: str) -> bool:
        return p == self.live or p.startswith(self.live + "/")

    # ------------------------------------------------------------------ replay
    def _replay(self) -> None:
        cwd: dict[str, str] = {}
        prev_after: dict[str, str | None] = {}
        for c in self.calls:
            f = c["file"]
            start = self.starts.get(f, self.live)
            cur = cwd.get(f, start)
            before_rec = prev_after.get(f, start)
            inp, name, loc = c["input"], c["name"], c["loc"]
            if name == "Bash":
                cmd = str(inp.get("command") or "")
                st = {"cwd": cur, "certain": True, "cds": [], "moved": False, "noop_cds": 0}
                self._shell(cmd, loc, st, top=True)
                after = st["cwd"] if self._inside(st["cwd"]) else self.live     # Claude Code resets outside the run dir
                # Amendment 10: only a cd that can change the directory makes a failed call's paths uncertain; a cd to
                # the directory the shell is already in (before any other cd of the call) leaves it the same either way
                failed_cd = bool(c["failed"] and st["moved"])
                if failed_cd:
                    # Claude Code keeps a cd only when the whole call succeeds: the kept cwd is the one before the call.
                    after = cur
                agree = (before_rec is None or before_rec == cur) and (c["cwd_after"] is None or c["cwd_after"] == after)
                if not agree or not st["certain"]:
                    for rec in st.get("recs", []):
                        rec["basis"] = "undetermined"
                    for e in st.get("execs", []):
                        e["certain"] = False
                    kept = 0
                elif failed_cd:
                    # In a failed call a cd may itself have failed. Amendment 7: a path joined to every cd before it by
                    # `&&` runs only if those cds succeeded, so it is resolved against their directory; a path reached
                    # after `;`, a newline, `||` or `&` that follows a cd may run after a failed cd and is undetermined.
                    for rec in st.get("recs", []):
                        if rec.get("exposed", True):
                            rec["basis"] = "undetermined"
                    for e in st.get("execs", []):
                        if e.get("exposed", True):
                            e["certain"] = False
                    kept = sum(1 for rec in st.get("recs", []) if not rec.get("exposed", True))
                else:
                    kept = 0
                self.commands.append({"loc": loc, "before": cur, "before_recorded": before_rec, "after": after,
                                      "after_recorded": c["cwd_after"], "cds": st["cds"],
                                      "certain": st["certain"] and not failed_cd, "failed": c["failed"], "agree": agree,
                                      "and_chain_paths": kept, "noop_cds": st["noop_cds"]})
                cwd[f] = c["cwd_after"] or after
            else:
                self.r.cwd, self.r.basis = cur, "command"
                for key in ("file_path", "path", "notebook_path"):
                    if isinstance(inp.get(key), str) and inp[key]:
                        self.r.check_path(inp[key], loc, name)
                if name == "Glob" and isinstance(inp.get("pattern"), str) and inp["pattern"].startswith(("/", "~", "..")):
                    self.r.check_path(inp["pattern"].split("*")[0] or "/", loc, name)
                if name in ("Write", "Edit", "NotebookEdit", "MultiEdit"):
                    fp = str(inp.get("file_path") or inp.get("notebook_path") or "")
                    suffix = Path(fp).suffix
                    body = "\n".join(str(inp.get(k, "")) for k in ("content", "new_string", "new_source"))
                    for ed in inp.get("edits") or []:
                        if isinstance(ed, dict):
                            body += "\n" + str(ed.get("new_string", ""))
                    ab = self.r.resolve_from(fp, cur) if fp else None
                    if ab:
                        self.writes[(loc, fp)] = ab
                        if name == "Write":
                            self.versions[ab] = str(inp.get("content", ""))
                    if suffix in CODE_SUFFIXES or suffix == "" or body.startswith("#!"):
                        self.units.append({"label": f"{loc} ({name} {Path(fp).name})", "abs": ab, "text": body,
                                           "kind": "shell" if suffix in (".sh", ".bash") else "code", "archived": False})
                cwd[f] = cur
            prev_after[f] = c["cwd_after"]

    def _shell(self, cmd: str, where: str, st: dict, top: bool) -> None:
        """Walk a shell command: the command text (heredoc bodies removed, as the audit splits it) is parsed as a whole,
        so quotes may span lines; `cd` updates st['cwd']; paths are checked at the current cwd; each heredoc body is
        handled with the cwd at its `<<` operator."""
        segs = list(heredoc_segments(cmd))
        outer = "\n".join(s[1] for s in segs if s[0] == "line")
        docs = [s for s in segs if s[0] == "doc"]
        doc_cwds = self._line(outer, where, st)
        if len(doc_cwds) != len(docs):
            st["certain"] = st["certain"] and not docs
            doc_cwds = [(st["cwd"], True)] * len(docs)
        end_exposed = st.get("exposed", False)
        for seg, (here, exposed) in zip(docs, doc_cwds):
            st["exposed"] = exposed                                 # the body runs at its `<<` operator
            header, body = seg[1], seg[2]
            kind, target = heredoc_kind(header)
            label = f"{where} (heredoc -> {target or kind})"
            if kind == "text":
                continue
            if Path(target).suffix in CODE_SUFFIXES or (kind == "shell" and target and not SHELLS.fullmatch(target)):
                ab = self.r.resolve_from(target, here)
                self.writes[(where, target)] = ab
                self.versions[ab] = body
                self.units.append({"label": label, "abs": ab, "text": body, "kind": kind, "archived": False})
            elif kind == "shell":                                   # bash <<EOF: a child shell, cwd not kept
                sub = {"cwd": here, "certain": st["certain"], "cds": st["cds"], "exposed": exposed,
                       "moved": st.get("moved", False), "noop_cds": 0}
                self._shell(body, label, sub, top=False)
                st["certain"] = st["certain"] and sub["certain"]
                st["moved"] = st.get("moved", False) or sub["moved"]     # as before Amendment 10: any cd in the call
                st["noop_cds"] = st.get("noop_cds", 0) + sub["noop_cds"]
                st.setdefault("recs", []).extend(sub.get("recs", []))
                st.setdefault("execs", []).extend(sub.get("execs", []))
            else:                                                   # python - <<EOF: inline code at this cwd
                self._check(lambda: self.r.scan_code(body, label), here, "command", st)
                self._loads(body, None, here, where, st, depth=0)
        st["exposed"] = end_exposed

    def _check(self, fn, cwd: str, basis: str, st: dict | None) -> list[dict]:
        n = len(self.r.records)
        self.r.cwd, self.r.basis = cwd, basis
        fn()
        new = self.r.records[n:]
        if st is not None:
            for rec in new:
                rec.setdefault("exposed", st.get("exposed", False))
            st.setdefault("recs", []).extend(new)
        return new

    def _line(self, line: str, where: str, st: dict) -> list[str]:
        """Shell text (newlines separate commands): returns the cwd at each heredoc operator (in order)."""
        try:
            toks = _tokens(line)
        except ValueError:                                          # unbalanced quotes: check the text as the audit does
            self._check(lambda: self.r.scan_command_paths(line, where), st["cwd"], "command", st)
            if re.search(r"(?:^|[\s;&|(])(?:cd|pushd|popd)\b", line):
                st["certain"] = False
            return []
        stack, words, docs = [], [], []
        comment = False

        def flush():
            if not words:
                return
            text = " ".join(t if t in REDIR else shlex.quote(t) for t in words)
            self._check(lambda: self.r.scan_command_paths(text, where), st["cwd"], "command", st)
            if any(t.startswith("<<") and t != "<<<" for t in words):
                docs.extend([(st["cwd"], st.get("exposed", False))] * sum(1 for t in words if t in ("<<", "<<-")))
            self._simple(words, where, st)
            words.clear()

        for t in toks:
            if comment:                                             # `# ...` up to the end of the line
                comment = t != "\n"
                if comment:
                    continue
            if t.startswith("#") and not (words and words[-1] in REDIR):
                comment = True
                continue
            if t in OPS:
                flush()
                if t not in ("&&", "|", "|&") and st.get("moved", bool(st["cds"])):
                    st["exposed"] = True                            # Amendment 7: may run after a failed cd
            elif t == "(":
                flush()
                stack.append(st["cwd"])
            elif t == ")":
                flush()
                if stack:
                    st["cwd"] = stack.pop()
            else:
                words.append(t)
        flush()
        if stack:
            st["certain"] = False
        return docs

    def _simple(self, words: list[str], where: str, st: dict) -> None:
        w, i = [], 0
        while i < len(words):                                       # drop redirections and their targets
            if words[i] in REDIR:
                i += 2
                continue
            w.append(words[i])
            i += 1
        while w and (w[0] in PREFIX_WORDS or re.fullmatch(r"[A-Za-z_]\w*=.*", w[0]) or re.fullmatch(r"\d+[smhd]?", w[0])):
            w.pop(0)
        if not w:
            return
        if w[0] == "cd":
            args = [a for a in w[1:] if not re.fullmatch(r"-[LPe@]+", a)]
            d = args[0] if args else "~"
            st["cds"].append(d)
            if "$" in d:
                d = expand_shell_env(d, self.r.env_paths())     # Amendment 6: cd $TMPDIR
            if d == "-" or any(ch in d for ch in "$`*?"):
                st["certain"] = False
                st["moved"] = True
                return
            new = self.r.resolve_from(d, st["cwd"])
            if not st.get("moved", bool(st["cds"][:-1])) and new == st["cwd"]:
                st["noop_cds"] = st.get("noop_cds", 0) + 1      # Amendment 10: to the directory it is already in
                return
            st["cwd"] = new
            st["moved"] = True
            return
        if w[0] in ("pushd", "popd"):
            st["cds"].append(w[0])
            st["certain"] = False
            st["moved"] = True
            return
        for j, t in enumerate(w):
            if INTERP.fullmatch(t):
                k = j + 1
                while k < len(w) and w[k].startswith("-") and w[k] not in ("-c", "-m", "-"):
                    k += 1
                if k >= len(w) or w[k] in ("-c", "-"):
                    if k < len(w) - 1 and w[k] == "-c":
                        self._loads(w[k + 1], None, st["cwd"], where, st, depth=0)
                    return
                if w[k] == "-m" and k + 1 < len(w):
                    self._exec(self.r.resolve_from(w[k + 1].replace(".", "/") + ".py", st["cwd"]), st, where)
                    return
                self._exec(self.r.resolve_from(w[k], st["cwd"]), st, where)
                return
            if SHELLS.fullmatch(t) and j + 1 < len(w) and not w[j + 1].startswith("-"):
                self._exec(self.r.resolve_from(w[j + 1], st["cwd"]), st, where)
                return
            if j == 0 and t.startswith("./") and Path(t).suffix in (".py", ".sh", ""):
                self._exec(self.r.resolve_from(t, st["cwd"]), st, where)
                return

    def _exec(self, ab: str, st: dict, where: str) -> None:
        e = {"abs": ab, "cwd": st["cwd"], "loc": where, "certain": st["certain"], "via": None,
             "exposed": st.get("exposed", False)}
        self.execs.append(e)
        st.setdefault("execs", []).append(e)
        self._loads(self._text(ab), ab, st["cwd"], where, st, depth=1)

    def _text(self, ab: str) -> str | None:
        if ab in self.versions:
            return self.versions[ab]
        for u in self.units:
            if u["archived"] and u["abs"] == ab:
                return u["text"]
        return None

    def _known(self, ab: str) -> bool:
        return ab in self.versions or any(u["abs"] == ab for u in self.units)

    def _loads(self, text: str | None, script: str | None, cwd: str, where: str, st: dict, depth: int,
               seen: set | None = None) -> None:
        """Scripts loaded by code running with this cwd: quoted .py/.sh paths (open/exec/runpy) and imports."""
        if not text or depth > 10:
            return
        seen = seen if seen is not None else ({script} if script else set())
        found = {self.r.resolve_from(m.group(1), cwd) for m in PY_LIT.finditer(text)}
        dirs = [cwd] + ([posixpath.dirname(script)] if script else []) + \
               [self.r.resolve_from(m.group(1), cwd) for m in SYS_PATH.finditer(text)]
        try:
            tree = ast.parse(text)
            mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | \
                   {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        except (SyntaxError, ValueError):
            mods = set()
        for m in mods:
            for d in dirs:
                found.add(posixpath.normpath(posixpath.join(d, m.replace(".", "/") + ".py")))
        for ab in sorted(found):
            if ab in seen or not self._known(ab):
                continue
            seen.add(ab)
            e = {"abs": ab, "cwd": cwd, "loc": where, "certain": st["certain"], "via": script or "inline code",
                 "exposed": st.get("exposed", False)}
            self.execs.append(e)
            st.setdefault("execs", []).append(e)
            self._loads(self._text(ab), ab, cwd, where, st, depth + 1, seen)

    # ------------------------------------------------------------------ code units
    def _scan_units(self) -> None:
        for u in self.units:
            runs = {}
            for e in self.execs:
                if u["abs"] and e["abs"] == u["abs"]:
                    runs[e["cwd"]] = runs.get(e["cwd"], True) and e["certain"]
            u["cwds"] = sorted(runs)
            if not runs:
                targets = [(self.live, "not executed")]
            else:
                targets = [(c, "executed" if ok else "undetermined") for c, ok in sorted(runs.items())]
            for cwd, basis in targets:
                if u["kind"] == "shell":
                    st = {"cwd": cwd, "certain": basis != "undetermined", "cds": []}
                    self._shell(u["text"], u["label"], st, top=False)
                    for rec in st.get("recs", []):
                        rec["basis"] = basis if st["certain"] or basis == "not executed" else "undetermined"
                else:
                    self._check(lambda: self.r.scan_code(u["text"], u["label"]), cwd, basis, None)

    # ------------------------------------------------------------------ report
    def report(self) -> dict:
        recs: dict[tuple, dict] = {}
        for r in self.r.records:
            recs.setdefault((r["where"], r["raw"], r["cwd"], r["basis"]), r)
        by_where: dict[tuple, list[dict]] = {}
        for r in recs.values():
            by_where.setdefault((r["where"], r["audit_path"]), []).append(r)
        out = []
        for f in self.audit.get("findings", []):
            if f.get("category") != "outside_path":
                continue
            path = str(f.get("message", "")).split(": ", 1)[-1]
            rs = by_where.get((f.get("where"), path), [])
            classes = {self.r.classify(r["path"]) for r in rs}
            bases = {r["basis"] for r in rs}
            if not rs:
                status = "undetermined"
                why = "no matching path in the replay"
            elif "undetermined" in bases:
                status, why = "undetermined", "working directory not certain or not matching the recorded one"
            elif bases == {"not executed"}:
                status, why = "not executed", "script never run in the transcript; resolved against the run dir as in the audit"
            elif classes <= set(OK_CLASSES):
                status, why = "cleared", "inside the run dir (or allowed) from every working directory it ran in"
            elif "outside" in classes:
                status, why = "confirmed", "outside the run dir from the working directory it ran in"
            else:
                status, why = "review list", "on the audit's review list from the working directory it ran in"
            out.append({"severity": f.get("severity"), "where": f.get("where"), "audit_path": self.show(path),
                        "status": status, "why": why, "raw": sorted({self.show(r["raw"]) for r in rs}),
                        "cwds": sorted({self.reldir(r["cwd"]) for r in rs if r["basis"] != "not executed"}),
                        "paths": sorted({self.show(self.rel(r["path"])) for r in rs})})
        missed = []
        for r in recs.values():
            if r["basis"] == "not executed" or r["path"] == r["audit_path"]:
                continue
            new_c, old_c = self.r.classify(r["path"]), self.r.classify(r["audit_path"])
            if new_c in ("outside", "review list") and old_c in OK_CLASSES:
                missed.append({"where": r["where"], "raw": self.show(r["raw"]), "cwd": self.reldir(r["cwd"]),
                               "path": self.show(r["path"]), "class": new_c, "basis": r["basis"]})
        invalid = [f for f in self.audit.get("findings", []) if f.get("severity") == "invalid"]
        inv_cleared = [o for o in out if o["severity"] == "invalid" and o["status"] == "cleared"]
        counts = {s: sum(o["status"] == s for o in out) for s in
                  ("cleared", "confirmed", "review list", "not executed", "undetermined")}
        cmds = [{**c, "before": self.reldir(c["before"]), "before_recorded": self.reldir(c["before_recorded"]),
                 "after": self.reldir(c["after"]), "after_recorded": self.reldir(c["after_recorded"]),
                 "cds": [self.show(d) for d in c["cds"]]}
                for c in self.commands if c["cds"] or not c["agree"] or not c["certain"]]
        return {"run_id": self.run_id, "tool_sha256": tool_sha256(),
                "audit_status": self.audit.get("status"), "audit_git_hash": self.audit.get("audit_git_hash"),
                "outside_path_findings": out, "counts": counts,
                "invalid_findings": len(invalid), "invalid_outside_path_cleared": len(inv_cleared),
                "all_invalid_cleared": bool(invalid) and len(inv_cleared) == len(invalid),
                "missed_by_audit": missed,
                "commands_with_cd": cmds,
                "commands_disagreeing": sum(1 for c in self.commands if not c["agree"]),
                "commands_uncertain": sum(1 for c in self.commands if not c["certain"]),
                "commands": len(self.commands), "notes": self.notes}


PUNCT = "();<>|&\n"
OPERATORS = sorted(["&&", "||", ";;", "|&", "<<<", "<<-", "<<", ">>", "&>>", "&>", ">&", "<&", ">|", "(", ")", ";", "|",
                    "&", "<", ">", "\n"], key=len, reverse=True)


def _tokens(text: str) -> list[str]:
    """shlex words and shell operators; newline is an operator; adjacent operator characters are split ("&&\\n" ->
    "&&", "\\n"). Comments are left to the caller (a word starting with '#')."""
    lx = shlex.shlex(text, posix=True, punctuation_chars=PUNCT)
    lx.whitespace = " \t\r"
    lx.whitespace_split = True
    lx.commenters = ""
    out = []
    for t in lx:
        if t and all(ch in PUNCT for ch in t):
            while t:
                op = next((o for o in OPERATORS if t.startswith(o)), t[0])
                out.append(op)
                t = t[len(op):]
        else:
            out.append(t)
    return out


def summary_line(r: dict) -> str:
    n = len(r["outside_path_findings"])
    c = r["counts"]
    inv = ("all invalid findings cleared" if r["all_invalid_cleared"]
           else f"{r['invalid_findings'] - r['invalid_outside_path_cleared']} invalid finding(s) remain")
    return (f"path_review {r['run_id']}: {n} outside_path finding(s): {c['cleared']} cleared, {c['confirmed']} confirmed, "
            f"{c['review list']} review list, {c['not executed']} not executed, {c['undetermined']} undetermined; "
            f"{len(r['missed_by_audit'])} missed by the audit; {inv}; cwd replay {r['commands'] - r['commands_disagreeing']}"
            f"/{r['commands']} Bash calls match the recorded cwd")


def render_md(r: dict) -> str:
    L = [f"Audit status {r['audit_status']} (audit unchanged). {summary_line(r)}.", ""]
    if r["outside_path_findings"]:
        L += ["| severity | where | audit resolved | status | ran from | re-resolved | token |", "|---|---|---|---|---|---|---|"]
        for o in r["outside_path_findings"]:
            L.append(f"| {o['severity']} | `{o['where']}` | `{o['audit_path']}` | **{o['status']}** | "
                     f"{', '.join(o['cwds']) or '–'} | {', '.join(f'`{p}`' for p in o['paths']) or '–'} | "
                     f"{', '.join(f'`{x}`' for x in o['raw']) or '–'} |")
    if r["commands_with_cd"]:
        L += ["", "Bash calls that change directory (replayed cwd vs the cwd Claude Code recorded; `cd` arguments only):", ""]
        for c in r["commands_with_cd"]:
            L.append(f"- `{c['loc']}`: before {c['before']} (recorded {c['before_recorded']}); "
                     f"cd {' ; cd '.join(f'`{d}`' for d in c['cds']) or '–'}; after {c['after']} (recorded "
                     f"{c['after_recorded']}){' (call failed: cwd not kept)' if c['failed'] else ''}"
                     f"{f'; {n} path(s) joined to the cd by && resolved (Amendment 7)' if (n := c.get('and_chain_paths')) else ''}"
                     f"{f'; {k} cd(s) to the directory the shell was already in (Amendment 10)' if (k := c.get('noop_cds')) else ''}"
                     f"{'' if c['agree'] else ' **MISMATCH**'}{'' if c['certain'] else ' **uncertain**'}")
    L += ["", "Paths outside the run dir that the audit resolved as inside: "
          + ("none." if not r["missed_by_audit"] else "")]
    for m in r["missed_by_audit"]:
        L.append(f"- `{m['where']}`: `{m['raw']}` from {m['cwd']} -> `{m['path']}` ({m['class']}, {m['basis']})")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", default=str(REPO_ROOT))
    ap.add_argument("--json", help="default: runs/wsp/<id>/path_review.json")
    ap.add_argument("--summary", action="store_true", help="print only the summary line")
    a = ap.parse_args(argv)
    pr = PathReview(a.run, Path(a.root))
    r = pr.report()
    write_json_atomic(Path(a.json) if a.json else pr.p.dir / "path_review.json", r)
    print(summary_line(r) if a.summary else render_md(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
