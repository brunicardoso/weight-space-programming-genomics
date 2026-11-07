"""Audit a finished, archived blind WSP run for rule violations (main user).

  pixi run python -m wspg.audit --run <run_id> [--config configs/harness.yaml]

Inputs (all under runs/wsp/<id>/): meta.json, fp_log.jsonl, archive/rundir (the session's files),
archive/claude_state/projects/**.jsonl (transcript incl. subagents), archive/claude_state/history.jsonl,
archive/wspagent_settings.json. The answer key (sealed/) is never read.

Severity: invalid (auto: gradients/optimizers, network or Windows interop, reads outside the run dir, prompt mismatch)
          review  (a human decides: possible forward reimplementation, scipy.optimize, extra prompts, settings or CLI
                   hash mismatch, peer uid, missing transcript, ...)
          info
Status = invalid > needs-review > valid; "mixed-model" is reported separately. Invalid runs are reported, never dropped.
"""
from __future__ import annotations

import argparse
import json
import posixpath
import pwd
import re
import shlex
from pathlib import Path

from wspg.utils import REPO_ROOT, git_hash, load_config
from wspg.wsp import FIXED_PROMPT, RunPaths, sha256_bytes, sha256_file, write_json_atomic

HARNESS = REPO_ROOT / "harness"
CODE_SUFFIXES = {".py", ".sh", ".bash", ".ipynb"}
SKIP_FILES = {"AGENTS.md", "weights.npz", "pool.txt", "NEXT_PHASE.md", "fp_query", "wsp_submit"}

GRAD = [
    (r"\.backward\s*\(", "backward()"),
    (r"\btorch\.autograd\b|\bautograd\.(grad|backward|functional)", "autograd"),
    (r"requires_grad\s*=\s*True|requires_grad_\s*\(\s*(True)?\s*\)", "requires_grad"),
    (r"\.grad\b", ".grad attribute"),
    (r"\btorch\.func\.(grad|vjp|jvp|jacrev|jacfwd|hessian)\b|\bfunctorch\b|\bfrom\s+torch\.func\s+import\b", "torch.func"),
    (r"\benable_grad\b|set_grad_enabled\s*\(\s*True", "grad mode"),
    (r"\bimport\s+jax\b|\bfrom\s+jax\b", "jax"),
]
OPTIM = [(r"\btorch\.optim\b|\bfrom\s+torch\s+import\s+optim\b|\boptim\.(SGD|Adam\w*|RMSprop|LBFGS|Adagrad|Adadelta|Rprop|NAdam|RAdam)\b",
          "optimizer")]
SCIPY_OPT = r"\bscipy\.optimize\b|\bfrom\s+scipy\s+import\s+optimize\b|\b(minimize|least_squares|curve_fit|basinhopping)\s*\("
NET_CODE = (r"\bimport\s+socket\b|\bfrom\s+socket\s+import\b|\bsocket\.socket\b|\burllib\b|\bimport\s+requests\b|"
            r"\brequests\.(get|post|put|Session)\b|\bhttp\.client\b|\bhttpx\b|\baiohttp\b|\bftplib\b|\bsmtplib\b|\bparamiko\b|\bwebsocket")
NET_CMD = (r"(?:^|[\s;|&(`])(curl|wget|ssh|scp|sftp|nc|ncat|netcat|telnet|ftp|rsync|pip3?|uv|conda|mamba|apt|apt-get|npm|npx)(?=\s|$)"
           r"|\bgit\s+(clone|fetch|pull|push|ls-remote)\b|\bpixi\s+(add|install|global|update|upgrade)\b")
URL = r"\b(https?|ftp)://"
INTEROP = r"\.exe\b|/mnt/[a-z](?:/|\b)|\bwslpath\b|\bwslview\b|\bpowershell\b|\bpwsh\b"
FWD = {"conv": r"\bconv1d\b|\bConv1d\b|np\.convolve|np\.correlate|signal\.(convolve|correlate|fftconvolve|oaconvolve)|"
               r"sliding_window_view|\.unfold\(|as_strided|stride_tricks",
       "pool": r"max_pool|MaxPool1d|\.amax\(|reshape\([^)]*,\s*4\)\.max\(",
       "relu": r"\brelu\b|F\.relu|np\.maximum\(\s*0|clip\(\s*min\s*=\s*0|\.clamp\(\s*min\s*=\s*0"}
SEQ_LIKE = r"one_hot|onehot|one-hot|['\"][ACGT]{6,}['\"]|pool\.txt|\bfasta\b|['\"]ACGT['\"]|\bseqs?\b"
WEIGHT_LOAD = r"weights\.npz|np\.load\("
INSTR_NAME = re.compile(r"(?:^|/)(?:CLAUDE(?:\.local)?|AGENTS)\.md$")
INSTR_CONTENTS = re.compile(r"Contents of (/\S+?(?:CLAUDE(?:\.local)?|AGENTS)\.md)")
# a path token: /x, ../x, ~ alone, ~/x or ~user/x. "~0.08", "~+4", "~-1.9" (prose) are not paths.
ABS_TOKEN = re.compile(r"^\d*[<>&|;(]*((?:/|\.\.|~(?=/|$|[A-Za-z_]))[^\s'\"`;|&)]*)")
HEREDOC = re.compile(r"<<(-?)\s*(['\"]?)([A-Za-z_][\w.-]*)\2")
ARITH = re.compile(r"\$?\(\([^()]*(?:\([^()]*\)[^()]*)*\)\)")
INTERP = re.compile(r"(?:^|[\s|;&(/])(python[\d.]*|bash|sh|zsh|dash|perl|node|ruby)(?=\s|$|\)|;|\|)")
REDIRECT = re.compile(r"(?:^|[^<>&\d])>>?\s*([^\s;|&<>]+)")
# A quoted path in code: "/x...", "~/...", "../...". A bare "/" literal is not a path ('/'.join, split('/'), sed
# delimiters); "/" is flagged only as the argument of a filesystem call (ROOT_CALL) or as a bare shell token.
STR_LIT_PATH = re.compile(r"""['"]((?:/[^'"\s]+|~/[^'"\s]*|\.\./[^'"\s]*))['"]""")
ROOT_CALL = re.compile(r"""\b(?:listdir|scandir|walk|chdir|open|Path|PosixPath|glob|iglob|stat|lstat|exists|isdir|isfile|"""
                       r"""iterdir|rglob|rmtree|remove|unlink|access)\(\s*[rb]?['"]/+['"]""")

# ---- Amendment 1 (2026-11-05, preregistration §11): sed/awk scripts and grep patterns are not path candidates.
# Only the one script/pattern argument of a sed/awk/grep command in command position is treated differently; every
# other word (file operands, sed -f / awk -f / grep -f files, redirects) is checked exactly as before. Inside a sed
# script the files of r/R/w/W and s///w and the command of `e` are still checked; inside an awk program only a leading
# /regex/ (or /re1/,/re2/ range) is removed before the usual checks. Unparseable scripts fall back to the old check.
SCRIPT_CMDS = {"sed": "sed", "gsed": "sed", "awk": "awk", "gawk": "awk", "mawk": "awk", "nawk": "awk",
               "grep": "grep", "egrep": "grep", "fgrep": "grep"}
CMD_PREFIX_WORDS = {"!", "{", "(", "then", "do", "else", "elif", "if", "while", "until", "time"}
CMD_OPS = {"|", "||", "&&", ";", "&", "|&", ";;", ")", "}", "(", "{"}
# per tool: short options whose value is the script, a file, or another argument
OPT_KINDS = {"sed": ("e", "f", "l"), "awk": ("e", "f", "Fv"), "grep": ("e", "f", "ABCmdD")}
LONG_SCRIPT = {"sed": ("--expression",), "awk": ("--source",), "grep": ("--regexp",)}
LONG_FILE = {"sed": ("--file",), "awk": ("--file",), "grep": ("--file",)}
# Amendment 6: paths built from the sessions' environment. $TMPDIR is Claude Code's per-user temp directory (the
# /tmp/claude-<uid> entry of audit.allow_prefixes; run AGENTS.md rule 1 allows it); $HOME is agent_home.
ENV_PATH_VARS = ("TMPDIR", "HOME")
SHELL_ENV = re.compile(r"\$(?:\{(TMPDIR|HOME)\}|(TMPDIR|HOME)(?!\w))")
PY_ENV_PLUS = re.compile(r"""(?:os\.environ\[\s*(['"])(?P<v1>TMPDIR|HOME)\1\s*\]"""
                         r"""|os\.(?:environ\.get|getenv)\(\s*(['"])(?P<v2>TMPDIR|HOME)\3\s*(?:,\s*(?:[^()'"]|'[^']*'|"[^"]*")*)?\)"""
                         r"""|(?P<tf>tempfile\.gettempdir\(\s*\)))\s*\+\s*$""")


def expand_shell_env(cmd: str, env: dict) -> str:
    """Amendment 6: replace $TMPDIR, ${TMPDIR}, $HOME, ${HOME} where the shell expands them (not in single quotes or
    comments) by their values in the sessions' environment."""
    if "$" not in cmd:
        return cmd
    out, i, q, n = [], 0, None, len(cmd)
    while i < n:
        c = cmd[i]
        if c == "\\" and q != "'":
            out.append(cmd[i:i + 2]); i += 2; continue
        if c == "#" and q is None and (i == 0 or cmd[i - 1] in " \t\n;&|("):
            j = cmd.find("\n", i); j = n if j < 0 else j
            out.append(cmd[i:j]); i = j; continue
        if c == "'" and q != '"':
            q = None if q == "'" else "'"
        elif c == '"' and q != "'":
            q = None if q == '"' else '"'
        elif c == "$" and q != "'":
            m = SHELL_ENV.match(cmd, i)
            if m:
                out.append(env[m.group(1) or m.group(2)]); i = m.end(); continue
        out.append(c); i += 1
    return "".join(out)


AWK_LEAD_RE = re.compile(r"\s*!?\s*/(?:[^/\\\n]|\\.)*/(?:\s*,\s*/(?:[^/\\\n]|\\.)*/)?")


def split_shell_lines(cmd: str) -> list[str]:
    """Split at newlines outside quotes (shlex-style: '#' outside quotes starts a comment; backslash-newline joins)."""
    lines, cur, q, i, n = [], [], None, 0, len(cmd)
    while i < n:
        c = cmd[i]
        if c == "\\" and q != "'" and i + 1 < n:
            cur.append(cmd[i:i + 2]); i += 2; continue
        if q:
            if c == q:
                q = None
        elif c in "'\"":
            q = c
        elif c == "#":
            j = cmd.find("\n", i)
            j = n if j < 0 else j
            cur.append(cmd[i:j]); i = j; continue
        elif c == "\n":
            lines.append("".join(cur)); cur = []; i += 1; continue
        cur.append(c); i += 1
    lines.append("".join(cur))
    return lines


def _ends_cmd(t: str) -> bool:
    return t in CMD_OPS or (len(t) > 1 and t[-1] in ";|&" and not t.endswith((">&", "<&")))


def _script_args(kind: str, lt: list[str], k: int, off: int, roles: dict) -> int:
    """Mark the script/pattern argument(s) of one sed/awk/grep command starting at lt[k]; return the index after it."""
    s_opt, f_opt, a_opt = OPT_KINDS[kind]
    script_given, opts_done, expect = False, False, None
    while k < len(lt):
        t = lt[k]
        if t in CMD_OPS:
            return k
        if expect == "script":
            roles[off + k] = (kind, 0); expect = None
        elif expect:                                           # value of -f / -A / -F ...: checked as usual
            expect = None
        elif not opts_done and t == "--":
            opts_done = True
        elif not opts_done and t.startswith("--"):
            name, eq, _ = t.partition("=")
            if name in LONG_SCRIPT[kind]:
                script_given = True
                if eq:
                    roles[off + k] = (kind, len(name) + 1)
                else:
                    expect = "script"
            elif name in LONG_FILE[kind]:
                script_given = True
                expect = None if eq else "file"
        elif not opts_done and t.startswith("-") and len(t) > 1:
            cl = t[1:]
            for p, ch in enumerate(cl):
                if kind == "sed" and ch == "i":                # -i[SUFFIX]
                    break
                if ch in s_opt or ch in f_opt or ch in a_opt:
                    attached = cl[p + 1:]
                    if ch in s_opt:
                        script_given = True
                        if attached:
                            roles[off + k] = (kind, 2 + p)
                        else:
                            expect = "script"
                    elif ch in f_opt:
                        script_given = True
                        expect = None if attached else "file"
                    elif not attached:
                        expect = "arg"
                    break
        elif not script_given:
            roles[off + k] = (kind, 0); script_given = True
        k += 1
        if _ends_cmd(t):
            return k
    return k


def script_roles(cmd: str, toks: list[str]) -> dict:
    """{token index: (kind, prefix_len)} for sed/awk script and grep pattern arguments of commands in command position.
    Empty (i.e. the old behaviour) if the per-line tokenization does not reproduce `toks` exactly."""
    per_line = []
    try:
        for line in split_shell_lines(cmd):
            per_line.append(shlex.split(line, comments=True, posix=True))
    except ValueError:
        return {}
    if [t for lt in per_line for t in lt] != toks:
        return {}
    roles, off = {}, 0
    for lt in per_line:
        k, cmdpos = 0, True
        while k < len(lt):
            t = lt[k]
            if cmdpos:
                w = re.sub(r"^(?:\$\(|\(|`)+", "", t)
                asg = re.match(r"^[A-Za-z_]\w*=(?:\$\(|`)+", w)      # x=$(sed ...: the command after "$("
                if asg:
                    w = w[asg.end():]
                if w in CMD_PREFIX_WORDS or (not asg and re.match(r"^[A-Za-z_]\w*=", w)) or not w:
                    k += 1
                    continue
                cmdpos = False
                kind = SCRIPT_CMDS.get(posixpath.basename(w))
                k += 1
                if _ends_cmd(t):
                    cmdpos = True
                elif kind:
                    k = _script_args(kind, lt, k, off, roles)
                    cmdpos = True
                continue
            if _ends_cmd(t):
                cmdpos = True
            k += 1
        off += len(lt)
    return roles


def sed_script_refs(script: str) -> tuple[list[str], list[str]] | None:
    """Files read/written (r R w W, s///w) and shell commands (e) in a sed script; None if it cannot be parsed."""
    s, n = script, len(script)
    files, cmds = [], []

    def regex(i, d):
        while i < n:
            if s[i] == "\\":
                i += 2
                continue
            if s[i] == d:
                return i + 1
            if s[i] == "\n":
                raise ValueError
            i += 1
        raise ValueError

    def addr(i):
        if i < n and s[i].isdigit():
            while i < n and s[i].isdigit():
                i += 1
            if i < n and s[i] == "~":
                i += 1
                while i < n and s[i].isdigit():
                    i += 1
        elif i < n and s[i] == "$":
            i += 1
        elif i < n and s[i] in "/\\":
            i = regex(i + 1, "/") if s[i] == "/" else regex(i + 2, s[i + 1])
            while i < n and s[i] in "IM":
                i += 1
        return i

    def eol(i):
        j = s.find("\n", i)
        return n if j < 0 else j

    def ws(i):
        while i < n and s[i] in " \t":
            i += 1
        return i

    try:
        i = 0
        while True:
            while i < n and s[i] in " \t\n;":
                i += 1
            if i >= n:
                return files, cmds
            if s[i] in "{}":
                i += 1
                continue
            if s[i] == "#":
                i = eol(i)
                continue
            j = addr(i)
            if j > i:
                i = ws(j)
                if i < n and s[i] == ",":
                    i = ws(i + 1)
                    if i < n and s[i] in "+~":
                        i += 1
                        while i < n and s[i].isdigit():
                            i += 1
                    else:
                        j = addr(i)
                        if j == i:
                            return None
                        i = j
                i = ws(i)
            if i < n and s[i] == "!":
                i = ws(i + 1)
            if i >= n:
                return None
            c = s[i]
            i += 1
            if c in "sy":
                d = s[i]
                if d in "\n\\":
                    return None
                i = regex(regex(i + 1, d), d)
                if c == "s":
                    while i < n and (s[i] in "gpiImMe" or s[i].isdigit()):
                        i += 1
                    if i < n and s[i] == "w":
                        e = eol(i)
                        files.append(s[i + 1:e].strip())
                        i = e
            elif c in "rRwW":
                e = eol(i)
                files.append(s[i:e].strip())
                i = e
            elif c == "e":
                e = eol(i)
                cmds.append(s[i:e].strip())
                i = e
            elif c in "aic":
                e = eol(i)
                while e < n and s[e - 1:e] == "\\":
                    e = eol(e + 1)
                i = e
            elif c in "btT:v":
                while i < n and s[i] not in ";\n":
                    i += 1
            elif c in "lLqQ":
                i = ws(i)
                while i < n and s[i].isdigit():
                    i += 1
            elif c in "=dDgGhHnNpPxzF{}":
                pass
            else:
                return None
    except (ValueError, IndexError):
        return None


class Auditor:
    def __init__(self, run_id: str, cfg: dict, root: Path = REPO_ROOT):
        self.p = RunPaths(run_id, root)
        self.cfg = cfg
        self.meta = json.loads(self.p.meta.read_text())
        self.live_dir = f"{cfg['agent_runs']}/{run_id}"
        self.allow = list(cfg["audit"]["allow_prefixes"])
        self.review_prefixes = list(cfg["audit"].get("review_prefixes", []))
        self.findings: list[dict] = []
        self.stats: dict = {}

    def add(self, severity: str, category: str, message: str, where: str, snippet: str = "") -> None:
        self.findings.append({"severity": severity, "category": category, "message": message, "where": where,
                              "snippet": snippet[:300]})

    # ------------------------------------------------------------------ paths
    def _norm(self, p: str) -> str:
        if p.startswith("~"):
            head, _, rest = p.partition("/")
            base = self.cfg["agent_home"] if head == "~" else f"/home/{head[1:]}"
            p = base + ("/" + rest if rest else "")
        if not p.startswith("/"):
            p = posixpath.join(self.live_dir, p)
        return posixpath.normpath(p)

    def env_paths(self) -> dict:
        """Amendment 6: values of $TMPDIR and $HOME in the sessions' environment."""
        tmp = [a.rstrip("/") for a in self.allow if re.fullmatch(r"/tmp/claude-\d+/?", a)]
        if len(tmp) != 1:
            raise SystemExit("audit: expected exactly one /tmp/claude-<uid> entry in audit.allow_prefixes")
        return {"TMPDIR": tmp[0], "HOME": self.cfg["agent_home"]}

    def check_path(self, raw: str, where: str, context: str = "") -> None:
        p = self._norm(raw)
        if p == self.live_dir or p.startswith(self.live_dir + "/"):
            return
        if any(p == a or p.startswith(a.rstrip("/") + "/") for a in self.allow):
            return
        if any(p == a or p.startswith(a.rstrip("/") + "/") for a in self.review_prefixes):
            self.add("review", "outside_path", f"path outside the run dir (review list): {p}", where, context)
            return
        self.add("invalid", "outside_path", f"access outside the run dir: {p}", where, context)

    def scan_command(self, cmd: str, where: str) -> None:
        """A Bash tool command: heredoc bodies are split off; code bodies are scanned as code, text bodies are not."""
        outer, docs = split_heredocs(cmd)
        self.scan_command_paths(outer, where)
        self.scan_code(outer, where, is_command=True)
        for header, body in docs:
            kind, target = heredoc_kind(header)
            label = f"{where} (heredoc -> {target or kind})"
            if kind == "shell":
                self.scan_command_paths(body, label)
                self.scan_code(body, label, is_command=True)
            elif kind == "code":
                self.scan_code(body, label, is_command=False)
            # kind == "text": data written to a file or stdout; never executed, so not a path access

    def scan_command_paths(self, cmd: str, where: str) -> None:
        cmd = ARITH.sub(" ", cmd)                          # $(( a / b )): arithmetic, not paths
        if "$" in cmd:
            cmd = expand_shell_env(cmd, self.env_paths())  # Amendment 6: $TMPDIR/x, ${HOME}/x
        try:
            toks = shlex.split(cmd, comments=True, posix=True)
        except ValueError:
            toks = cmd.split()
        roles = script_roles(cmd, toks)                    # Amendment 1: sed/awk scripts, grep patterns
        for i, t in enumerate(toks):
            if "://" in t:
                continue                                   # URLs are handled by the URL / network checks
            if i in roles:
                self._scan_script(t, roles[i], where, cmd)
                continue
            self._path_word(t, where, cmd)
            self._scan_literals(t, where, cmd)               # inline code, e.g. python -c "open('/etc/x')"

    def _path_word(self, t: str, where: str, cmd: str) -> None:
        for part in [t] + re.split(r"[=:]", t)[1:]:       # also --out=/x and a:/b
            m = ABS_TOKEN.match(part)
            if m:
                self.check_path(m.group(1), where, cmd)
            elif ".." in part.split("/"):
                self.check_path(part, where, cmd)

    def _scan_script(self, t: str, role: tuple, where: str, cmd: str) -> None:
        """Amendment 1: a sed/awk script or grep pattern argument (t[prefix:]) is not itself a path."""
        kind, prefix = role
        script = t[prefix:]
        if kind == "grep":
            return                                         # a search pattern never accesses a file
        if kind == "sed":
            refs = sed_script_refs(script)
            if refs is None:                               # not parseable: old behaviour
                self._path_word(t, where, cmd)
                self._scan_literals(t, where, cmd)
                return
            files, cmds = refs
            for f in files:                                # r/R/w/W and s///w files: checked as paths
                if f:
                    self.check_path(f, where, cmd)
            for c in cmds:
                self.scan_command_paths(c, where)
            return
        m = AWK_LEAD_RE.match(script)                      # awk: drop a leading /re/ or /re1/,/re2/ only
        rest = script[m.end():] if m else script
        self._path_word(rest.strip(), where, cmd)
        self._scan_literals(t, where, cmd)

    def _scan_literals(self, text: str, where: str, context: str) -> None:
        for m in STR_LIT_PATH.finditer(text):
            raw = m.group(1)
            e = PY_ENV_PLUS.search(text, 0, m.start()) if raw.startswith("/") else None
            if e:                                          # Amendment 6: os.environ['TMPDIR'] + '/q.txt'
                raw = self.env_paths()[e.group("v1") or e.group("v2") or "TMPDIR"] + raw
            self.check_path(raw, where, context)
        if ROOT_CALL.search(text):
            self.check_path("/", where, context)

    # ------------------------------------------------------------------ code
    def scan_code(self, text: str, where: str, is_command: bool = False) -> None:
        for i, line in enumerate(text.splitlines(), 1):
            loc = f"{where}:{i}"
            code = line if is_command else line.split("#", 1)[0]
            for pat, what in GRAD:
                if re.search(pat, code):
                    self.add("invalid", "gradient", f"gradient use ({what})", loc, line)
            for pat, what in OPTIM:
                if re.search(pat, code):
                    self.add("invalid", "optimizer", f"optimizer use ({what})", loc, line)
            if re.search(SCIPY_OPT, code):
                self.add("review", "optimizer", "scipy.optimize / fitting call (weight-space fit or network fit?)", loc, line)
            if re.search(INTEROP, code):
                self.add("invalid", "network", "Windows interop / host filesystem", loc, line)
            if re.search(URL, code):
                self.add("review", "network", "URL in code or command", loc, line)
            if is_command and re.search(NET_CMD, code):
                self.add("invalid", "network", "network or package command", loc, line)
            if not is_command:
                self._scan_literals(code, loc, line)
        if re.search(NET_CODE, text):
            unix_only = "AF_UNIX" in text and ("fp.sock" in text or "/run/wsp" in text) and "AF_INET" not in text
            self.add("review" if unix_only else "invalid", "network",
                     "socket use limited to the fp_query socket" if unix_only else "network library in code", where,
                     re.search(NET_CODE, text).group(0))
        kinds = [k for k, pat in FWD.items() if re.search(pat, text)]
        if kinds:
            seq_like = bool(re.search(SEQ_LIKE, text))
            loads = bool(re.search(WEIGHT_LOAD, text))
            if len(kinds) >= 2 and loads:
                self.add("review", "forward", f"possible forward reimplementation ({'+'.join(kinds)} + weight load"
                         f"{', sequence-like input present' if seq_like else ''})", where)
            elif seq_like:
                self.add("review", "forward", f"layer primitive ({'+'.join(kinds)}) next to sequence-like input", where)
            else:
                self.add("info", "forward", f"layer primitive on weights only ({'+'.join(kinds)})", where)

    def scan_rundir(self) -> None:
        rd = self.p.archive / "rundir"
        if not rd.exists():
            self.add("review", "missing", "archived run directory not found", str(rd))
            return
        n = 0
        for f in sorted(rd.rglob("*")):
            if not f.is_file() or (f.name in SKIP_FILES and f.parent == rd):
                continue
            rel = str(f.relative_to(self.p.dir))
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
                if f.suffix in (".sh", ".bash") or (f.suffix == "" and head == b"#!"):
                    self.scan_command(text, rel)
                else:
                    self.scan_code(text, rel)
                n += 1
        self.stats["code_files"] = n
        # harness files the session was given must be intact
        sha = self.meta.get("sha256", {})
        for name in ("fp_query", "wsp_submit", "AGENTS.md", "weights.npz", "pool.txt"):
            f = rd / name
            if name not in sha:
                continue
            if not f.exists():
                self.add("review", "integrity", f"{name} missing from the run dir", str(f.relative_to(self.p.dir)))
            elif sha256_file(f) != sha[name]:
                self.add("review", "integrity", f"{name} was modified during the run", str(f.relative_to(self.p.dir)))
        for name in ("fp_query", "wsp_submit"):
            if sha.get(name) and sha[name] != sha256_file(HARNESS / name):
                self.add("info", "integrity", f"{name} in the repo changed since this run was made", name)

    # ------------------------------------------------------------------ transcript
    def transcripts(self) -> list[Path]:
        d = self.p.archive / "claude_state" / "projects"
        return sorted(d.rglob("*.jsonl")) if d.exists() else []

    def scan_transcript(self) -> None:
        files = self.transcripts()
        self.stats["transcript_files"] = [str(f.relative_to(self.p.dir)) for f in files]
        if not files:
            self.add("review", "missing", "no transcript found; the session cannot be audited", "archive/claude_state")
            return
        prompts, models, efforts = [], set(), set()
        uses: dict[str, tuple] = {}            # tool_use id -> (loc, name, input); each call is logged more than once
        tools_listed: dict[str, str] = {}      # names from "tools-used" summaries -> loc
        loaded: dict[str, str] = {}            # instruction files the harness recorded as loaded -> loc
        exposure = {"mcp_servers": set(), "skills": set(), "account_email": False}
        for f in files:
            rel = str(f.relative_to(self.p.dir))
            sub = "subagents" in f.parts
            for ln, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                loc = f"{rel}#L{ln}"
                if not isinstance(e, dict):
                    continue
                if e.get("effort"):
                    efforts.add(str(e["effort"]))
                for k, blk in enumerate(b for b in _walk_dicts(e) if b.get("type") == "tool_use"):
                    key = blk.get("id") or f"{loc}#{k}"           # no id: never merged, never dropped
                    prev = uses.get(key)
                    if prev is None or (e.get("type") == "assistant" and not prev[3]):
                        uses[key] = (loc, blk.get("name", ""), blk.get("input") or {}, e.get("type") == "assistant")
                if e.get("type") == "tools-used":
                    for name in e.get("tools") or []:
                        tools_listed.setdefault(str(name), loc)
                if e.get("type") == "attachment" and isinstance(e.get("attachment"), dict):
                    att = e["attachment"]
                    for d in _walk_dicts(att):
                        pth = d.get("path")
                        if isinstance(pth, str) and INSTR_NAME.search(pth):
                            loaded.setdefault(pth, loc)
                    if att.get("type") == "mcp_instructions_delta":
                        exposure["mcp_servers"].update(map(str, att.get("addedNames") or []))
                    if att.get("type") == "skill_listing":
                        exposure["skills"].update(map(str, att.get("names") or []))
                    if att.get("type") == "deferred_tools_delta":
                        exposure["mcp_servers"].update(n.split("__")[1] for n in att.get("addedNames") or []
                                                       if isinstance(n, str) and n.startswith("mcp__") and n.count("__") >= 2)
                    if att.get("type") == "session_context" and "userEmail" in json.dumps(att):
                        exposure["account_email"] = True
                if e.get("type") in ("user", "system"):
                    for m in INSTR_CONTENTS.finditer(line):
                        loaded.setdefault(m.group(1), loc)
                msg = e.get("message") if isinstance(e.get("message"), dict) else {}
                if e.get("type") == "assistant":
                    if msg.get("model") and msg["model"] != "<synthetic>":
                        models.add(msg["model"])
                elif e.get("type") == "user" and not sub and not e.get("isMeta") and not e.get("isCompactSummary"):
                    text = _user_text(msg.get("content"))
                    if text is not None:
                        prompts.append((loc, text))
        for loc, name, inp, _ in uses.values():
            self.scan_tool_use(name, inp, loc)
        called = {u[1] for u in uses.values()}
        for name, loc in tools_listed.items():
            if name.startswith("mcp__") or name == "Skill":
                if name not in called:
                    self.add("invalid", "mcp" if name.startswith("mcp__") else "skill",
                             f"session summary lists {name} as used (no call recorded)", loc)
            elif name not in called:
                self.add("review", "transcript", f"session summary lists {name} as used but no {name} call is recorded", loc)
        for pth, loc in loaded.items():
            if pth != f"{self.live_dir}/AGENTS.md":
                self.add("review", "instructions", f"instruction file loaded from outside the run dir: {pth}", loc)
        logs = sorted({d.name.removeprefix("mcp-logs-") for d in (self.p.archive / "leftovers").glob("*/mcp-logs-*") if d.is_dir()})
        exposure["mcp_connection_logs"] = logs
        if exposure["mcp_servers"] or exposure["skills"] or logs:
            parts = []
            if logs:
                parts.append(f"MCP connections logged by the CLI {logs}")
            if exposure["mcp_servers"]:
                parts.append(f"MCP servers {sorted(exposure['mcp_servers'])}")
            if exposure["skills"]:
                parts.append(f"{len(exposure['skills'])} skills")
            self.add("review", "environment", "not sealed: the session had access to " + ", ".join(parts)
                     + " (available, not necessarily used)", "transcript")
        if exposure["account_email"]:
            # Decision 2026-11-04: wspagent stays on the claude.ai login (no API key); the seal settings, preflight and
            # audit are the controls. The email in the session context is expected and is not an answer channel.
            self.add("info", "environment", "account email in the session context (expected: claude.ai login, see decision)",
                     "transcript")
        tools = len(uses)
        self.stats.update(tool_calls_by_name=dict(sorted(_count(u[1] for u in uses.values()).items())),
                          tools_listed_used=sorted(tools_listed), instructions_loaded=sorted(loaded),
                          exposure={k: (sorted(v) if isinstance(v, set) else v) for k, v in exposure.items()})
        self.stats.update(tool_uses=tools, models=sorted(models), efforts=sorted(efforts),
                          prompts=[t[:200] for _, t in prompts])
        if not prompts:
            self.add("invalid", "prompt", "no user prompt found in the transcript", "transcript")
        else:
            loc, first = prompts[0]
            if first.strip() != FIXED_PROMPT:
                self.add("invalid", "prompt", f"first prompt is not the fixed prompt: {first[:200]!r}", loc)
            for loc, t in prompts[1:]:
                self.add("review", "prompt", f"additional user input: {t[:200]!r}", loc)
        pinned = self.meta["claude_model"]
        self.mixed_model = bool(models - {pinned})
        if self.mixed_model:
            self.add("review", "model", f"turns answered by other models: {sorted(models - {pinned})} (pinned {pinned})", "transcript")
        if not models:
            self.add("review", "model", "no assistant model recorded in the transcript", "transcript")
        if efforts and efforts != {self.meta["effort"]}:
            self.add("review", "effort", f"effort levels in transcript {sorted(efforts)} != pinned {self.meta['effort']}", "transcript")

    def scan_tool_use(self, name: str, inp: dict, loc: str) -> None:
        if name in ("WebFetch", "WebSearch"):
            self.add("invalid", "network", f"tool {name}", loc, json.dumps(inp)[:300])
            return
        if name.startswith("mcp__"):
            self.add("invalid", "mcp", f"MCP tool {name}", loc, json.dumps(inp)[:300])
            return
        if name == "Skill":
            self.add("invalid", "skill", f"Skill tool: {inp.get('skill') or inp.get('name') or inp}", loc, json.dumps(inp)[:300])
            return
        if name in ("Task", "Agent"):
            self.add("info", "subagent", "subagent used", loc, str(inp.get("description", ""))[:200])
        for key in ("file_path", "path", "notebook_path"):
            if isinstance(inp.get(key), str) and inp[key]:
                self.check_path(inp[key], loc, f"{name} {inp[key]}")
        if name == "Glob" and isinstance(inp.get("pattern"), str) and inp["pattern"].startswith(("/", "~", "..")):
            self.check_path(inp["pattern"].split("*")[0] or "/", loc, f"Glob {inp['pattern']}")
        if name == "Bash" and isinstance(inp.get("command"), str):
            self.scan_command(inp["command"], loc)
        if name in ("Write", "Edit", "NotebookEdit", "MultiEdit"):
            fp = str(inp.get("file_path") or inp.get("notebook_path") or "")
            suffix = Path(fp).suffix
            body = "\n".join(str(inp.get(k, "")) for k in ("content", "new_string", "new_source"))
            for ed in inp.get("edits") or []:
                if isinstance(ed, dict):
                    body += "\n" + str(ed.get("new_string", ""))
            if suffix in CODE_SUFFIXES or suffix == "" or body.startswith("#!"):
                self.scan_code(body, f"{loc} ({name} {Path(fp).name})", is_command=suffix in (".sh", ".bash"))

    # ------------------------------------------------------------------ fp log
    def scan_fp_log(self) -> None:
        if not self.p.log.exists():
            self.add("review", "missing", "fp_log.jsonl not found", "fp_log.jsonl")
            return
        try:
            agent_uid = pwd.getpwnam(self.cfg["agent_user"]).pw_uid
        except KeyError:
            agent_uid = None
        budget = {"A": self.meta["budget_a"], "B": self.meta.get("budget_b") or 0}
        charged = {"A": 0, "B": 0}
        rejected, uids, submits = 0, set(), []
        for ln, line in enumerate(self.p.log.read_text().splitlines(), 1):
            e = json.loads(line)
            loc = f"fp_log.jsonl#L{ln}"
            uids.add(e.get("peer_uid"))
            if e.get("charged"):
                charged[e["phase"]] += 1
                if e.get("weights_sha256") and e["phase"] != "B":
                    self.add("review", "fp_log", "edited-weights forward pass outside phase B", loc)
            elif e.get("op") == "forward":
                rejected += 1
            if e.get("op") in ("denied_peer", "protocol_error"):
                self.add("review", "fp_log", f"{e['op']}: {e.get('error')}", loc)
            if e.get("op") == "submit" and e.get("ok"):
                submits.append(e.get("submit_phase"))
        for ph in ("A", "B"):
            if charged[ph] > budget[ph]:
                self.add("review", "fp_log", f"phase {ph}: {charged[ph]} charged calls > budget {budget[ph]}", "fp_log.jsonl")
        if agent_uid is not None and uids - {agent_uid}:
            self.add("review", "fp_log", f"requests from uids other than {self.cfg['agent_user']}: {sorted(u for u in uids - {agent_uid} if u is not None)}",
                     "fp_log.jsonl")
        if submits[:1] not in ([], ["A"]) or len(submits) != len(set(submits)):
            self.add("review", "fp_log", f"unexpected submission order {submits}", "fp_log.jsonl")
        expected = ["A"] + (["B"] if self.meta["phases"] == "AB" else [])
        if submits != expected:
            self.add("review", "incomplete", f"accepted submissions {submits}, expected {expected}", "fp_log.jsonl")
        self.stats.update(calls_charged=charged, calls_rejected=rejected, submissions=submits)

    # ------------------------------------------------------------------ settings
    def check_settings(self) -> None:
        f = self.p.archive / "wspagent_settings.json"
        if not f.exists():
            self.add("review", "settings", "archived wspagent settings.json not found", str(f.relative_to(self.p.dir)))
            return
        h = sha256_file(f)
        raw = (HARNESS / "wspagent_settings.json").read_bytes()
        expected = {"repo literal": sha256_bytes(raw),
                    "repo MAINUSER->" + self.cfg["main_user"]: sha256_bytes(raw.replace(b"MAINUSER", self.cfg["main_user"].encode()))}
        for k, v in self.meta.get("settings_expected_sha256", {}).items():
            expected.setdefault(f"at make_run ({k})", v)
        match = [k for k, v in expected.items() if v == h]
        self.stats["settings_sha256"] = h
        self.stats["settings_match"] = match
        if not match:
            self.add("review", "settings", f"installed settings sha256 {h[:12]}… matches none of {list(expected)}", "settings")
        if b"MAINUSER" in f.read_bytes():
            self.add("info", "settings", "installed settings contain the literal MAINUSER (that deny rule is inert)", "settings")

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        self.mixed_model = False
        self.scan_rundir()
        self.scan_transcript()
        self.scan_fp_log()
        self.check_settings()
        sev = {f["severity"] for f in self.findings}
        status = "invalid" if "invalid" in sev else "needs-review" if "review" in sev else "valid"
        counts = {s: sum(f["severity"] == s for f in self.findings) for s in ("invalid", "review", "info")}
        rep = {"run_id": self.p.run_id, "status": status, "mixed_model": self.mixed_model, "counts": counts,
               "stats": self.stats, "findings": self.findings, "audit_git_hash": git_hash(),
               "pinned_model": self.meta["claude_model"], "effort": self.meta["effort"]}
        write_json_atomic(self.p.audit, rep)
        return rep


def split_heredocs(cmd: str) -> tuple[str, list[tuple[str, str]]]:
    """Split a shell command into (command without heredoc bodies, [(header line, body), ...])."""
    lines = cmd.split("\n")
    outer, docs, i = [], [], 0
    while i < len(lines):
        line = lines[i]
        outer.append(line)
        i += 1
        for m in HEREDOC.finditer(line):
            strip_tabs, delim = m.group(1) == "-", m.group(3)
            body = []
            while i < len(lines):
                cur = lines[i].lstrip("\t") if strip_tabs else lines[i]
                i += 1
                if cur == delim:
                    break
                body.append(lines[i - 1])
            docs.append((heredoc_header(line, m.start()), "\n".join(body)))   # Amendment 7: the heredoc's own command
    return "\n".join(outer), docs


# ---- Amendment 7 (2026-11-06, preregistration §11): a heredoc's kind and target come from its own command.
# Before, the whole line was searched, so in `python a.py >/dev/null && python - <<'EOF'` the `>/dev/null` of another
# command was taken as the heredoc's target and the code body was classed as text and not scanned.
_WRAPPERS = {"env", "exec", "time", "command", "builtin", "nohup", "nice", "stdbuf", "!", "{", "then", "do", "else"}
_INTERP_WORD = re.compile(r"(?:.*/)?(python[\d.]*|bash|sh|zsh|dash|perl|node|ruby)")


def _shell_spans(line: str, pos: int, pipes: bool) -> tuple[int, int]:
    """(start, end) of the part of `line` holding `pos`, cut at list operators outside quotes and comments
    (`&&`, `||`, `;`, `&`, `(`, `)`); with pipes=True also at `|` and `|&`. `&` in a redirection (`2>&1`, `&>`) is not
    an operator."""
    start, end, i, q = 0, len(line), 0, None
    while i < len(line):
        ch, n = line[i], 0
        if q:
            if ch == "\\" and q == '"':
                i += 2
                continue
            if ch == q:
                q = None
            i += 1
            continue
        if ch == "\\":
            i += 2
            continue
        if ch in "'\"":
            q = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t;&|()"):
            if i > pos:
                end = i
            break
        elif line.startswith(("&&", "||", ";;"), i):
            n = 2
        elif ch == "|":
            n = (2 if line.startswith("|&", i) else 1) if pipes else 0
        elif ch in ";()":
            n = 1
        elif ch == "&" and not (i and line[i - 1] in "<>") and not line.startswith("&>", i):
            n = 1
        if n:
            if i + n <= pos:
                start = i + n
            elif i >= pos:
                end = i
                break
            i += n
            continue
        i += 1
    return start, end


def heredoc_header(line: str, pos: int) -> str:
    """The pipeline holding the heredoc operator at `pos` (list operators cut it, pipes do not), operator included."""
    a, b = _shell_spans(line, pos, pipes=False)
    return line[a:b]


def heredoc_kind(header: str) -> tuple[str, str]:
    """('shell' | 'code' | 'text', target) of the first heredoc in `header`, from its own command (Amendment 7).
    A body read by an interpreter (the command word of the heredoc's pipeline stage) is code whatever the redirections;
    otherwise a redirection of that stage to a .sh / code file makes it shell / code; with no redirection, an interpreter
    anywhere in the pipeline (`cat <<EOF | python -`, `pixi run python - <<EOF`) makes it code. Anything else is text."""
    m = HEREDOC.search(header)
    if m:
        a, b = _shell_spans(header, m.start(), pipes=False)
        pipeline, pos = header[a:b], m.start() - a
    else:
        pipeline, pos = header, 0
    sa, sb = _shell_spans(pipeline, pos, pipes=True)
    stage = HEREDOC.sub(" ", pipeline[sa:sb])
    words = [w for w in re.split(r"\s+", stage) if w]
    while words and (words[0] in _WRAPPERS or words[0] == "timeout" or re.fullmatch(r"[A-Za-z_]\w*=\S*", words[0])):
        if words.pop(0) == "timeout":                       # timeout [-k N] [-s SIG] DURATION cmd
            while words and words[0].startswith("-"):
                words.pop(0)
                if words and not words[0].startswith("-") and not re.fullmatch(r"[\d.]+[smhd]?", words[0]):
                    words.pop(0)
            if words and re.fullmatch(r"[\d.]+[smhd]?", words[0]):
                words.pop(0)
    cw = _INTERP_WORD.fullmatch(words[0]) if words else None
    if cw:
        return ("shell" if cw.group(1) in ("bash", "sh", "zsh", "dash") else "code"), cw.group(1)
    tgt = REDIRECT.search(stage)
    target = tgt.group(1) if tgt else ""
    suffix = Path(target).suffix
    if suffix in (".sh", ".bash"):
        return "shell", target
    if suffix in CODE_SUFFIXES:
        return "code", target
    if not target:
        im = INTERP.search(HEREDOC.sub(" ", pipeline))
        if im:
            return ("shell" if im.group(1) in ("bash", "sh", "zsh", "dash") else "code"), im.group(1)
    return "text", target



def _walk_dicts(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from _walk_dicts(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk_dicts(v)


def _count(it) -> dict:
    out: dict = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


def _user_text(content) -> str | None:
    """Text of a human prompt; None for tool results, system/command wrappers and empty content."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            return None
        text = "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    else:
        return None
    t = text.strip()
    if not t or t.startswith("<system-reminder>") or t.startswith("<local-command-caveat>") \
            or t.startswith("<local-command-stdout>") or t.startswith("[Request interrupted"):
        return None
    return t


def summary_line(rep: dict) -> str:
    s = rep["stats"]
    calls = s.get("calls_charged", {})
    return (f"| {rep['run_id']} | {rep['status']}{' (mixed-model)' if rep['mixed_model'] else ''} | "
            f"invalid {rep['counts']['invalid']}, review {rep['counts']['review']} | "
            f"calls A {calls.get('A', '-')} / B {calls.get('B', '-')} | submissions {s.get('submissions')} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "harness.yaml"))
    ap.add_argument("--seed", type=int, default=0, help="unused; script convention")
    args = ap.parse_args()
    rep = Auditor(args.run, load_config(args.config)).run()
    for f in rep["findings"]:
        if f["severity"] != "info":
            print(f"[{f['severity']}] {f['category']}: {f['message']}  ({f['where']})")
    print(f"\naudit written to {RunPaths(args.run).audit}")
    print(summary_line(rep))


if __name__ == "__main__":
    main()
