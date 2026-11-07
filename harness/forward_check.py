"""Read-only AST checker: did a WSP session run (part of) the network outside fp_query? Not a §9 file.

  pixi run python harness/forward_check.py --run <run_id> [--json PATH] [--md PATH]

Parses the code (not the text) of every script in runs/wsp/<id>/archive/rundir and every script or inline program
written or run through the transcripts (Write/Edit/MultiEdit, heredocs, `python -c`, `python - <<EOF`). Comments and
docstrings never count. A light taint analysis follows values from weights.npz:

  weight-derived (W)    loaded from weights.npz (key tracked, e.g. conv2_w), or transformed weights only;
  constructed input     np.zeros/ones/eye/random..., numeric literals, (I);
  sequence (S)          string literals of A/C/G/T, or anything derived from them;
  parameter (P)         a function parameter in the per-definition analysis;
  other (U)             unknown origin, e.g. loaded from another file.

A *layer application* is a linear op (matmul/@, dot, einsum, tensordot, convolve/correlate, conv1d, linear, torch
Conv1d/Linear modules, elementwise weight*input) combining a weight with a non-weight operand, or with an activation;
its result is an *activation*. Weight x weight products (weight algebra, e.g. W2 @ W1) are not activations.
A *forward step* is a nonlinearity (np.maximum(a, 0), clip(0, ...), relu, clamp(min=0), where(a>0, a, 0), a*(a>0),
a[a<0]=0, sigmoid/tanh) or a pooling (max/amax/max_pool1d/MaxPool1d...) applied to an activation.

Calls into the agent's own scripts are followed: user functions (analysed at each call site with the real arguments),
`import`/`from ... import` (the imported module's top level runs, `if __name__ == "__main__"` blocks do not),
`exec(open(...).read())` and runpy. Input type of a chain: "DNA/one-hot" (sequence-derived, or anything fed to an
input-layer tensor, i.e. a conv weight with 4 input channels), "synthetic activations" (constructed input fed to a
later layer), or "other" (weights fed as input, parameters, unknown files).

Output: names, line numbers and categories only -- never code text, string literals or tool results; scripts inside a
submission directory are not analysed (counted only). Heuristic: it reports evidence for a reviewer; it decides nothing.
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
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from path_review import PathReview  # noqa: E402
from review_packet import heredoc_segments, tool_uses  # noqa: E402

from wspg.audit import heredoc_kind  # noqa: E402
import wspg.audit  # noqa: E402
import path_review  # noqa: E402
import review_packet  # noqa: E402


def checker_sha256() -> str:
    """Version of this check: forward_check.py with the code it reuses (heredoc parsing in audit.py and review_packet.py,
    the cwd replay in path_review.py; Amendment 7)."""
    files = [Path(__file__), Path(wspg.audit.__file__), Path(path_review.__file__), Path(review_packet.__file__)]
    return hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()
from wspg.utils import REPO_ROOT, git_hash  # noqa: E402
from wspg.wsp import RunPaths  # noqa: E402

# ----------------------------------------------------------------------------- vocabulary
LINEAR_CALLS = {"dot", "matmul", "einsum", "tensordot", "inner", "vdot", "convolve", "correlate", "correlate1d",
                "convolve1d", "fftconvolve", "oaconvolve", "conv1d", "conv2d", "conv_transpose1d", "linear", "mm",
                "bmm", "mv", "addmm", "addmv", "baddbmm", "chain_dot", "multi_dot"}
CONSTRUCTORS = {"zeros", "ones", "full", "empty", "eye", "identity", "arange", "linspace", "logspace", "rand", "randn",
                "randint", "random", "random_sample", "normal", "uniform", "standard_normal", "integers", "zeros_like",
                "ones_like", "full_like", "empty_like", "rand_like", "randn_like", "tri", "diag_indices"}
POOL_CALLS = {"max_pool1d", "max_pool2d", "adaptive_max_pool1d", "avg_pool1d", "adaptive_avg_pool1d", "lp_pool1d",
              "max", "amax", "nanmax", "maximum_filter1d"}
NONLIN_CALLS = {"relu", "relu_", "relu6", "leaky_relu", "elu", "gelu", "silu", "softplus", "sigmoid", "tanh",
                "expit", "softmax", "log_softmax", "hardtanh"}
MODULES_LIN = {"Conv1d", "Conv2d", "Linear", "LazyConv1d", "LazyLinear", "ConvTranspose1d"}
MODULES_NL = {"ReLU", "ReLU6", "LeakyReLU", "ELU", "GELU", "SiLU", "Sigmoid", "Tanh", "Softplus", "Softmax"}
MODULES_POOL = {"MaxPool1d", "MaxPool2d", "AdaptiveMaxPool1d", "AvgPool1d", "AdaptiveAvgPool1d"}
DNA_LITERAL = re.compile(r"^[ACGTNacgtn]{4,}$")
SEQ_ALPHABET = re.compile(r"^[ACGT]{4}$|^[acgt]{4}$")
INPUT_ORIGINS = {"I", "S", "U", "P"}
PY_RUN = re.compile(r"(?:^|[\s;&|(/])python[\d.]*((?:\s+-[A-Za-z]+)*)\s+([^\s;&|<>()]+\.py)\b")
PY_MOD = re.compile(r"(?:^|[\s;&|(/])python[\d.]*(?:\s+-[A-Za-z]+)*\s+-m\s+([\w.]+)")
PY_INLINE = re.compile(r"(?:^|[\s;&|(/])python[\d.]*(?:\s+-[A-Za-z]+)*\s+-c\s")


# ----------------------------------------------------------------------------- values
@dataclass(frozen=True)
class Val:
    o: frozenset = frozenset()       # origins: W I C S U P
    wk: frozenset = frozenset()      # weight keys
    act: bool = False                # an activation (weight applied to input)
    layers: tuple = ()               # weight keys applied as layers, in order
    inp: frozenset = frozenset()     # origins of the input an activation chain started from
    tags: frozenset = frozenset()    # NPZ, MOD:lin|nl|pool|seq, FUNC:<id>, CLASS:<id>, INST:<id>, MODULE:<rel>
    parts: tuple = ()                # nn.Sequential components

    @property
    def is_weight(self) -> bool:
        """Weights or weight algebra. A constructed array (zeros, ...) filled with weight-derived values is input."""
        return "W" in self.o and not self.act and not self.o & INPUT_ORIGINS

    @property
    def is_input(self) -> bool:
        return self.act or bool(self.o & INPUT_ORIGINS)


def join(*vals: Val) -> Val:
    vals = [v for v in vals if v is not None]
    if not vals:
        return Val()
    layers: list = []
    for v in vals:
        layers += [x for x in v.layers if x not in layers]
    return Val(frozenset().union(*(v.o for v in vals)), frozenset().union(*(v.wk for v in vals)),
               any(v.act for v in vals), tuple(layers), frozenset().union(*(v.inp for v in vals)),
               frozenset(t for v in vals for t in v.tags if t.startswith(("NPZ",))), ())


C = Val(frozenset({"C"}))
U = Val(frozenset({"U"}))


def _name(node) -> str:
    """Last component of a called name: np.random.randn -> randn; x.max -> max."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _dotted(node) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _is_zero(node) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and node.value == 0 \
        or isinstance(node, ast.UnaryOp) and _is_zero(node.operand)


# ----------------------------------------------------------------------------- events
@dataclass
class Event:
    unit: str          # code unit (script path or transcript location of inline code)
    line: int
    kind: str          # layer | nonlin | pool
    func: str          # enclosing function qualname ("" at module level)
    layers: tuple
    inp: frozenset
    first_input_layer: bool

    def input_type(self) -> str:
        return input_type(self.inp, self.first_input_layer)


DNA_TYPE = "DNA/one-hot (sequence-derived)"
INPUT_SPACE_TYPE = "constructed array fed to the input layer (e.g. PWM, base composition, synthetic one-hot)"


def input_type(inp: frozenset, first_input_layer: bool) -> str:
    if "S" in inp:
        return DNA_TYPE
    if first_input_layer:
        return INPUT_SPACE_TYPE
    if inp and inp <= {"I", "C"}:
        return "synthetic activations"
    if "I" in inp and inp <= {"I", "C", "W"}:
        return "synthetic activations (constructed array holding weight-derived values)"
    if "P" in inp:
        return "parameter"
    if "U" in inp:
        return "other: input of unknown origin (e.g. loaded from a file)"
    if "W" in inp:
        return "other: weight-derived input"
    return "other"


@dataclass
class CallSite:
    unit: str
    line: int
    caller: str
    callee: str        # function qualname (with defining unit)
    in_loop: bool
    mult: float | None = 1   # evaluations per run of the enclosing code (product of literal loop lengths; None = unknown)
    kinds: set = field(default_factory=set)
    types: set = field(default_factory=set)
    layers: list = field(default_factory=list)


# ----------------------------------------------------------------------------- project
class Project:
    """The agent's code: script versions by path, input-layer weight keys, saved arrays (np.save) by file name."""

    def __init__(self, input_layer_keys: set[str], output_layer_keys: set[str] = frozenset(),
                 saved: dict | None = None):
        self.files: dict[str, str] = {}
        self.input_layer_keys = set(input_layer_keys)
        self.output_layer_keys = set(output_layer_keys)
        self.saved: dict[str, Val] = saved if saved is not None else {}

    def resolve(self, name: str) -> str | None:
        """A path, file name or module name -> a known script path."""
        n = name.strip().strip("'\"")
        n = re.sub(r"^/home/[^/]+/runs/[0-9a-f]{16}/", "", n)
        while n.startswith("./"):
            n = n[2:]
        if n in self.files:
            return n
        cand = n if n.endswith(".py") else n.replace(".", "/") + ".py"
        if cand in self.files:
            return cand
        base = Path(cand).name
        hits = [f for f in self.files if Path(f).name == base]
        return hits[0] if len(hits) == 1 else None


class Analyzer:
    """Abstract interpretation of one execution (a script run as __main__, an inline program, or a module import)."""

    MAX_DEPTH = 6

    def __init__(self, project: Project):
        self.p = project
        self.events: list[Event] = []
        self.calls: list[CallSite] = []
        self.funcs: dict[str, tuple] = {}       # id -> (FunctionDef, globals env, unit, qualname)
        self.classes: dict[str, tuple] = {}     # id -> (ClassDef, globals env, unit)
        self.instances: dict[str, dict] = {}    # id -> attribute env
        self.modules: dict[str, dict] = {}      # rel -> module env (imports executed once per execution)
        self.stack: list[str] = []
        self.parse_errors: list[str] = []
        self.imports: set[str] = set()
        self.ctx_unit = ""
        self.ctx_func = ""
        self.loop = 0
        self.mult: float | None = 1
        self.loopvals: dict[str, list[int]] = {}
        self._seen: set = set()
        self._collect: list[list] = []
        self._call_keys: set = set()

    # -- entry points
    def run_unit(self, unit: str, code: str, as_main: bool, env: dict | None = None) -> dict:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            self.parse_errors.append(f"{unit}: syntax error at line {e.lineno}")
            return env if env is not None else {}
        env = env if env is not None else {}
        env.setdefault("__name__", Val(frozenset({"C"}), tags=frozenset({"MAIN" if as_main else "NOTMAIN"})))
        saved = (self.ctx_unit, self.ctx_func, self.loop)
        self.ctx_unit, self.ctx_func, self.loop = unit, "", 0
        self.block(tree.body, env, env)
        self.ctx_unit, self.ctx_func, self.loop = saved
        return env

    def run_module(self, rel: str, as_main: bool, env: dict | None = None) -> dict | None:
        if rel not in self.p.files:
            return None
        key = rel if not as_main else None
        if key and key in self.modules:
            return self.modules[key]
        if rel in self.stack or len(self.stack) >= self.MAX_DEPTH:
            return self.modules.get(rel, {})
        self.stack.append(rel)
        self.imports.add(rel)
        menv: dict = {} if env is None else env
        if key:
            self.modules[key] = menv
        self.run_unit(rel, self.p.files[rel], as_main, menv)
        self.stack.pop()
        return menv

    # -- events
    def emit(self, kind: str, line: int, v: Val, extra_layer: str | None = None):
        layers = v.layers + ((extra_layer,) if extra_layer and extra_layer not in v.layers else ())
        if kind in ("nonlin", "pool") and any(k in self.p.output_layer_keys for k in v.layers):
            return          # a max/clip over logits (the network's output) is not a step inside the network
        # an input-layer tensor applied to constructed or sequence input: one-hot space. Unknown-origin operands
        # (e.g. a vector loaded from a file of unknown provenance) are not assumed to be sequences.
        first = bool(layers) and layers[0] in self.p.input_layer_keys and bool(v.inp & {"I", "S", "P"})
        ev = Event(self.ctx_unit, line, kind, self.ctx_func, layers, v.inp, first)
        key = (self.ctx_unit, line, kind, self.ctx_func, layers, v.inp, first)
        if key not in self._seen:
            self._seen.add(key)
            self.events.append(ev)
        for c in self._collect:
            c.append(ev)

    # -- statements
    def block(self, body, env, genv):
        for st in body:
            self.stmt(st, env, genv)

    def stmt(self, st, env, genv):
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fid = f"{self.ctx_unit}::{(self.ctx_func + '.') if self.ctx_func else ''}{st.name}@{st.lineno}"
            qual = f"{(self.ctx_func + '.') if self.ctx_func else ''}{st.name}"
            self.funcs[fid] = (st, genv, self.ctx_unit, qual)
            env[st.name] = Val(tags=frozenset({f"FUNC:{fid}"}))
        elif isinstance(st, ast.ClassDef):
            cid = f"{self.ctx_unit}::{st.name}@{st.lineno}"
            self.classes[cid] = (st, genv, self.ctx_unit)
            env[st.name] = Val(tags=frozenset({f"CLASS:{cid}"}))
        elif isinstance(st, ast.Assign):
            v = self.ev(st.value, env, genv)
            for t in st.targets:
                self.assign(t, v, env, genv, st)
        elif isinstance(st, ast.AnnAssign):
            if st.value is not None:
                self.assign(st.target, self.ev(st.value, env, genv), env, genv, st)
        elif isinstance(st, ast.AugAssign):
            cur = self.ev(st.target, env, genv)
            rhs = self.ev(st.value, env, genv)
            v = self.binop(st.op, cur, rhs, st.lineno)
            self.assign(st.target, v, env, genv, st, augment=True)
        elif isinstance(st, (ast.For, ast.AsyncFor)):
            it = self.ev(st.iter, env, genv)
            n = self.iter_len(st.iter, env, genv)
            self.assign(st.target, self.element(it), env, genv, st)
            name = st.target.id if isinstance(st.target, ast.Name) else None
            saved_vals = self.loopvals.get(name) if name else None
            if name:                      # values of a literal range's loop variable, for inner range(<var>, ...)
                vals = self.range_values(st.iter, env, genv)
                if vals is None:
                    self.loopvals.pop(name, None)
                else:
                    self.loopvals[name] = vals
            self.loop_body(st.body, env, genv, n)
            if name:
                if saved_vals is None:
                    self.loopvals.pop(name, None)
                else:
                    self.loopvals[name] = saved_vals
            self.block(st.orelse, env, genv)
        elif isinstance(st, ast.While):
            self.ev(st.test, env, genv)
            self.loop_body(st.body, env, genv, None)
            self.block(st.orelse, env, genv)
        elif isinstance(st, ast.If):
            main_guard = self._main_guard(st.test)
            is_main = "MAIN" in self.lookup("__name__", env, genv).tags
            if main_guard == "main" and not is_main:
                self.block(st.orelse, env, genv)
                return
            if main_guard == "notmain" and is_main:
                self.block(st.orelse, env, genv)
                return
            self.ev(st.test, env, genv)
            self.block(st.body, env, genv)
            self.block(st.orelse, env, genv)
        elif isinstance(st, (ast.With, ast.AsyncWith)):
            for item in st.items:
                v = self.ev(item.context_expr, env, genv)
                if item.optional_vars is not None:
                    self.assign(item.optional_vars, v, env, genv, st)
            self.block(st.body, env, genv)
        elif isinstance(st, ast.Try) or type(st).__name__ == "TryStar":
            self.block(st.body, env, genv)
            for h in st.handlers:
                self.block(h.body, env, genv)
            self.block(st.orelse, env, genv)
            self.block(st.finalbody, env, genv)
        elif isinstance(st, ast.Return):
            v = self.ev(st.value, env, genv) if st.value is not None else C
            env["__return__"] = join(env.get("__return__"), v) if "__return__" in env else v
        elif isinstance(st, ast.Expr):
            self.ev(st.value, env, genv)
        elif isinstance(st, ast.Import):
            for a in st.names:
                rel = self.p.resolve(a.name)
                if rel:
                    menv = self.run_module(rel, False)
                    env[a.asname or a.name.split(".")[0]] = Val(tags=frozenset({f"MODULE:{rel}"}))
                    if menv is not None and a.asname is None and "." in a.name:
                        env[a.name.split(".")[0]] = Val(tags=frozenset({f"MODULE:{rel}"}))
                else:
                    env[a.asname or a.name.split(".")[0]] = Val(tags=frozenset({f"EXT:{a.name}"}))
        elif isinstance(st, ast.ImportFrom):
            mod = st.module or ""
            rel = self.p.resolve(mod) if mod else None
            for a in st.names:
                sub = self.p.resolve(f"{mod}.{a.name}" if mod else a.name)
                if sub and not rel:
                    self.run_module(sub, False)
                    env[a.asname or a.name] = Val(tags=frozenset({f"MODULE:{sub}"}))
                elif rel:
                    menv = self.run_module(rel, False) or {}
                    if a.name == "*":
                        env.update({k: v for k, v in menv.items() if not k.startswith("_")})
                    else:
                        env[a.asname or a.name] = menv.get(a.name, U)
                else:
                    env[a.asname or a.name] = Val(tags=frozenset({f"EXT:{mod}.{a.name}"}))
        elif isinstance(st, (ast.Global, ast.Nonlocal, ast.Pass, ast.Break, ast.Continue, ast.Delete)):
            pass
        elif isinstance(st, (ast.Raise, ast.Assert)):
            for ch in ast.iter_child_nodes(st):
                if isinstance(ch, ast.expr):
                    self.ev(ch, env, genv)
        elif hasattr(ast, "Match") and isinstance(st, ast.Match):
            self.ev(st.subject, env, genv)
            for c in st.cases:
                self.block(c.body, env, genv)

    def loop_body(self, body, env, genv, n):
        saved = self.mult
        self.loop += 1
        self.mult = None if saved is None or n is None else saved * n
        for _ in range(2):          # twice, so values carried around the loop reach the top of the body
            self.block(body, env, genv)
        self.loop -= 1
        self.mult = saved

    def _int_values(self, node) -> list[int] | None:
        """Possible values of a range() argument: an int literal, -literal, an enclosing literal loop's variable, or
        either of those +/- an int literal."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return [node.value]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v = self._int_values(node.operand)
            return None if v is None else [-x for x in v]
        if isinstance(node, ast.Name) and node.id in self.loopvals:
            return self.loopvals[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            a, b = self._int_values(node.left), self._int_values(node.right)
            if a is None or b is None or (len(a) > 1 and len(b) > 1):
                return None
            return [x + y if isinstance(node.op, ast.Add) else x - y for x in a for y in b]
        return None

    def range_values(self, node, env, genv) -> list[int] | None:
        """The values of range(<literal ints>), or None."""
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range" and node.args:
            vals = [self._int_values(a) for a in node.args]
            if all(v is not None and len(v) == 1 for v in vals):
                try:
                    return list(range(*[v[0] for v in vals]))
                except (TypeError, ValueError):
                    return None
        return None

    def iter_len(self, node, env, genv) -> float | None:
        """Number of iterations of a literal loop: range(<ints>), a list/tuple literal or a name bound to one. A range
        whose bounds depend on an enclosing literal loop's variable (range(f, 64)) gets its mean length over that
        variable's values, which makes the product over nested loops exact (before `continue`/`if`)."""
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return None if any(isinstance(e, ast.Starred) for e in node.elts) else len(node.elts)
        if isinstance(node, ast.Name):
            v = self.lookup(node.id, env, genv)
            return len(v.parts) if v.parts and "LIT" in v.tags else None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            f, a = node.func.id, node.args
            if f == "range" and a and len(a) <= 3:
                vals = [self._int_values(x) for x in a]
                if any(v is None for v in vals) or sum(len(v) > 1 for v in vals) > 1:
                    return None
                try:
                    lens = [len(range(*combo)) for combo in _combos(vals)]
                except (TypeError, ValueError):
                    return None
                return sum(lens) / len(lens)
            if f in ("enumerate", "list", "tuple", "reversed", "sorted") and a:
                return self.iter_len(a[0], env, genv)
            if f == "zip" and a:
                ns = [self.iter_len(x, env, genv) for x in a]
                return None if None in ns else min(ns)
        return None

    @staticmethod
    def _main_guard(test) -> str | None:
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.left, ast.Name) \
                and test.left.id == "__name__" and isinstance(test.comparators[0], ast.Constant) \
                and test.comparators[0].value == "__main__":
            return "main" if isinstance(test.ops[0], ast.Eq) else "notmain" if isinstance(test.ops[0], ast.NotEq) else None
        return None

    def assign(self, t, v, env, genv, st, augment=False):
        if isinstance(t, ast.Name):
            env[t.id] = v
        elif isinstance(t, (ast.Tuple, ast.List)):
            same = len(v.parts) == len(t.elts) and not any(isinstance(el, ast.Starred) for el in t.elts)
            for i, el in enumerate(t.elts):
                self.assign(el.value if isinstance(el, ast.Starred) else el, v.parts[i] if same else self.element(v),
                            env, genv, st)
        elif isinstance(t, ast.Subscript):
            base = self._base_name(t)
            idx = self.ev(t.slice, env, genv)
            cur = self.ev(t.value, env, genv)
            # a[a < 0] = 0 / a[mask] = 0 on an activation: ReLU written as masked assignment
            if cur.act and _is_zero(getattr(st, "value", None)) and isinstance(t.slice, ast.Compare):
                self.emit("nonlin", st.lineno, cur)
            if base:
                seq = Val(frozenset({"S"})) if "S" in idx.o else None
                new = join(cur, v, seq) if not augment else join(cur, v)
                self._set_path(base, new, env, genv)
        elif isinstance(t, ast.Attribute):
            path = _dotted(t)
            if path:
                env[path] = v
                root = path.split(".")[0]
                if root in env and (v.o & {"W"} or v.wk):     # module.weight = W[...] : the module carries the weight
                    env[root] = replace(env[root], o=env[root].o | v.o, wk=env[root].wk | v.wk)
                inst = self._instance_of(root, env, genv)
                if inst is not None and len(path.split(".")) == 2:
                    inst[path.split(".")[1]] = v
        elif isinstance(t, ast.Starred):
            self.assign(t.value, v, env, genv, st)

    def _base_name(self, t) -> str | None:
        while isinstance(t, (ast.Subscript, ast.Attribute)):
            t = t.value
        return t.id if isinstance(t, ast.Name) else None

    def _set_path(self, name, v, env, genv):
        if name in env or env is genv:
            env[name] = v
        else:
            genv[name] = v

    def _instance_of(self, name, env, genv):
        v = env.get(name) or genv.get(name)
        if v is None:
            return None
        for t in v.tags:
            if t.startswith("INST:"):
                return self.instances.setdefault(t[5:], {})
        return None

    def lookup(self, name, env, genv) -> Val:
        if name in env:
            return env[name]
        if name in genv:
            return genv[name]
        return U if name not in ("np", "numpy", "torch", "F", "nn", "scipy", "math", "itertools") else \
            Val(tags=frozenset({f"EXT:{name}"}))

    @staticmethod
    def element(v: Val) -> Val:
        return replace(v, tags=frozenset(t for t in v.tags if t.startswith("NPZ")), parts=())

    # -- expressions
    def ev(self, node, env, genv) -> Val:
        if node is None:
            return C
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                if DNA_LITERAL.match(node.value) or SEQ_ALPHABET.match(node.value):
                    return Val(frozenset({"S"}))
                return Val(frozenset({"C"}), tags=frozenset({"STR"}))
            return C
        if isinstance(node, ast.Name):
            return self.lookup(node.id, env, genv)
        if isinstance(node, ast.Attribute):
            path = _dotted(node)
            if path and path in env:
                return env[path]
            base = self.ev(node.value, env, genv)
            for t in base.tags:
                if t.startswith("MODULE:"):
                    menv = self.modules.get(t[7:]) or {}
                    return menv.get(node.attr, U)
                if t.startswith("INST:"):
                    inst = self.instances.get(t[5:], {})
                    if node.attr in inst:
                        return inst[node.attr]
                if t.startswith("EXT:"):
                    return Val(tags=frozenset({f"EXT:{t[4:]}.{node.attr}"}))
            if node.attr in ("T", "mT", "data", "real", "values", "shape", "ndim", "size", "dtype"):
                return C if node.attr in ("shape", "ndim", "size", "dtype") else self.element(base)
            return self.element(base)
        if isinstance(node, ast.Subscript):
            base = self.ev(node.value, env, genv)
            idx = self.ev(node.slice, env, genv)
            if "NPZ" in base.tags:
                key = node.slice.value if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str) else "?"
                return Val(frozenset({"W"}), frozenset({key}))
            out = self.element(base)
            if "S" in idx.o and not base.o - {"S", "C"}:   # 'ACGT'[i], mapping['A'] ...
                out = join(out, Val(frozenset({"S"})))
            return out
        if isinstance(node, ast.BinOp):
            left, right = self.ev(node.left, env, genv), self.ev(node.right, env, genv)
            v = self.binop(node.op, left, right, node.lineno)
            # a * (a > 0): ReLU as a mask product
            if isinstance(node.op, ast.Mult) and (left.act or right.act):
                for a, b in ((node.left, node.right), (node.right, node.left)):
                    if isinstance(b, ast.Compare) and len(b.comparators) == 1 and _is_zero(b.comparators[0]) \
                            and ast.dump(b.left) == ast.dump(a):
                        self.emit("nonlin", node.lineno, self.ev(a, env, genv))
            return v
        if isinstance(node, ast.UnaryOp):
            return self.ev(node.operand, env, genv)
        if isinstance(node, (ast.BoolOp,)):
            return join(*(self.ev(x, env, genv) for x in node.values))
        if isinstance(node, ast.Compare):
            return join(self.ev(node.left, env, genv), *(self.ev(x, env, genv) for x in node.comparators))
        if isinstance(node, ast.IfExp):
            self.ev(node.test, env, genv)
            return join(self.ev(node.body, env, genv), self.ev(node.orelse, env, genv))
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            vals = [self.ev(e.value if isinstance(e, ast.Starred) else e, env, genv) for e in node.elts]
            v = join(*vals) if vals else C
            lit = not any(isinstance(e, ast.Starred) for e in node.elts)
            return replace(v, parts=tuple(vals), tags=v.tags | ({"LIT"} if lit else set()))
        if isinstance(node, ast.Dict):
            return join(*(self.ev(x, env, genv) for x in list(node.values) + [k for k in node.keys if k is not None]))
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            local = dict(env)
            saved = self.mult
            for g in node.generators:
                n = self.iter_len(g.iter, local, genv)
                self.mult = None if self.mult is None or n is None else self.mult * n
                self.assign(g.target, self.element(self.ev(g.iter, local, genv)), local, genv, node)
                for cond in g.ifs:
                    self.ev(cond, local, genv)
            self.loop += 1
            if isinstance(node, ast.DictComp):
                v = join(self.ev(node.key, local, genv), self.ev(node.value, local, genv))
            else:
                v = self.ev(node.elt, local, genv)
            self.loop -= 1
            self.mult = saved
            return self.element(v)
        if isinstance(node, ast.Call):
            return self.call(node, env, genv)
        if isinstance(node, ast.Starred):
            return self.ev(node.value, env, genv)
        if isinstance(node, ast.JoinedStr):
            return Val(frozenset({"C"}), tags=frozenset({"STR"}))
        if isinstance(node, ast.Slice):
            return join(*(self.ev(x, env, genv) for x in (node.lower, node.upper, node.step) if x is not None))
        if isinstance(node, ast.Lambda):
            return C
        if isinstance(node, ast.NamedExpr):
            v = self.ev(node.value, env, genv)
            self.assign(node.target, v, env, genv, node)
            return v
        vals = [self.ev(ch, env, genv) for ch in ast.iter_child_nodes(node) if isinstance(ch, ast.expr)]
        return join(*vals) if vals else C

    def binop(self, op, a: Val, b: Val, line: int) -> Val:
        if isinstance(op, ast.MatMult) or isinstance(op, ast.Mult) and self._linear_pair(a, b):
            return self.linear([a, b], line)
        return join(a, b)

    @staticmethod
    def _linear_pair(a: Val, b: Val) -> bool:
        """weight * input elementwise (a hand-written convolution); weight * constant is weight algebra."""
        return (a.is_weight and b.is_input) or (b.is_weight and a.is_input)

    def linear(self, vals: list[Val], line: int) -> Val:
        weights = [v for v in vals if v.is_weight]
        inputs = [v for v in vals if v.is_input]
        if weights and inputs:
            src = join(*inputs)
            wk = sorted(frozenset().union(*(w.wk for w in weights))) or ["?"]
            inp = src.inp if src.act else frozenset(src.o - {"C"})
            new_layers = tuple(src.layers) + tuple(k for k in wk if k not in src.layers)
            out = Val(frozenset(src.o | {"W"}), frozenset(), True, new_layers, inp)
            self.emit("layer", line, replace(out, layers=new_layers))
            return out
        return join(*vals)

    # -- calls
    def call(self, node: ast.Call, env, genv) -> Val:
        fn = node.func
        name = _name(fn)
        dotted = _dotted(fn)
        args = [self.ev(a.value if isinstance(a, ast.Starred) else a, env, genv) for a in node.args]
        kw = {k.arg: self.ev(k.value, env, genv) for k in node.keywords}
        recv = self.ev(fn.value, env, genv) if isinstance(fn, ast.Attribute) else None
        callee = self.ev(fn, env, genv) if isinstance(fn, (ast.Name, ast.Attribute)) else Val()
        line = node.lineno

        # the agent's own functions, classes, instances, modules
        for t in callee.tags:
            if t.startswith("FUNC:"):
                return self.call_user(t[5:], node, args, kw, env, genv, bound=None)
            if t.startswith("CLASS:"):
                return self.instantiate(t[6:], node, args, kw)
            if t.startswith("INST:"):
                return self.call_method(t[5:], "forward", node, args, kw)
        if recv is not None:
            for t in recv.tags:
                if t.startswith("INST:") and self._has_method(t[5:], name):
                    return self.call_method(t[5:], name, node, args, kw)

        # exec(open(path).read()), runpy.run_path(path)
        if name in ("exec", "run_path", "run_module") and node.args:
            rel = self._code_path(node.args[0])
            if rel:
                if name == "exec":
                    self.run_module(rel, True, env if env is genv else genv)
                else:
                    self.run_module(rel, True)
                return C

        # torch.nn modules and their application
        if name in MODULES_LIN:
            in4 = node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == 4
            return Val(frozenset({"W"}), frozenset({"<conv in=4>" if in4 else f"<{name}>"}), tags=frozenset({"MOD:lin"}))
        if name in MODULES_NL:
            return Val(tags=frozenset({"MOD:nl"}))
        if name in MODULES_POOL:
            return Val(tags=frozenset({"MOD:pool"}))
        if name in ("Sequential", "ModuleList"):
            parts = tuple(args) if name == "Sequential" else (args[0].parts if args else ())
            return Val(tags=frozenset({"MOD:seq"}), parts=parts)
        if "MOD:lin" in callee.tags or "MOD:nl" in callee.tags or "MOD:pool" in callee.tags or "MOD:seq" in callee.tags:
            x = args[0] if args else U
            return self.apply_module(callee, x, line)

        # weights
        if name in ("load", "NpzFile") and node.args:
            a0 = node.args[0]
            s = a0.value if isinstance(a0, ast.Constant) and isinstance(a0.value, str) else self._str_path(a0, env, genv)
            if s is not None and (s.endswith(".npz") and "weights" in s or Path(s).name == "weights.npz"):
                return Val(frozenset({"W"}), frozenset(), tags=frozenset({"NPZ"}))
            if s is not None:
                return self.p.saved.get(Path(s).name, Val(frozenset({"U"})))
            return Val(frozenset({"U"}))
        if name in ("save", "savez", "savez_compressed", "save_npy") and len(node.args) >= 2:
            a0 = node.args[0]
            s = a0.value if isinstance(a0, ast.Constant) and isinstance(a0.value, str) else None
            if s:
                fn_ = Path(s).name + ("" if s.endswith((".npy", ".npz")) else ".npy")
                v = join(*args[1:], *kw.values())
                self.p.saved[fn_] = join(self.p.saved[fn_], v) if fn_ in self.p.saved else v
            return C
        if name in ("loadtxt", "genfromtxt", "fromfile", "read_csv", "open", "read", "readlines", "read_text"):
            return Val(frozenset({"U"})) if not (recv and "S" in recv.o) else Val(frozenset({"S"}))
        if name in ("dict", "OrderedDict") and args and "NPZ" in args[0].tags:
            return args[0]
        if name in ("items", "values") and recv is not None and "NPZ" in recv.tags:
            return Val(frozenset({"W"}), frozenset({"?"}))

        # constructed arrays
        if name in CONSTRUCTORS and name not in ("random",) or (name == "random" and dotted.endswith("random.random")):
            return Val(frozenset({"I"}))
        if name in ("array", "asarray", "tensor", "as_tensor", "from_numpy", "stack", "concatenate", "vstack", "hstack",
                    "cat", "copy", "astype", "float", "double", "reshape", "transpose", "flip", "roll", "pad", "squeeze",
                    "unsqueeze", "expand_dims", "repeat", "tile", "broadcast_to", "clone", "detach", "numpy", "contiguous",
                    "view", "permute", "flatten", "ravel", "swapaxes", "moveaxis", "atleast_2d", "sliding_window_view",
                    "as_strided", "unfold", "to", "cpu", "list"):
            base = join(*(args[:1] if name not in ("stack", "concatenate", "vstack", "hstack", "cat") else args),
                        recv if recv is not None and not any(t.startswith("EXT") for t in recv.tags) else None)
            if base.o == frozenset({"C"}) or not base.o:
                return Val(frozenset({"I"}))
            return self.element(base)

        # forward steps on activations
        x = recv if (recv is not None and recv.act and name in POOL_CALLS | NONLIN_CALLS | {"clip", "clamp", "clamp_min",
                                                                                              "clip_"}) else (args[0] if args else U)
        if name in NONLIN_CALLS and x.act:
            self.emit("nonlin", line, x)
            return x
        if name in ("maximum", "fmax") and len(args) == 2 and (args[0].act or args[1].act):
            other = node.args[1] if args[0].act else node.args[0]
            if _is_zero(other) or isinstance(other, ast.Constant):
                self.emit("nonlin", line, args[0] if args[0].act else args[1])
            else:
                self.emit("pool", line, join(*args))
            return join(*args)
        if name in ("clip", "clamp", "clamp_min", "clip_") and x.act:
            lo = (node.args[1] if recv is None or not recv.act else node.args[0]) if len(node.args) > (0 if recv is not None and recv.act else 1) else None
            lo = kw_node(node, "min") or kw_node(node, "a_min") or lo
            if lo is not None and _is_zero(lo) or name == "clamp_min":
                self.emit("nonlin", line, x)
            return x
        if name == "where" and len(node.args) == 3 and (args[1].act or args[2].act):
            if _is_zero(node.args[2]) or _is_zero(node.args[1]):
                self.emit("nonlin", line, args[1] if args[1].act else args[2])
            return join(*args[1:])
        if name in POOL_CALLS and x.act and not (isinstance(fn, ast.Name) and fn.id == "max"):
            self.emit("pool", line, x)
            return x
        if name in LINEAR_CALLS:
            vals = args + list(kw.values()) + ([recv] if recv is not None and not any(t.startswith("EXT") for t in recv.tags) else [])
            return self.linear(vals, line)

        # anything else: the result depends on everything it was given
        rest = list(args) + list(kw.values())
        if recv is not None and not any(t.startswith(("EXT", "MODULE")) for t in recv.tags):
            rest.append(recv)
        if not rest:
            return C
        v = join(*rest)
        return self.element(v)

    def _str_path(self, node, env, genv) -> str | None:
        """Constant pieces of a path expression (os.path.join(d, 'weights.npz'), f-strings, d + '/weights.npz')."""
        consts = [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        if isinstance(node, ast.Name):
            return None
        return consts[-1] if consts else None

    def _code_path(self, node) -> str | None:
        """exec(open('x.py').read()) / runpy.run_path('x.py') -> script path."""
        for n in ast.walk(node):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.endswith(".py"):
                return self.p.resolve(n.value)
        return None

    def apply_module(self, mod: Val, x: Val, line: int) -> Val:
        if "MOD:lin" in mod.tags:
            return self.linear([replace(mod, tags=frozenset(), act=False), x], line)
        if "MOD:nl" in mod.tags:
            if x.act:
                self.emit("nonlin", line, x)
            return x
        if "MOD:pool" in mod.tags:
            if x.act:
                self.emit("pool", line, x)
            return x
        if "MOD:seq" in mod.tags:
            for part in mod.parts:
                x = self.apply_module(part, x, line)
            return x
        return x

    def call_user(self, fid, node, args, kw, env, genv, bound) -> Val:
        if fid not in self.funcs:
            return U
        fdef, fgenv, unit, qual = self.funcs[fid]
        if fid in self.stack or len(self.stack) >= self.MAX_DEPTH:
            return join(*args, *kw.values()) if args or kw else U
        local = self.bind(fdef, ([bound] if bound is not None else []) + list(args), kw, fgenv)
        saved = (self.ctx_unit, self.ctx_func, self.loop)
        caller_unit, caller_func, in_loop, mult = self.ctx_unit, self.ctx_func, self.loop > 0, self.mult
        chain = tuple(self.stack)
        self.ctx_unit, self.ctx_func = unit, qual
        self.stack.append(fid)
        hits: list = []
        self._collect.append(hits)
        self.block(fdef.body, local, fgenv)
        self._collect.pop()
        self.stack.pop()
        self.ctx_unit, self.ctx_func, self.loop = saved
        steps = [e for e in hits if e.kind in ("nonlin", "pool")]
        first_layer = [e for e in hits if e.kind == "layer" and e.first_input_layer]
        key = (caller_unit, node.lineno, fid, chain)
        if (steps or first_layer) and key not in self._call_keys:
            self._call_keys.add(key)
            cs = CallSite(caller_unit, node.lineno, caller_func, f"{qual} ({unit})", in_loop, mult)
            for e in steps + first_layer:
                cs.kinds.add(e.kind)
                cs.types.add(e.input_type())
                cs.layers += [x for x in e.layers if x not in cs.layers]
            self.calls.append(cs)
        return local.get("__return__", C)

    def bind(self, fdef, args, kw, fgenv) -> dict:
        a = fdef.args
        params = [p.arg for p in a.posonlyargs + a.args]
        local: dict = {}
        defaults = [None] * (len(params) - len(a.defaults)) + list(a.defaults)
        for i, p in enumerate(params):
            if i < len(args):
                local[p] = args[i]
            elif p in kw:
                local[p] = kw[p]
            else:
                local[p] = self.ev(defaults[i], fgenv, fgenv) if defaults[i] is not None else U
        if a.vararg:
            local[a.vararg.arg] = join(*args[len(params):]) if len(args) > len(params) else C
        for p, d in zip(a.kwonlyargs, a.kw_defaults):
            local[p.arg] = kw.get(p.arg, self.ev(d, fgenv, fgenv) if d is not None else U)
        if a.kwarg:
            local[a.kwarg.arg] = join(*kw.values()) if kw else C
        return local

    def instantiate(self, cid, node, args, kw) -> Val:
        cdef, cgenv, unit = self.classes[cid]
        iid = f"{cid}#{node.lineno}"
        inst = self.instances.setdefault(iid, {})
        v = Val(tags=frozenset({f"INST:{iid}"}))
        for st in cdef.body:
            if isinstance(st, ast.FunctionDef):
                inst.setdefault(f"__method__{st.name}", Val(tags=frozenset({f"METHOD:{cid}:{st.name}"})))
        self.call_method(iid, "__init__", node, args, kw, inst_val=v)
        return v

    def _has_method(self, iid, mname) -> bool:
        cid = iid.split("#")[0]
        return cid in self.classes and any(isinstance(st, ast.FunctionDef) and st.name == mname
                                           for st in self.classes[cid][0].body)

    def call_method(self, iid, mname, node, args, kw, inst_val=None) -> Val:
        cid = iid.split("#")[0]
        if cid not in self.classes:
            return U
        cdef, cgenv, unit = self.classes[cid]
        for st in cdef.body:
            if isinstance(st, ast.FunctionDef) and st.name == mname:
                fid = f"{unit}::{cdef.name}.{st.name}@{st.lineno}"
                self.funcs.setdefault(fid, (st, cgenv, unit, f"{cdef.name}.{st.name}"))
                selfv = inst_val or Val(tags=frozenset({f"INST:{iid}"}))
                return self.call_user(fid, node, args, kw, {}, cgenv, bound=selfv)
        return U

    # -- per-definition analysis: every function with its parameters as unknown input (P)
    def analyse_definitions(self, unit: str):
        for fid, (fdef, fgenv, u, qual) in list(self.funcs.items()):
            if u != unit:
                continue
            a = fdef.args
            local = {p.arg: Val(frozenset({"P"})) for p in a.posonlyargs + a.args + a.kwonlyargs}
            if a.vararg:
                local[a.vararg.arg] = Val(frozenset({"P"}))
            if a.kwarg:
                local[a.kwarg.arg] = Val(frozenset({"P"}))
            if "." in qual and a.args:        # a method: self is the instance, not input
                local[a.args[0].arg] = Val()
            saved = (self.ctx_unit, self.ctx_func, self.loop)
            self.ctx_unit, self.ctx_func = u, qual
            self.stack.append(fid)
            self.block(fdef.body, local, fgenv)
            self.stack.pop()
            self.ctx_unit, self.ctx_func, self.loop = saved


def _combos(vals: list[list[int]]):
    """Argument tuples for range(): at most one argument has several values."""
    i = next((k for k, v in enumerate(vals) if len(v) > 1), None)
    if i is None:
        yield tuple(v[0] for v in vals)
        return
    for x in vals[i]:
        yield tuple(x if k == i else v[0] for k, v in enumerate(vals))


def kw_node(node: ast.Call, name: str):
    for k in node.keywords:
        if k.arg == name:
            return k.value
    return None


# ----------------------------------------------------------------------------- transcript timeline
@dataclass
class Execution:
    loc: str
    kind: str            # script | module | inline
    target: str          # script path or "inline"
    events: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    imports: set = field(default_factory=set)
    parse_errors: list = field(default_factory=list)
    version: str = ""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:10]


class Checker:
    def __init__(self, run_id: str, root: Path = REPO_ROOT):
        self.run_id = run_id
        self.root = root
        self.p = RunPaths(run_id, root)
        self.rundir = self.p.archive / "rundir"
        self.uses = tool_uses(self.p.archive, self.p.dir)
        self.input_layer_keys, self.output_layer_keys = self._layer_keys()
        self.saved: dict[str, Val] = {}          # np.save'd arrays by file name, filled in transcript order
        self.submission_dirs = self._submission_dirs()
        self.skipped: set[str] = set()
        self.archived = self._archived_scripts()
        self.versions: dict[str, list[tuple[str, str]]] = {}   # rel -> [(loc, text)]
        self.executions: list[Execution] = []
        self.notes: list[str] = []

    def _layer_keys(self) -> tuple[set[str], set[str]]:
        """Input layer: conv weights with 4 input channels. Output layer: the weight/bias producing the single logit."""
        inp, out = {"<conv in=4>"}, set()
        try:
            import numpy as np
            with np.load(self.rundir / "weights.npz") as w:
                for k in w.files:
                    shape = w[k].shape
                    if len(shape) == 3 and shape[1] == 4:
                        inp.add(k)
                    if len(shape) == 2 and shape[0] == 1:
                        out.add(k)
        except Exception:                                   # noqa: BLE001 - report without the key lists
            pass
        return inp, out

    def _submission_dirs(self) -> set[str]:
        dirs = {"phase_b"}
        for u in self.uses:
            if u["name"] == "Bash":
                for m in re.finditer(r"wsp_submit\s+phase_b\s+(\S+)", str(u["input"].get("command") or "")):
                    dirs.add(self.norm(m.group(1)).rstrip("/"))
        return dirs

    def norm(self, path: str) -> str:
        p = str(path).strip().strip("'\"")
        p = re.sub(rf"^/home/[^/]+/runs/{self.run_id}/", "", p)
        p = re.sub(r"^archive/rundir/", "", p)
        while p.startswith("./"):
            p = p[2:]
        return p

    def _blocked(self, rel: str) -> bool:
        if rel.split("/")[0] in self.submission_dirs:
            self.skipped.add(rel)
            return True
        return False

    def _archived_scripts(self) -> dict[str, str]:
        out = {}
        if self.rundir.exists():
            for f in sorted(self.rundir.rglob("*.py")):
                rel = str(f.relative_to(self.rundir))
                if "__pycache__" in rel or self._blocked(rel):
                    continue
                out[rel] = f.read_text(errors="replace")
        return out

    # -- replay the transcript: script versions and executions in order
    def replay(self):
        cur: dict[str, str] = {}
        cwd_rel = ""
        pr = PathReview(self.run_id, self.root)
        live, before = pr.live, {c["loc"]: c["before"] for c in pr.commands}     # replayed cwd at the start of each call
        for u in self.uses:
            inp, name = u["input"], u["name"]
            if u["loc"] in before:
                b = before[u["loc"]]
                cwd_rel = b[len(live) + 1:] if b.startswith(live + "/") else ""
            if name == "Write" and str(inp.get("file_path", "")).endswith(".py"):
                rel = self.norm(inp["file_path"])
                if not self._blocked(rel):
                    cur[rel] = str(inp.get("content", ""))
                    self._version(rel, u["loc"], cur[rel])
            elif name in ("Edit", "MultiEdit") and str(inp.get("file_path", "")).endswith(".py"):
                rel = self.norm(inp["file_path"])
                if self._blocked(rel):
                    continue
                text = cur.get(rel, self.archived.get(rel))
                edits = inp.get("edits") or [inp]
                ok = text is not None
                for e in edits:
                    old, new = str(e.get("old_string", "")), str(e.get("new_string", ""))
                    if ok and old in text:
                        text = text.replace(old, new) if e.get("replace_all") else text.replace(old, new, 1)
                    else:
                        ok = False
                if ok:
                    cur[rel] = text
                    self._version(rel, u["loc"], text)
                else:
                    self.notes.append(f"{u['loc']}: edit of {rel} could not be replayed; later runs use the archived file")
                    cur.pop(rel, None)
            elif name == "Bash":
                cmd = str(inp.get("command") or "")
                for seg in heredoc_segments(cmd):
                    if seg[0] == "line":
                        for m in re.finditer(r"(?:^|&&|;|\()\s*cd\s+([^\s;&|()]+)", seg[1]):
                            cwd_rel = _cd(cwd_rel, self.norm(m.group(1)))
                        self._shell_line(seg[1], u, cur, cwd_rel)
                        continue
                    header, body = seg[1], seg[2]
                    kind, target = heredoc_kind(header)
                    if kind == "code" and target.endswith(".py"):
                        rel = self.norm(target if "/" in target or not cwd_rel else f"{cwd_rel}/{target}")
                        if not self._blocked(rel):
                            cur[rel] = body
                            self._version(rel, u["loc"], body)
                    elif kind == "code" and re.fullmatch(r"python[\d.]*", Path(target).name or ""):
                        self._execute(u["loc"], "inline", "inline", body, cur)
                    elif kind == "shell" and not target:
                        for line in body.split("\n"):
                            self._shell_line(line, u, cur, cwd_rel)

    def _version(self, rel, loc, text):
        vs = self.versions.setdefault(rel, [])
        if not vs or vs[-1][1] != text:
            vs.append((loc, text))

    def _shell_line(self, line, u, cur, cwd_rel):
        for m in PY_RUN.finditer(line):
            path = m.group(2)
            rel = self.norm(path)
            if cwd_rel and "/" not in rel and not rel.startswith("/"):
                rel = f"{cwd_rel}/{rel}"
            if self._blocked(rel):
                continue
            self._execute(u["loc"], "script", rel, None, cur)
        for m in PY_MOD.finditer(line):
            self._execute(u["loc"], "module", m.group(1).replace(".", "/") + ".py", None, cur)
        if PY_INLINE.search(line):
            try:
                words = shlex.split(line, posix=True)
            except ValueError:
                words = []
            for i, w in enumerate(words[:-1]):
                if w == "-c" and re.search(r"python[\d.]*$", " ".join(words[:i])):
                    self._execute(u["loc"], "inline", "inline", words[i + 1], cur)

    def _project(self, cur) -> Project:
        pr = Project(self.input_layer_keys, self.output_layer_keys, self.saved)
        pr.files = {**self.archived, **cur}
        return pr

    def _execute(self, loc, kind, target, code, cur):
        pr = self._project(cur)
        an = Analyzer(pr)
        ex = Execution(loc, kind, target)
        if kind == "inline":
            an.run_unit(f"{loc} (inline)", code, True)
        else:
            rel = pr.resolve(target)
            if rel is None:
                self.notes.append(f"{loc}: ran {target}, which is not a known script")
                return
            ex.target = rel
            ex.version = _sha(pr.files[rel])
            an.stack.append(rel)
            an.imports.add(rel)
            an.run_unit(rel, pr.files[rel], True)
            an.stack.pop()
        ex.events, ex.calls = an.events, an.calls
        ex.imports = an.imports - {ex.target}
        ex.parse_errors = an.parse_errors
        self.executions.append(ex)

    # -- per-script analysis of the final (archived) and transcript-only versions
    def definitions(self) -> dict[str, dict]:
        out = {}
        texts = dict(self.archived)
        for rel, vs in self.versions.items():
            if rel not in texts:
                texts[rel] = vs[-1][1]
        pr = Project(self.input_layer_keys, self.output_layer_keys, self.saved)
        pr.files = dict(texts)
        for rel in sorted(texts):
            an = Analyzer(pr)
            env = {"__name__": Val(frozenset({"C"}), tags=frozenset({"NOTMAIN"}))}
            an.stack.append(rel)
            an.run_unit(rel, texts[rel], False, env)
            # also the __main__ part, for module-level events
            an2 = Analyzer(pr)
            an2.stack.append(rel)
            an2.run_unit(rel, texts[rel], True)
            an.analyse_definitions(rel)
            defs = {}
            for e in an.events:
                if e.unit == rel and e.func:
                    d = defs.setdefault(e.func, {"kinds": set(), "layers": [], "types": set(), "line": None})
                    d["kinds"].add(e.kind)
                    d["layers"] += [x for x in e.layers if x not in d["layers"]]
                    d["types"].add(e.input_type())
            for fid, (fdef, _, u, qual) in an.funcs.items():
                if u == rel and qual in defs:
                    defs[qual]["line"] = fdef.lineno
            module_level = [e for e in an2.events if e.unit == rel and not e.func]
            out[rel] = {"source": "run dir" if rel in self.archived else "transcript only (not in archived run dir)",
                        "versions": len(self.versions.get(rel, [])),
                        "forward_functions": {q: d for q, d in defs.items() if d["kinds"] & {"nonlin", "pool"}},
                        "layer_only_functions": {q: d for q, d in defs.items() if not d["kinds"] & {"nonlin", "pool"}},
                        "module_level": module_level,
                        "calls_as_main": an2.calls,
                        "parse_errors": an.parse_errors}
        return out


# ----------------------------------------------------------------------------- report
def _cd(cwd_rel: str, d: str) -> str:
    """Directory (relative to the run dir; "" = the run dir) after `cd d` from cwd_rel. Outside the run dir -> ""."""
    d = d.rstrip("/")
    if d in ("", "~", ".") or d.startswith(("/", "~")):
        return cwd_rel if d == "." else ""
    p = posixpath.normpath(posixpath.join(cwd_rel or ".", d))
    return "" if p == "." or p.startswith("..") else p


def summarise_execution(ex: Execution) -> dict:
    steps = [e for e in ex.events if e.kind in ("nonlin", "pool")]
    first = [e for e in ex.events if e.kind == "layer" and e.first_input_layer]
    layers = [e for e in ex.events if e.kind == "layer"]
    evals: dict[str, int | None] = {}
    for c in ex.calls:
        name = c.callee.split(" ")[0]
        evals[name] = None if evals.get(name, 0) is None or c.mult is None else evals.get(name, 0) + c.mult
    evals = {k: (None if v is None else int(round(v))) for k, v in evals.items()}
    return {"forward": bool(steps or first),
            "types": sorted({e.input_type() for e in steps + first}),
            "units": sorted({e.unit for e in steps + first}),
            "layer_types": sorted({e.input_type() for e in layers}),
            "evaluations": {k: ("unknown" if v is None else v) for k, v in evals.items()},
            "module_level_steps": sum(1 for e in steps + first if not e.func)}


def build_report(ck: Checker) -> dict:
    ck.replay()
    defs = ck.definitions()
    execs = []
    for ex in ck.executions:
        s = summarise_execution(ex)
        execs.append({"loc": ex.loc, "kind": ex.kind, "target": ex.target, "version": ex.version,
                      "imports": sorted(ex.imports), **s,
                      "call_sites": [{"unit": c.unit, "line": c.line, "callee": c.callee, "in_loop": c.in_loop,
                                      "mult": c.mult, "ops": sorted(c.kinds), "input": sorted(c.types),
                                      "layers": c.layers} for c in ex.calls],
                      "parse_errors": ex.parse_errors})
    per_script = {}
    for rel in sorted(defs):
        runs = [e for e in execs if e["target"] == rel]
        via = [e for e in execs if rel in e["imports"] or rel in e["units"] and e["target"] != rel]
        per_script[rel] = {"runs": len(runs), "runs_forward": sum(1 for e in runs if e["forward"]),
                           "imported_by_runs": len(via),
                           "runs_forward_via_this": sum(1 for e in via if rel in e["units"])}
    fwd = [e for e in execs if e["forward"]]
    known = [v for e in fwd for v in e["evaluations"].values() if isinstance(v, int)]
    unknown = sum(1 for e in fwd for v in e["evaluations"].values() if not isinstance(v, int))
    local = {"forward_executions": len(fwd), "executions": len(execs),
             "evaluations": sum(known) if not unknown else None, "evaluations_known_part": sum(known),
             "evaluation_sites_unknown": unknown,
             "module_level_steps": sum(e["module_level_steps"] for e in fwd),
             "input_types": sorted({t for e in fwd for t in e["types"]})}
    return {"run": ck.run_id, "generated": datetime.now().isoformat(timespec="seconds"), "git": git_hash()[:12],
            "checker_sha256": checker_sha256(), "local": local,
            "input_layer_keys": sorted(k for k in ck.input_layer_keys if not k.startswith("<")),
            "output_layer_keys": sorted(ck.output_layer_keys),
            "scripts": defs, "executions": execs, "per_script": per_script,
            "skipped_in_submission_dirs": len(ck.skipped), "notes": ck.notes,
            "pycache": sorted(p.name for p in ck.rundir.rglob("__pycache__/*.pyc")) if ck.rundir.exists() else []}


def summary_line(r: dict) -> str:
    fwd = [e for e in r["executions"] if e["forward"]]
    types = sorted({t for e in fwd for t in e["types"]})
    scripts = sorted({u.split(" ")[0] for e in fwd for u in e["units"]})
    return (f"forward_check {r['run']}: {'FORWARD' if fwd else 'none'}; {len(fwd)} of {len(r['executions'])} "
            f"executions reach forward computation" + (f"; input: {' | '.join(types)}; in: {', '.join(scripts)}" if fwd else ""))


def render_md(r: dict) -> str:
    def fmt_set(s):
        return ", ".join(sorted(s)) if s else "-"

    L = [f"# Forward check, run {r['run']}", "",
         f"Generated {r['generated']} by harness/forward_check.py at git {r['git']}. Static AST analysis; names, line "
         "numbers and categories only. Heuristic evidence for a reviewer, not a decision.", "",
         f"Input-layer tensors (4 input channels): {', '.join(r['input_layer_keys']) or 'none found'}; output layer "
         f"(1 output; max/clip after it is on logits, not counted): {', '.join(r['output_layer_keys']) or 'none found'}. "
         f"Scripts analysed: {len(r['scripts'])}. Transcript executions found: {len(r['executions'])}. "
         f"Code files in submission directories not analysed: {r['skipped_in_submission_dirs']}.",
         f"Compiled modules in __pycache__ (imported at least once): {', '.join(r['pycache']) or 'none'}.", "",
         "## Scripts that define or run forward computation", ""]
    any_fwd = False
    for rel, d in r["scripts"].items():
        ml = [e for e in d["module_level"] if e.kind in ("nonlin", "pool") or e.first_input_layer]
        calls = [c for c in d["calls_as_main"]]
        if not d["forward_functions"] and not ml and not calls:
            continue
        any_fwd = True
        ps = r["per_script"][rel]
        L += [f"### {rel}", f"Source: {d['source']}; versions written in the transcript: {d['versions']}. "
              f"Run as a script {ps['runs']}x ({ps['runs_forward']} of them reaching forward steps); imported or "
              f"exec'd by {ps['imported_by_runs']} other run(s) ({ps['runs_forward_via_this']} reaching forward "
              "steps in this file)."]
        for q, fd in d["forward_functions"].items():
            L.append(f"- function `{q}` (line {fd['line']}): applies {fmt_set(fd['kinds'])} to activations of "
                     f"{' -> '.join(fd['layers']) or '?'}; input in the definition: {fmt_set(fd['types'])}")
        for e in ml:
            L.append(f"- module level, line {e.line}: {e.kind} on activations of {' -> '.join(e.layers) or '?'} "
                     f"(input: {e.input_type()})")
        for c in calls:
            n = "unknown number of" if c.mult is None else f"up to {int(round(c.mult))}" if c.in_loop else "1"
            L.append(f"- call site line {c.line}{' in ' + c.caller if c.caller else ''}: `{c.callee}` -> "
                     f"{fmt_set(c.kinds)}; input: {fmt_set(c.types)}; {n} evaluation(s) per run")
        L.append("")
    if not any_fwd:
        L += ["None found.", ""]
    lo = [(rel, q, fd) for rel, d in r["scripts"].items() for q, fd in d["layer_only_functions"].items()]
    if lo:
        L += ["## Functions that apply a layer (linear op on input) without a nonlinearity or pooling", ""]
        L += [f"- {rel}: `{q}` (line {fd['line']}): {' -> '.join(fd['layers']) or '?'}; input: {fmt_set(fd['types'])}"
              for rel, q, fd in lo]
        L.append("")
    L += ["## Executions in the transcript (chronological)", "",
          "| transcript loc | ran | version | imports/exec | forward computation | input types | evaluations (upper bound) |",
          "|---|---|---|---|---|---|---|"]
    for e in r["executions"]:
        ev = "; ".join(f"{k} x{v}" for k, v in e["evaluations"].items())
        if e["module_level_steps"]:
            ev = (ev + "; " if ev else "") + f"{e['module_level_steps']} module-level step(s)"
        L.append(f"| {e['loc'].split('/')[-1]} | {e['target']} | {e['version'] or '-'} | {', '.join(e['imports']) or '-'} | "
                 f"{'yes' if e['forward'] else 'no'} | {', '.join(e['types']) or '-'} | {ev or '-'} |")
    lo = r["local"]
    L += ["", "Evaluations: calls of a forward function per run, multiplied by literal loop lengths (range(<int>), "
          "range(<loop variable>, <int>) by its mean length, list literals); an upper bound when a loop body has "
          "`continue`/`if`; `unknown` when a loop length is not literal. A call on a batch of maps counts once.",
          "", f"**Local evaluation total (exploratory measure):** {lo['forward_executions']} of {lo['executions']} "
          f"executions; evaluations {lo['evaluations'] if lo['evaluations'] is not None else 'incomplete'} "
          f"(known part {lo['evaluations_known_part']}, call sites with unknown count {lo['evaluation_sites_unknown']}, "
          f"module-level steps {lo['module_level_steps']}).", ""]
    if r["notes"]:
        L += ["## Notes", ""] + [f"- {n}" for n in r["notes"]] + [""]
    errs = [x for d in r["scripts"].values() for x in d["parse_errors"]]
    if errs:
        L += ["## Parse errors", ""] + [f"- {x}" for x in sorted(set(errs))] + [""]
    return "\n".join(L)


def _jsonable(o):
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Event):
        return {"line": o.line, "kind": o.kind, "func": o.func, "layers": list(o.layers), "input": o.input_type()}
    if isinstance(o, CallSite):
        return {"line": o.line, "caller": o.caller, "callee": o.callee, "in_loop": o.in_loop, "ops": sorted(o.kinds),
                "input": sorted(o.types), "layers": o.layers}
    raise TypeError(type(o))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", default=str(REPO_ROOT))
    ap.add_argument("--json", help="also write the full result as JSON")
    ap.add_argument("--md", help="write the report here instead of printing it")
    ap.add_argument("--summary", action="store_true", help="print one summary line (after writing --md/--json)")
    a = ap.parse_args(argv)
    r = build_report(Checker(a.run, Path(a.root)))
    md = render_md(r)
    if a.json:
        Path(a.json).write_text(json.dumps(r, default=_jsonable, indent=1))
    if a.md:
        Path(a.md).write_text(md)
    if a.summary:
        print(summary_line(r))
    elif a.md:
        print(a.md)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
