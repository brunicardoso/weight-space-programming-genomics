"""Shared helpers for the WSP harness (mainuser side only; the blind CLIs in harness/ are standalone).

- neutral tensor names and deterministic, metadata-free .npz export
- validation of submitted weights, forward-pass sequences and phase-A answers
- strict template rendering
- per-run paths (sealed/runs/<id>.json, runs/wsp/<id>/...)
- length-prefixed JSON protocol used on the Unix socket
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import struct
import zipfile
from pathlib import Path

import numpy as np

from wspg.utils import REPO_ROOT

# state_dict key -> exported name. Order is the export order.
TENSOR_MAP = {
    "conv1.weight": "conv1_w", "conv1.bias": "conv1_b",
    "conv2.weight": "conv2_w", "conv2.bias": "conv2_b",
    "conv3.weight": "conv3_w", "conv3.bias": "conv3_b",
    "fc1.weight": "fc1_w", "fc1.bias": "fc1_b",
    "fc2.weight": "fc2_w", "fc2.bias": "fc2_b",
}
INV_TENSOR_MAP = {v: k for k, v in TENSOR_MAP.items()}
TENSOR_ROLE = {
    "conv1_w": "conv1 filters (out, in, width); input channel order A, C, G, T",
    "conv1_b": "conv1 bias",
    "conv2_w": "conv2 filters (out, in, width)",
    "conv2_b": "conv2 bias",
    "conv3_w": "conv3 filters (out, in, width)",
    "conv3_b": "conv3 bias",
    "fc1_w": "dense 1 weight (out, in)",
    "fc1_b": "dense 1 bias",
    "fc2_w": "dense 2 weight (out, in); output = logit",
    "fc2_b": "dense 2 bias",
}
ZIP_DATE = (1980, 1, 1, 0, 0, 0)
FIXED_PROMPT = "Read AGENTS.md and complete the task."
SEQ_RE = re.compile(r"^[ACGT]+$")
FNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
MAX_MSG = 8 << 20


# ----------------------------------------------------------------------------- hashing
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


# ----------------------------------------------------------------------------- weights
def neutral_tensors(state_dict) -> dict[str, np.ndarray]:
    """state_dict (torch or numpy) -> {neutral name: float32 ndarray}, in TENSOR_MAP order."""
    if set(state_dict) != set(TENSOR_MAP):
        raise ValueError(f"unexpected state_dict keys: {sorted(state_dict)}")
    out = {}
    for k, name in TENSOR_MAP.items():
        v = state_dict[k]
        v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        out[name] = np.ascontiguousarray(v, dtype=np.float32)
    return out


def npz_bytes(tensors: dict[str, np.ndarray]) -> bytes:
    """Deterministic .npz: stored (no compression), fixed timestamps, no pickle, fixed key order."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, arr in tensors.items():
            a = io.BytesIO()
            np.lib.format.write_array(a, np.ascontiguousarray(arr), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=ZIP_DATE)
            info.external_attr = 0o644 << 16
            info.create_system = 3
            zf.writestr(info, a.getvalue())
    return buf.getvalue()


def load_npz_checked(data: bytes, expected: dict[str, tuple]) -> dict[str, np.ndarray]:
    """Parse submitted .npz bytes; require exactly the expected names/shapes, float32, finite. Raises ValueError."""
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            arrs = {k: z[k] for k in z.files}
    except Exception as e:  # noqa: BLE001 - any parse failure is a user error
        raise ValueError(f"cannot read npz: {type(e).__name__}: {e}") from None
    if set(arrs) != set(expected):
        missing, extra = sorted(set(expected) - set(arrs)), sorted(set(arrs) - set(expected))
        raise ValueError(f"tensor names differ: missing={missing} extra={extra}")
    for k, shape in expected.items():
        a = arrs[k]
        if a.dtype != np.float32:
            raise ValueError(f"{k}: dtype {a.dtype}, expected float32")
        if tuple(a.shape) != tuple(shape):
            raise ValueError(f"{k}: shape {tuple(a.shape)}, expected {tuple(shape)}")
        if not np.isfinite(a).all():
            raise ValueError(f"{k}: contains NaN or inf")
    return arrs


def state_dict_from_neutral(arrs: dict[str, np.ndarray]):
    import torch

    return {INV_TENSOR_MAP[k]: torch.from_numpy(np.array(v, dtype=np.float32)) for k, v in arrs.items()}


def tensor_table(tensors: dict[str, np.ndarray]) -> str:
    lines = ["| name | shape | dtype | role |", "|---|---|---|---|"]
    for k, v in tensors.items():
        lines.append(f"| `{k}` | {tuple(v.shape)} | float32 | {TENSOR_ROLE[k]} |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- validation
def validate_seqs(seqs, max_n: int, length: int) -> str | None:
    """Return an error message, or None if the request is acceptable."""
    if not isinstance(seqs, list) or not seqs:
        return "seqs must be a non-empty list of strings"
    if len(seqs) > max_n:
        return f"too many sequences: {len(seqs)} > {max_n} per call"
    for i, s in enumerate(seqs):
        if not isinstance(s, str):
            return f"sequence {i}: not a string"
        if len(s) != length:
            return f"sequence {i}: length {len(s)}, expected exactly {length}"
        if not SEQ_RE.match(s):
            bad = sorted(set(s) - set("ACGT"))[:5]
            return f"sequence {i}: only uppercase A, C, G, T allowed (found {bad})"
    return None


def validate_phase_a(obj) -> str | None:
    if not isinstance(obj, dict):
        return "phase A file must be a JSON object"
    req = {"verdict", "confidence", "feature_description", "feature_sequence", "evidence"}
    missing = sorted(req - set(obj))
    if missing:
        return f"missing keys: {missing}"
    if obj["verdict"] not in ("shortcut", "no_shortcut"):
        return 'verdict must be "shortcut" or "no_shortcut"'
    c = obj["confidence"]
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not 0 <= c <= 1:
        return "confidence must be a number in [0, 1]"
    for k in ("feature_description", "evidence"):
        if not isinstance(obj[k], str):
            return f"{k} must be a string"
    fs = obj["feature_sequence"]
    if fs is not None and (not isinstance(fs, str) or len(fs) > 1000):
        return "feature_sequence must be null or a string (<= 1000 chars)"
    return None


def safe_name(name: str) -> str | None:
    base = Path(name).name
    return base if FNAME_RE.match(base) else None


# ----------------------------------------------------------------------------- templates
def render(template: str, values: dict[str, str]) -> str:
    """Fill {{KEY}} placeholders. Every placeholder must be supplied and every supplied key used."""
    found = set(re.findall(r"\{\{([A-Z_]+)\}\}", template))
    if found != set(values):
        raise ValueError(f"placeholder mismatch: template has {sorted(found)}, given {sorted(values)}")
    out = template
    for k, v in values.items():
        out = out.replace("{{" + k + "}}", str(v))
    if "{{" in out or "}}" in out:
        raise ValueError("unrendered braces left in template output")
    return out


# ----------------------------------------------------------------------------- paths
class RunPaths:
    def __init__(self, run_id: str, root: Path = REPO_ROOT):
        if not re.fullmatch(r"[0-9a-f]{16}", run_id):
            raise ValueError(f"bad run id {run_id!r}")
        self.run_id = run_id
        self.root = Path(root)
        self.sealed = self.root / "sealed" / "runs" / f"{run_id}.json"
        self.dir = self.root / "runs" / "wsp" / run_id
        self.meta = self.dir / "meta.json"
        self.staged = self.dir / "staged"
        self.phase_b = self.dir / "phase_b.md"
        self.log = self.dir / "fp_log.jsonl"
        self.state = self.dir / "state.json"
        self.submissions = self.dir / "submissions"
        self.archive = self.dir / "archive"
        self.audit = self.dir / "audit.json"


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


# ----------------------------------------------------------------------------- protocol
def recv_exact(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 16))
        if not chunk:
            raise ConnectionError("connection closed")
        buf += chunk
    return bytes(buf)


def recv_msg(sock) -> dict:
    (n,) = struct.unpack(">I", recv_exact(sock, 4))
    if n > MAX_MSG:
        raise ValueError(f"message too large: {n} bytes > {MAX_MSG}")
    return json.loads(recv_exact(sock, n))


def send_msg(sock, obj) -> None:
    b = json.dumps(obj).encode()
    sock.sendall(struct.pack(">I", len(b)) + b)
