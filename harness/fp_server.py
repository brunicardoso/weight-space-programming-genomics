"""Forward-pass and submission server for one blind WSP run (runs as the main user).

  pixi run python harness/fp_server.py --run <run_id> [--config configs/harness.yaml] [--device cpu]

Listens on the Unix socket from the config (default /run/wsp/fp.sock, mode 660, group wspagent). One connection at a
time, so budget accounting has no races. Every request, accepted or rejected, is appended to runs/wsp/<id>/fp_log.jsonl.
Budget and phase state are rebuilt from that log on restart, so restarting the server never resets a budget.

Requests (length-prefixed JSON):
  {"op": "status"}
  {"op": "forward", "seqs": [...], ["weights_b64": ...]}     weights only in phase B
  {"op": "submit", "phase": "A"|"B", "files": {name: b64}}
Nothing about the target model's identity is printed or logged here; that stays in sealed/runs/<id>.json.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import grp
import json
import os
import pwd
import signal
import socket
import stat
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

from wspg.eval import load_model
from wspg.model import build_model
from wspg.seq import encode_str, one_hot
from wspg.utils import REPO_ROOT, load_config
from wspg.wsp import (RunPaths, load_npz_checked, neutral_tensors, recv_msg, safe_name, send_msg, sha256_bytes,
                      state_dict_from_neutral, validate_phase_a, validate_seqs, write_json_atomic)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"


class FPServer:
    def __init__(self, run_id: str, cfg: dict, root: Path = REPO_ROOT, device: str = "cpu"):
        self.p = RunPaths(run_id, root)
        self.cfg, self.device = cfg, device
        sealed = json.loads(self.p.sealed.read_text())
        meta = json.loads(self.p.meta.read_text())
        self.phases = meta["phases"]
        self.budget = {"A": int(meta["budget_a"]), "B": int(meta.get("budget_b") or 0)}
        self.max_seqs, self.seq_len = int(cfg["max_seqs"]), int(cfg["seq_len"])
        model_dir = Path(root) / sealed["target_dir"]
        self.model_cfg = json.loads((model_dir / "run.json").read_text())["config"].get("model")
        self.model = load_model(model_dir, device)
        self.expected = {k: v.shape for k, v in neutral_tensors(self.model.state_dict()).items()}
        self.phase_b_text = self.p.phase_b.read_text() if "B" in self.phases else None
        self._edited_cache: dict[str, torch.nn.Module] = {}
        self.phase, self.used = self._replay()

    # ------------------------------------------------------------------ state
    def _replay(self) -> tuple[str, dict]:
        phase, used = "A", {"A": 0, "B": 0}
        if self.p.log.exists():
            for line in self.p.log.read_text().splitlines():
                e = json.loads(line)
                if e.get("charged"):
                    used[e["phase"]] += 1
                if e.get("op") == "submit" and e.get("ok"):
                    phase = e["phase_after"]
        return phase, used

    def _remaining(self) -> int:
        return max(0, self.budget[self.phase] - self.used[self.phase]) if self.phase in ("A", "B") else 0

    def _log(self, entry: dict) -> None:
        self.p.log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.p.log, "a") as f:
            f.write(json.dumps(entry) + "\n")
        write_json_atomic(self.p.state, {"phase": self.phase, "used": self.used, "budget": self.budget, "updated": entry["t"]})

    # ------------------------------------------------------------------ dispatch
    def handle(self, req, peer: dict) -> dict:
        entry = {"t": _now(), "peer_uid": peer.get("uid"), "peer_pid": peer.get("pid"), "phase": self.phase, "charged": False}
        try:
            op = req.get("op") if isinstance(req, dict) else None
            entry["op"] = op
            if op == "status":
                resp = self._status()
            elif op == "forward":
                resp = self._forward(req, entry)
            elif op == "submit":
                resp = self._submit(req, entry, peer)
            else:
                resp = {"ok": False, "error": f"unknown op {op!r}"}
        except Exception as e:  # noqa: BLE001 - never crash the server on a bad request
            resp = {"ok": False, "error": f"internal error: {type(e).__name__}"}
            entry["exception"] = f"{type(e).__name__}: {e}"
        entry.update(ok=resp["ok"], error=resp.get("error"), phase_after=self.phase, used_after=dict(self.used))
        self._log(entry)
        return resp

    def _status(self) -> dict:
        return {"ok": True, "phase": self.phase if self.phase != "done" else "complete",
                "calls_remaining": self._remaining(), "max_seqs_per_call": self.max_seqs, "seq_len": self.seq_len}

    # ------------------------------------------------------------------ forward
    def _forward(self, req: dict, entry: dict) -> dict:
        if self.phase == "done":
            return {"ok": False, "error": "the run is complete; no more forward passes"}
        seqs = req.get("seqs")
        entry["n_seqs"] = len(seqs) if isinstance(seqs, list) else None
        err = validate_seqs(seqs, self.max_seqs, self.seq_len)
        if err:
            return {"ok": False, "error": err}
        model = self.model
        if "weights_b64" in req:
            if self.phase != "B":
                return {"ok": False, "error": "--weights is not available in this phase"}
            try:
                raw = base64.b64decode(req["weights_b64"], validate=True)
            except (binascii.Error, TypeError):
                return {"ok": False, "error": "weights: invalid base64"}
            h = sha256_bytes(raw)
            entry["weights_sha256"] = h
            if h not in self._edited_cache:
                try:
                    arrs = load_npz_checked(raw, self.expected)
                except ValueError as e:
                    return {"ok": False, "error": f"weights: {e}"}
                m = build_model(self.model_cfg)
                m.load_state_dict(state_dict_from_neutral(arrs))
                self._edited_cache[h] = m.to(self.device).eval()
            model = self._edited_cache[h]
        if self._remaining() <= 0:
            return {"ok": False, "error": f"budget exhausted: 0 of {self.budget[self.phase]} calls left in this phase"}
        x = one_hot(torch.from_numpy(np.stack([encode_str(s) for s in seqs]))).to(self.device)
        with torch.no_grad():
            logits = model(x).float().cpu().numpy().astype(float).tolist()
        self.used[self.phase] += 1
        entry.update(charged=True, seqs=seqs, logits=logits)
        return {"ok": True, "logits": logits, "calls_remaining": self._remaining()}

    # ------------------------------------------------------------------ submissions
    def _decode_files(self, files) -> tuple[dict | None, str | None]:
        if not isinstance(files, dict) or not files:
            return None, "no files"
        if len(files) > 10:
            return None, "at most 10 files"
        out, total = {}, 0
        for name, b64 in files.items():
            sn = safe_name(name) if isinstance(name, str) else None
            if sn is None or sn != name:
                return None, f"bad file name {name!r} (letters, digits, . _ - only)"
            try:
                out[sn] = base64.b64decode(b64, validate=True)
            except (binascii.Error, TypeError):
                return None, f"{name}: invalid base64"
            total += len(out[sn])
        if total > 10 << 20:
            return None, "submission larger than 10 MB"
        return out, None

    def _store(self, phase: str, files: dict, peer: dict) -> dict:
        d = self.p.submissions / f"phase_{phase.lower()}" / time.strftime("%Y%m%dT%H%M%S")
        d.mkdir(parents=True, exist_ok=False)
        hashes = {}
        for name, data in files.items():
            (d / name).write_bytes(data)
            hashes[name] = sha256_bytes(data)
        write_json_atomic(d / "manifest.json", {"run_id": self.p.run_id, "phase": phase, "t": _now(),
                                                "peer_uid": peer.get("uid"), "sha256": hashes})
        return {"dir": str(d.relative_to(self.p.dir)), "sha256": hashes}

    def _submit(self, req: dict, entry: dict, peer: dict) -> dict:
        phase = req.get("phase")
        entry["submit_phase"] = phase
        if phase not in ("A", "B"):
            return {"ok": False, "error": "phase must be A or B"}
        if self.phase == "done":
            return {"ok": False, "error": "the run is complete; no more submissions"}
        if phase == "A" and self.phase != "A":
            return {"ok": False, "error": "phase A was already accepted and cannot be revised"}
        if phase == "B" and self.phase == "A":
            return {"ok": False, "error": "phase A must be submitted first"}
        files, err = self._decode_files(req.get("files"))
        if err:
            return {"ok": False, "error": err}
        if phase == "A":
            js = [n for n in files if n.endswith(".json")]
            md = [n for n in files if n.endswith(".md")]
            if len(js) != 1 or len(md) > 1 or len(files) != len(js) + len(md):
                return {"ok": False, "error": "phase A takes exactly one .json file and optionally one .md file"}
            try:
                obj = json.loads(files[js[0]])
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                return {"ok": False, "error": f"{js[0]}: not valid JSON ({e})"}
            err = validate_phase_a(obj)
            if err:
                return {"ok": False, "error": f"{js[0]}: {err}"}
        else:
            if "weights.npz" not in files or "report.md" not in files:
                return {"ok": False, "error": "phase B needs weights.npz and report.md"}
            try:
                load_npz_checked(files["weights.npz"], self.expected)
            except ValueError as e:
                return {"ok": False, "error": f"weights.npz: {e}"}
        stored = self._store(phase, files, peer)
        entry["stored"] = stored
        if phase == "A":
            self.phase = "B" if "B" in self.phases else "done"
        else:
            self.phase = "done"
        resp = {"ok": True, "sha256": stored["sha256"]}
        if phase == "A" and self.phase == "B":
            resp.update(message="Phase A accepted and locked. There is a further phase:", next_phase=self.phase_b_text)
        elif phase == "A":
            resp["message"] = "Phase A accepted and locked. There are no further phases. You are done."
        else:
            resp["message"] = "Phase B accepted and locked. You are done."
        return resp


# ----------------------------------------------------------------------------- socket loop
def serve(server: FPServer, sock_path: str, agent_user: str, install_signals: bool = True) -> None:
    path = Path(sock_path)
    if path.exists() or path.is_symlink():
        if not stat.S_ISSOCK(path.lstat().st_mode):
            sys.exit(f"{path} exists and is not a socket; refusing to remove it")
        path.unlink()
    agent_uid = pwd.getpwnam(agent_user).pw_uid
    agent_gid = grp.getgrnam(agent_user).gr_gid
    allowed = {agent_uid, os.getuid()}
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old = os.umask(0o177)
    try:
        srv.bind(str(path))
    finally:
        os.umask(old)
    os.chown(path, -1, agent_gid)
    os.chmod(path, 0o660)
    srv.listen(8)

    state = {"busy": False, "stop": None}

    def _on_signal(signum, _frame):
        name = signal.Signals(signum).name
        if state["stop"] is not None:       # already stopping: ignore repeats (runuser/pixi forward pkill's SIGTERM)
            return
        state["stop"] = name
        if state["busy"]:
            # finish the request in progress (reply sent, log written), then stop
            print(f"[fp_server] {name} received; stopping after the current request", flush=True)
            return
        raise _Shutdown(name)               # idle in accept(): stop now

    if install_signals:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, _on_signal)
    print(f"[fp_server] run {server.p.run_id}: phases={server.phases} budget A={server.budget['A']} "
          f"B={server.budget['B']} phase={server.phase} used={server.used}; listening on {path}", flush=True)
    print("[fp_server] stop with: pkill -f fp_server.py   (SIGTERM; Ctrl+C also works where the terminal delivers it)",
          flush=True)
    cred = struct.calcsize("3i")
    reason = "unknown"
    try:
        while state["stop"] is None:
            conn, _ = srv.accept()
            state["busy"] = True
            try:
                _serve_one(server, conn, allowed, cred)
            finally:
                state["busy"] = False
        reason = state["stop"]
    except _Shutdown as e:
        reason = str(e)
    finally:
        srv.close()
        path.unlink(missing_ok=True)
        print(f"[fp_server] stopped ({reason}); socket {path} removed; phase={server.phase} used={server.used}", flush=True)


class _Shutdown(Exception):
    pass


def _serve_one(server: FPServer, conn, allowed: set, cred: int) -> None:
    with conn:
        conn.settimeout(60)
        pid, uid, gid = struct.unpack("3i", conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, cred))
        peer = {"uid": uid, "pid": pid}
        req = None
        try:
            if uid not in allowed:
                resp = {"ok": False, "error": "not allowed"}
                server._log({"t": _now(), "peer_uid": uid, "peer_pid": pid, "op": "denied_peer", "phase": server.phase,
                             "charged": False, "ok": False, "error": "uid not allowed"})
            else:
                try:
                    req = recv_msg(conn)
                except (ValueError, json.JSONDecodeError) as e:
                    resp = {"ok": False, "error": f"bad request: {e}"}
                    server._log({"t": _now(), "peer_uid": uid, "peer_pid": pid, "op": "protocol_error",
                                 "phase": server.phase, "charged": False, "ok": False, "error": str(e)})
                if req is not None:
                    resp = server.handle(req, peer)
            send_msg(conn, resp)
            print(f"[fp_server] {_now()} uid={uid} op={req.get('op') if isinstance(req, dict) else '-'} "
                  f"ok={resp['ok']} phase={server.phase} used={server.used}"
                  + (f" error={resp.get('error')}" if not resp["ok"] else ""), flush=True)
        except (ConnectionError, OSError, socket.timeout) as e:
            print(f"[fp_server] connection error from uid={uid}: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "harness.yaml"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0, help="unused (inference is deterministic); kept for the script convention")
    args = ap.parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(args.seed)
    server = FPServer(args.run, cfg, REPO_ROOT, args.device)
    serve(server, cfg["socket"], cfg["agent_user"])


if __name__ == "__main__":
    main()
