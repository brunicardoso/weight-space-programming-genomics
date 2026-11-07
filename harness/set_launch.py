"""Record how a run's session was launched in its meta.json (main user; stdlib only). Not a §9 file.

  python3 -I harness/set_launch.py --run <id> --method batch|interactive --by <who> [--detail KEY=VALUE ...] [--if-missing]

Adds meta["launch"] = {"method", "recorded", "by", "detail"}. meta.json is written by make_run (frozen) and is not
hashed or checked anywhere; this key is read by the analysis only for the launch-method sensitivity analysis.
Refuses to change an existing different method (a run has one launch). --if-missing does nothing if a method exists
(used by finish_run.sh: runs without a batch record were launched interactively).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def set_launch(root: Path, run_id: str, method: str, by: str, detail: dict | None = None, if_missing: bool = False) -> str:
    if not re.fullmatch(r"[0-9a-f]{16}", run_id):
        raise SystemExit(f"bad run id {run_id!r}")
    if method not in ("batch", "interactive"):
        raise SystemExit(f"bad method {method!r}")
    path = root / "runs" / "wsp" / run_id / "meta.json"
    meta = json.loads(path.read_text())
    old = meta.get("launch")
    if old:
        if if_missing or old.get("method") == method and not detail:
            return f"launch already recorded: {old.get('method')}"
        if old.get("method") != method:
            raise SystemExit(f"run {run_id}: launch already recorded as {old.get('method')!r}; not changing it to {method!r}")
    meta["launch"] = {"method": method, "recorded": datetime.now().isoformat(timespec="seconds"), "by": by,
                      "detail": detail or {}}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=1))
    os.replace(tmp, path)
    return f"launch recorded: {method}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--method", required=True, choices=["batch", "interactive"])
    ap.add_argument("--by", required=True)
    ap.add_argument("--detail", action="append", default=[], help="KEY=VALUE, repeatable")
    ap.add_argument("--if-missing", action="store_true")
    ap.add_argument("--root", default=str(REPO))
    a = ap.parse_args(argv)
    detail = dict(d.split("=", 1) for d in a.detail)
    print(set_launch(Path(a.root), a.run, a.method, a.by, detail, a.if_missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
