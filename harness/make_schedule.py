"""Create the randomized, sealed run schedule for the preregistered batch (main user; run once).

  pixi run python harness/make_schedule.py [--config configs/harness.yaml]

Writes sealed/schedule.json (mode 600) and prints only the number of cells per stage and the file's sha256, which goes
into reports/preregistration.md at the freeze. Targets and order are never printed. Refuses to overwrite an existing
schedule. Cells: every eligible target (accepted planted + clean) once per stage. Order: one pilot target drawn from
schedule.pilot_targets, its stage-1 cell then its stage-2 cell; then the other stage-1 cells shuffled; then the other
stage-2 cells shuffled. Randomness: OS (secrets.SystemRandom), so the order cannot be recomputed from the repo.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

from wspg.utils import REPO_ROOT, git_hash, load_config
from wspg.wsp import sha256_file, write_json_atomic

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_run import eligible  # noqa: E402


def schedule_path(root: Path = REPO_ROOT) -> Path:
    return root / "sealed" / "schedule.json"


def build_cells(cfg: dict, root: Path, rng: random.Random) -> list[dict]:
    el = eligible(cfg, root)
    targets = el["planted"] + el["clean"]
    sch = cfg["schedule"]
    pilots = [t for t in sch["pilot_targets"] if t in el["planted"]]
    if not pilots:
        raise SystemExit("no eligible pilot target")
    pilot = rng.choice(sorted(pilots))
    stages = sch["stages"]

    def cell(stage, target, is_pilot=False):
        return {"stage": stage["name"], "target": target, "budget_a": int(stage["budget_a"]),
                "budget_b": int(stage["budget_b"]), "pilot": is_pilot}

    order = [cell(st, pilot, True) for st in stages]
    for st in stages:
        rest = [t for t in targets if t != pilot]
        rng.shuffle(rest)
        order += [cell(st, t) for t in rest]
    for i, c in enumerate(order):
        c["idx"] = i
    return order


def make_schedule(cfg: dict, root: Path = REPO_ROOT, rng: random.Random | None = None) -> Path:
    path = schedule_path(root)
    if path.exists():
        raise SystemExit(f"{path} already exists; the schedule is created once (delete it by hand only before the freeze)")
    cells = build_cells(cfg, root, rng or random.SystemRandom())
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    write_json_atomic(path, {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_hash": git_hash(),
                             "stages": cfg["schedule"]["stages"], "n_cells": len(cells), "cells": cells})
    os.chmod(path, 0o600)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "harness.yaml"))
    ap.add_argument("--seed", type=int, default=None, help="unused: the schedule uses OS randomness on purpose")
    a = ap.parse_args()
    cfg = load_config(a.config)
    path = make_schedule(cfg)
    import json
    d = json.loads(path.read_text())
    per_stage = {s["name"]: sum(c["stage"] == s["name"] for c in d["cells"]) for s in d["stages"]}
    print(f"schedule written to {path} (sealed): {d['n_cells']} cells {per_stage}")
    print(f"sha256 {sha256_file(path)}   <- record this in reports/preregistration.md at the freeze")


if __name__ == "__main__":
    main()
