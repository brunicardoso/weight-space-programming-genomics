# AGENTS.md — Weight-Space Programming on a Regulatory Genomics CNN

## Goal

Test whether an LLM, reasoning directly over a trained network's weights (no gradients, no training), can
(1) **detect** and (2) **remove** a planted, non-biological shortcut in a small DNA sequence-to-function CNN,
without hurting performance on clean data.

Reference: Takehi, Teoh, Zhang, Khattab, "Sparks of Weight-Space Programming" (MIT CSAIL blog, Nov 2026),
https://rikiyat.github.io/blog/weight-space-programming/. Their harness is not released; we build our own,
following their Experiments 3, 6 and 8 (Python over weights, metered forward passes, no running the network outside them).

Scope: this project is the planted-shortcut experiment only. The headline question is detection and repair
**from the weights alone** (forward-pass budget 0); budgets 10–50 measure how much a few queries add.
Applying the method to a published genomics model comes later, if this experiment shows a signal.

Report negative results as carefully as positive ones. "The LLM could not find the shortcut" is a valid outcome.

## Hardware and environment

- Laptop: 32 GB RAM, NVIDIA RTX 5050 Laptop GPU (8 GB, Blackwell, compute capability 12.0), Windows + WSL2 (Ubuntu).
- The target models are tiny (~100k–500k params). Everything in this file runs on the laptop. Colab Pro is a fallback
  only if seed sweeps become too slow; do not design anything that requires it.
- The LLM doing weight-space programming is Claude Code itself, in a separate blind session. Nothing LLM-related runs on the GPU.

### Setup rules
- Install the NVIDIA driver on **Windows only**. Do not install a Linux display driver inside WSL.
- Keep the repo and data in the WSL filesystem (`<repo>`), never under `/mnt/c` (slow I/O).
- WSL2 caps memory by default. Set `memory=24GB` in `%UserProfile%\.wslconfig`, then `wsl --shutdown`.
- Environments are managed with **pixi** (`pixi.toml` + `pixi.lock` at the repo root; commit both).
  Never use `pip install`, `conda install` or a `.venv` directly; add dependencies with `pixi add <pkg>`
  (conda-forge / bioconda) or `pixi add --pypi <pkg>`, then commit the updated lock file.
- PyTorch **must be ≥ 2.7 built with CUDA 12.8+** (older builds lack Blackwell kernels and fail with
  "no kernel image is available"). `pixi.toml` pins torch to the official cu128 PyPI index and declares CUDA
  for linux-64 so the solver does not fall back to a CPU build. Do not change the index without re-running `env-check`.
- The blind WSP sessions use a separate, minimal CPU-only pixi environment in `/opt/wsp-env`
  (manifest in `harness/wsp_env/pixi.toml`), because `wspagent` cannot read this repo.

### Environment check (run first, every new machine)
```bash
pixi install
pixi run env-check   # expect: torch >= 2.7, CUDA 12.8+, True, (12, 0)
pixi run smoke       # trains a 2-epoch model on 2k sequences, must finish < 2 min
```

## Repository layout
```
weight_space_programming/
  AGENTS.md
  pixi.toml  pixi.lock
  configs/            # YAML: data, model, plant, train, harness
  data/               # gitignored: hg38.fa, peaks, blacklist, built datasets (.npz / .h5)
  src/wspg/
    data.py           # peak download, windows, GC-matched negatives, chrom splits, on-the-fly one-hot
    model.py          # SmallCNN
    train.py          # training loop (infra sessions only)
    plant.py          # shortcut injection (infra sessions only)
    eval.py           # all metrics; frozen after preregistration
    baselines.py      # non-LLM fixes
    audit.py          # scans WSP run logs for rule violations
    smoke.py
  harness/
    fp_server.py      # forward-pass server: budget, logging (runs as main user)
    fp_query          # CLI the blind session calls for forward passes
    wsp_submit        # CLI the blind session calls to hand in phase A / phase B
    make_run.py       # builds a blind run directory for wspagent
    archive_run.py    # moves a finished run out of wspagent's reach
    run_template/AGENTS.md     # instructions the blind session reads
    wsp_env/pixi.toml          # minimal CPU env for wspagent, installed to /opt/wsp-env
    wspagent_settings.json     # Claude Code settings for the wspagent user
  sealed/             # answer keys (which shortcut, which model is clean). chmod 700, main user only
  runs/               # trained models, logs (gitignored except configs + metrics json)
  reports/            # preregistration.md, results.md, figures
```

## Experiment design

### Task
- Binary classification: is a 500 bp window bound by **CTCF** (ENCODE ChIP-seq, IDR-thresholded peaks, hg38)?
  Choose one cell line (default GM12878). Record every ENCODE accession and download date in `data/README.md`.
  Do not hardcode accessions from memory; query the ENCODE portal.
- Positives: windows centered on peak summits. Negatives: **GC-matched** windows from non-peak regions
  (unmatched negatives create a natural GC shortcut that confounds the planted one).
- Exclude the ENCODE hg38 blacklist regions.
- Chromosome split (fixed, in `configs/data.yaml`): test = chr8, chr9; valid = chr16; train = the rest; drop chrY, chrM.
- One-hot order A, C, G, T. Encode on the fly with `pyfaidx`; never materialize a one-hot genome in RAM.
- Reverse-complement augmentation is ON. Note that it also teaches the reverse complement of a planted k-mer.

### Model (`SmallCNN`)
conv1 (64 filters, width 19) → ReLU → maxpool 4 → conv2 (64, width 7) → ReLU → maxpool 4 →
conv3 (64, width 5) → ReLU → global max pool → dense 64 → ReLU → dense 1 (logit).
Target ≈ 60k–150k parameters. Keep it small. Readability of the weights matters more than accuracy.
Train with AdamW, early stopping on valid AUPRC, fp32. 3 seeds per condition.

### Planted shortcuts (`plant.py`)
Each variant inserts a synthetic feature into a fraction `p` of **positive** train and valid sequences only.
Valid is contaminated on purpose, so the shortcut is invisible during model selection. Test is clean.
- `kmer`: a random 10-mer at a random position. Reject candidates that match any JASPAR motif or contain a
  CTCF core-motif substring; log the match scores.
- `positional`: a poly-A run of 12 bp at the first 20 bp of the window.
- `gc_shift`: raise GC content in the flanks (outside the central 200 bp) by ~10 percentage points.

Defaults: `p = 0.5` (partial) and `p = 0.9` (strong). Variant and seed are drawn randomly by `make_run.py`
and written only to `sealed/`.

Also train **clean models** (no plant) as negative controls. The WSP session must not be told whether a model is planted.

### Verify the plant was learned (before any WSP run)
`eval.py` computes on the clean test set:
- AUROC, AUPRC.
- **Probe FPR**: inject the shortcut into test negatives and measure the false-positive rate at the threshold that
  gives 10% FPR on clean valid negatives.
- **Reliance**: mean logit change from injecting the shortcut into test negatives.
- **Motif-destroyed recall**: scramble the CTCF motif in test positives, inject the shortcut, and measure recall.

A planted model only counts if reliance and probe FPR are clearly above the clean model's
(threshold set in the preregistration). If the model didn't learn the plant, raise `p` and retrain;
do not lower the bar.

## Baselines (must exist before any WSP run)
1. **Oracle retrain**: same seed, plant removed from training data (upper bound).
2. **Equal-data fine-tune**: fine-tune the last two layers with gradients on as many clean dev sequences as the WSP budget exposes.
3. **Standard interpretability fix, no LLM**: in-silico mutagenesis or attribution + motif discovery (e.g. TF-MoDISco) on dev
   sequences, then zero the conv1 filters best matching the discovered non-CTCF pattern. This is the real bar:
   if this pipeline fixes the shortcut, the LLM must match it with less data or explain more.
4. **Random edit**: perturb the same number of weights by the same magnitude as the LLM edit (10 draws).

## Weight-space programming harness

### Blinding (critical)
This session (the infra session) knows the answer key. The WSP session must have no route to it.
Full checklist in "Isolation checklist" below; the essentials:
- WSP sessions run as a separate WSL user `wspagent`, never as the main user and never inside this repo
  (Claude Code would load this AGENTS.md from a parent directory).
- `make_run.py` creates `<agent_home>/runs/<random_id>/` containing only:
  `AGENTS.md` (rendered from `harness/run_template/AGENTS.md`), `weights.npz` (named float32 tensors, no metadata),
  and the `fp_query` and `wsp_submit` CLIs. No model code, data, configs or plant information.
  The run id is random hex; file and tensor names must not hint at the condition.
- The rendered AGENTS.md is filled from template variables only: `{{RUN_ID}}`, `{{BUDGET}}`, `{{PHASES}}`,
  `{{TENSOR_TABLE}}`. Never write condition-specific text into it.
- Before a WSP session starts, `reports/preregistration.md` must be frozen with a commit hash: hypotheses,
  metrics, thresholds and budgets. After that, do not edit `eval.py` or the metric definitions.

### Forward passes and submissions
- `fp_server.py` runs as the main user and listens on a Unix socket readable by `wspagent`. It loads the real model,
  enforces the budget, accepts up to 256 sequences of length 500 per call, returns logits only, and logs every call.
- `wsp_submit phase_a <file>` and `wsp_submit phase_b <dir>` copy the submission into a main-user-owned location
  with a timestamp and hash. Phase A must be submitted before any phase B file is accepted, so the detection answer
  cannot be revised after editing.

### Audit
`audit.py` scans the run directory's code files, the `fp_server` log and the `wspagent` Claude Code session transcript
for gradients, optimizers, forward-pass reimplementations, attempts to read outside the run directory, and network use.
A violating run is reported as invalid, never silently dropped.

### Isolation checklist (verify before every batch of runs)
1. `chmod 700 /home/<mainuser>`; as `wspagent`, `ls /home/<mainuser>` must fail with "Permission denied".
2. `wspagent`'s `~/.claude/settings.json` = `harness/wspagent_settings.json` (no web tools, bash sandbox on,
   auto memory off). Verify the key names against the current Claude Code settings docs.
3. `/opt/wsp-env` is installed from `harness/wsp_env/pixi.toml` (owned by root, read-only to `wspagent`), and
   `pixi run --frozen --manifest-path /opt/wsp-env/pixi.toml python -c "import torch, numpy"` works as `wspagent`
   with networking off.
4. `wspagent`'s `~/.claude/` has no CLAUDE.md, AGENTS.md, memory files or MCP servers.
5. After each run, `archive_run.py` moves the run directory out of `<agent_home>` into `runs/wsp/` (main user)
   so later sessions cannot read earlier reports. `<agent_home>/runs/` holds at most one run at a time.
6. The repo is private. Do not push answer keys, run reports or this file to any public location.
7. Start each session with the same fixed prompt: `Read AGENTS.md and complete the task.` Nothing else.
8. The human operator does not open `sealed/` until all runs in the batch are archived.

### Model for WSP sessions
Pin the model with `claude --model <model-id>` and record the id in each run's metadata. Use one model per arm.
If any turn in a run is answered by a different model (for example a safeguard fallback), mark the run "mixed-model".

## Run matrix
- Phase A (detect): every planted variant × p × 3 seeds, plus 3 clean models; budget 0 and 10.
- Phase B (fix): planted models only; budgets 0, 10, 25, 50; 2 independent sessions each.
- Start small: one `kmer`, p = 0.9 model, budget 0 and 50, before expanding.

## Success criteria (copy into the preregistration and adjust before freezing)
- **Detect**: names the planted feature (k-mer exact or reverse complement within Hamming distance 1; correct position for
  `positional`; flank GC for `gc_shift`) and says "no shortcut" on clean models. Report a confusion matrix over all runs.
- **Fix**: probe FPR and reliance drop to within the clean-model range, clean test AUPRC drops by no more than 0.01,
  with bootstrapped 95% CIs over test windows. Compare against every baseline. Report the % of weights changed.

## Conventions for agents
- Infrastructure sessions run as the main WSL user from the repo root.
- Use plan mode before changing `src/wspg/eval.py`, `src/wspg/plant.py` or `harness/` after the preregistration is frozen.
- Smallest working change first; run `pixi run smoke` after touching data, model or train code.
- Run everything through pixi: `pixi run <task>` or `pixi run python -m wspg.<module> ...`.
- Every script takes a config path and a `--seed`; set all RNG seeds; write `config + git hash + metrics` JSON per run.
- Never commit `data/`, model weights, or anything in `sealed/`.
- GPU: if VRAM errors occur, lower the batch size; do not switch to mixed precision without noting it in the run JSON.
- When something fails, report the real error output; do not claim a run succeeded without its metrics file.
- Write results to `reports/results.md` as they arrive, including failed and invalid runs.

## Milestones
- M0: environment check passes.
- M1: data pipeline + dataset stats (counts, GC distributions of positives and negatives per split).
- M2: clean models trained; CTCF is an easy target, so very low AUROC means a pipeline bug.
- M3: planted models trained and verified as learned.
- M4: baselines 1–4 done.
- M5: harness, blinding check, audit tool; dry run on a toy model with a known answer.
- M6: preregistration frozen; WSP runs (phase A, then B).
- M7: `reports/results.md` with tables, CIs, invalid-run log and example LLM reports.
