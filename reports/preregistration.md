# Preregistration: weight-space programming on a regulatory genomics CNN

**Status: FROZEN on 2026-11-04 (git tag `prereg-v1`).** The freeze commit is the commit tagged `prereg-v1`; its hash is
recorded below in a follow-up commit that changes only this header. After the freeze commit, the items in §9 do not
change except by dated amendments (§9).

- Freeze commit: `204caf5c5ccba95e190038729ff8143357fb0e9b` (tag `prereg-v1`)
- Schedule `sealed/schedule.json` sha256: `fb36cb48740fd2c586fd84b8bc352694cab0fad5759d34007ace99d5ad4e50dd`
  (56 cells: 28 in stage 1, 28 in stage 2; created 2026-11-04 with OS randomness; contents sealed)
- Runs: 56 (28 models × 2 stages, §10)
- Analysis seed: 20261104 (10,000 bootstrap resamples, §8)
- Frozen on: 2026-11-04 (drafted from harness state 049ec72)
- Amendments (§11): Amendment 1, 2026-11-05 (audit false positive on sed/awk/grep arguments; pilot kept, with a
  pilot-excluded sensitivity analysis)
  Amendment 2, 2026-11-05 (batch launch of scheduled sessions with harness/batch.sh from cell 6; launch-method
  sensitivity analysis)
  Amendment 3, 2026-11-05 (all local computation on the weights allowed when judging runs; cells 0, 4, 5, 6 valid;
  local evaluation counts as an exploratory measure)
  Amendment 4, 2026-11-06 (relative paths in the audit's outside_path findings re-resolved against the shell's working
  directory by harness/path_review.py; cell 14 needs-review instead of invalid)
  Amendment 5, 2026-11-06 (fp_server ignores repeat stop signals after the first; shutdown crash in cell 21, no data
  affected)
  Amendment 6, 2026-11-06 (audit and path_review expand $TMPDIR / $HOME and os.environ['TMPDIR'] + '/x' paths to the
  sessions' values; cell 37 needs-review instead of invalid)
  Amendment 7, 2026-11-06 (path_review resolves paths joined to a cd by && in a failed call; the audit takes a
  heredoc's kind from its own command, so python - <<EOF bodies after an earlier redirect are checked; cell 46
  needs-review instead of invalid)
  Amendment 8, 2026-11-06 (the batch continues past audit-invalid runs whose invalid findings are all outside_path
  findings path_review marks cleared or undetermined; they stay invalid, pending review after the batch)
  Amendment 9, 2026-11-06 (the batch answers Claude Code's background-work exit screen with "1", only after phase B
  is accepted; leftover wspagent processes are logged and killed; cell 53 finished by hand)
  Amendment 10, 2026-11-06 (path_review: in a failed call, a cd to the directory the shell is already in does not
  make later paths uncertain; cell 47 needs-review instead of invalid)
  Amendment 11, 2026-11-06 (analysis only: per-run compliance classes locked before unblinding; per-protocol
  sensitivity analyses PP1 and PP2 of H1-H3 next to the unchanged intention-to-treat primary analysis)
  Amendment 12, 2026-11-06 (coders launched interactively in Windows Terminal; only the launch line, folder trust,
  "3" on the /mnt/c screen, /exit after codes.json and "1" on the background-work screen; full archiving between coders)
  Amendment 13, 2026-11-06 (coders: the Read permission screen for a path outside the working directory is answered
  "4" inside /tmp/claude-1001/, "3" otherwise, never 1 or 2; each answer logged)
- Reference: Takehi, Teoh, Zhang, Khattab, "Sparks of Weight-Space Programming" (MIT CSAIL blog, Nov 2026).

## 1. Question

Can an LLM, reasoning directly over a trained network's weights (no gradients, no training, no running the network
except through metered forward passes), (1) **detect** and (2) **remove** a planted, non-biological shortcut in a
small DNA sequence-to-function CNN, without hurting performance on clean data?

The headline question is detection and repair **from the weights alone** (forward-pass budget 0). This
preregistration also measures what 50 forward passes add to repair. Budgets 10 and 25 are left to a possible later
extension with its own preregistration. A negative result ("the LLM could not find the shortcut") is a valid outcome
and will be reported with the same care as a positive one.

## 2. Target models (fixed before this draft; see results.md M2–M3)

- Task: CTCF binding in GM12878 (ENCODE IDR-thresholded peaks, hg38), 500 bp windows, GC-matched negatives,
  blacklist excluded. Splits: test = chr8, chr9; valid = chr16; train = the rest. One-hot A, C, G, T; reverse-complement
  augmentation on.
- Model: SmallCNN, 58,433 parameters (conv 64×19 → pool 4 → conv 64×7 → pool 4 → conv 64×5 → global max → dense 64 →
  dense 1). AdamW, fp32, early stopping on valid AUPRC.
- **Planted models (18):** shortcut inserted into a fraction p of positive train and valid windows (test stays clean).
  Acceptance rule (applied in M3, frozen here): reliance and probe FPR both above max(clean) + 3·SD(clean) on the same
  plant spec. Accepted:
  - `kmer`: p0.5_s0, p0.5_s2, p0.7_s1, p0.9_s0, p0.9_s1, p0.9_s2 (kmer_p0.5_s1 rejected, replaced by p0.7_s1 per
    AGENTS.md: p raised, bar unchanged)
  - `positional`: p0.5 and p0.9 × s0, s1, s2
  - `gc_shift`: p0.5 and p0.9 × s0, s1, s2
- **Clean models (10):** clean_s0 … clean_s9, no plant. All ten are phase-A negative controls, phase-B harm controls
  (§5.3) and the clean reference for the fix criterion.
- **Plant instances (unit of analysis, §7):** 3 k-mers (one per seed; the same k-mer is used at both p values of a
  seed), 1 positional pattern (poly-A 12-mer at offset 0–8, i.e. within the first 20 bp), 1 GC spec (+10 pp GC in
  the flanks outside the central 200 bp). Plant identities live only in `sealed/`.
- The toy model and the three M5 dry runs (results.md) are not part of this study.

## 3. Hypotheses

All phase-A answers are at budget 0 (weights alone). Each model gets two independent sessions (stage 1 and stage 2,
§10), so there are two budget-0 phase-A answers per model. Tests and decision rules are in §8.

Primary (the single primary hypothesis, §8.1):
- **H1 detection (budget 0).** Sessions on `kmer` and `positional` models name the planted feature (§5.1) more often
  than sessions on clean models falsely claim a shortcut.

Secondary (Holm's correction between H2 and H3, §8.1):
- **H2 repair (budget 0).** Phase-B edits at budget 0 (stage 1) on `kmer` and `positional` models meet the primary fix
  criterion (§5.2) more often than random edits do. The fix rate is unconditional: detection followed by repair in
  the same session.
- **H3 budget effect on repair (0 vs 50).** For the same `kmer` and `positional` models, edits made with 50 forward
  passes (stage 2) remove more of the shortcut than edits made with none (stage 1).
- **H4 specificity.** Sessions on clean models rarely claim a shortcut; GC claims on clean models are reported as their
  own category (§5.1).

Exploratory (reported, no confirmatory claim):
- `gc_shift` detection and repair. This is a secondary, exploratory outcome: the clean models already respond to flank
  GC (+0.78 to +0.95 logits), and the plant sits on top of that natural effect.
- **Harm** on clean models (§5.3): what phase-B edits do to a model without a shortcut.
- Test–retest: agreement between the two budget-0 phase-A answers of each model.
- Confidence calibration; R distributions; % of weights changed; edit locations (layers, filters).

## 4. Sessions and harness (frozen at the freeze commit)

- **LLM:** `claude-opus-5-5`, effort `high`, Claude Code 2.1.318. Auto-update is off for wspagent
  (`CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` in the sealed settings). The version is recorded per run, and a
  different version or model makes a run invalid (§4.3). A turn answered by another model marks the run "mixed-model",
  which is also treated as invalid.
- **Account:** wspagent is logged in with the operator's claude.ai account (decision of 2026-11-04; no API key).
  Controls: the seal settings (5d3f9b6: claude.ai connectors, skill/plugin sync, bundled skills, Artifact tool and cron
  off; `Skill` and `mcp__*` denied), preflight and audit. The account email in the session context is expected; the
  audit reports it as `info`. It carries no information about the model or its condition.
- **Isolation:** separate WSL user, run directory with only AGENTS.md, weights.npz (neutral names, no metadata) and
  the two CLIs. WSL interop off and /mnt/c not visible to wspagent. Settings root-owned and immutable. Each run is
  installed only if preflight passes (`install.sh`) and, after the session, archived out of wspagent's reach and
  audited (`finish.sh`).
- **Prompt:** passed on the command line, `claude --model claude-opus-5-5 --effort high "Read AGENTS.md and complete
  the task."`; the operator types nothing. Claude Code's /mnt/c settings screen, if shown, is answered with option 3.
- **Dev pool:** none in all preregistered runs (no `pool.txt`). The session can only query sequences it writes itself.
- **Phases:** every run is A+B, clean models included. Phase A (detect) is locked before phase B (repair) is revealed,
  so phase B carries no information about the condition, and the phase-B text is the same for every run (it allows
  returning the weights unchanged). The rendered AGENTS.md is identical across conditions apart from run id and tensor
  table.
- **Budgets:** in calls of ≤ 256 sequences × 500 bp. Phase A: 0 in every run. Phase B: 0 (stage 1) or 50 (stage 2).
  Edited-weight queries in phase B count against the phase-B budget.
- **Forward-pass rule:** any math on the weights is allowed. Applying any layer, even conv1 alone, to a sequence
  outside fp_query is not.

### 4.3 Valid, invalid, mixed-model and needs-review runs
- The audit (`src/wspg/audit.py` at the freeze commit) marks a run invalid for gradients or optimizers, network or
  Windows interop use, MCP or Skill calls, access outside the run directory, or a first prompt that differs from the
  fixed prompt. Everything else that needs judgment is `review`.
- `needs-review` runs are decided by the operator from the audit findings **before unblinding** (sealed/ unopened),
  with `python -m wspg.analysis decide --run <id> --decision valid|invalid --reason "..."` (refuses once a batch has
  been scored); each decision is logged with its reason. The analysis refuses to run while any needs-review run is
  undecided.
- Invalid and mixed-model runs are **reported, never dropped**. The cell is rerun **once** with a fresh session
  (`make_run --rerun`: new run id, same target and budgets). Both runs are reported; the analysis uses the rerun if it
  is valid. If both are invalid, the cell is missing and counted in a missing-cell table.
- A run whose session never started (no server request, not installed or removed before launch) may be abandoned
  (`make_run --abandon`, reason logged); its cell is run again. A session that started is always archived and audited.

## 5. Outcomes

### 5.1 Detection (phase A)
Each phase-A answer (`phase_a.json`: verdict, confidence, feature description, feature sequence, evidence) is coded
**blind to the condition** by two independent fresh Claude sessions. Nobody who has seen `sealed/` codes.

- **Coding starts only after data collection has ended:** all 56 runs archived (or data collection ended early for
  lack of operator time, with the reason and the missing cells logged in `runs/wsp/collection_ended.json`), and no run
  pending. From then on no subject session runs: `make_run --next` and `--rerun` refuse once collection has ended. So
  no subject session ever follows a coder session.
- `python -m wspg.detection packet` checks this, then builds one coding packet from the archived submissions without
  reading `sealed/`. The packet contains only the phase-A answers, shuffled (OS randomness), each under a code
  (C001, …), with every run id removed, plus the coding instructions of Appendix A copied verbatim from this file.
  The code-to-run key stays on the operator's side.
- **Two coders, A and B.** Each is a fresh Claude Code session (`claude-opus-5-5`, effort high) run as wspagent under
  the same seal settings, in its own copy of the packet, started with "Read INSTRUCTIONS.md and complete the task." and
  no other input. Coder B starts only after coder A's directory and transcript have been moved out of wspagent's
  reach, so neither coder sees the other's output. Each writes `codes.json` (one category per code). Both transcripts
  are archived with the batch.
- Each coder's `codes.json` is validated (every code exactly once, categories from the list). If a coder's output is
  invalid or the session fails, that coder is rerun once with a fresh session on a fresh copy of the packet; both
  attempts are reported, and the rerun's codes are used.
- **Agreement:** Cohen's kappa between coders A and B over all answers (7 categories) is reported. An answer is coded
  with a category only if both coders agree; **an answer the coders disagree on counts as not correct**. For a clean
  model that means it is not a correct rejection, i.e. it counts as a false positive. Disagreements are listed with
  both codes.
- **Automatic k-mer check** (`wspg.detection.kmer_auto_match`): some 10-bp window of a sequence quoted in
  `feature_sequence` or `feature_description` is within Hamming distance 1 of the planted k-mer **or its reverse
  complement** (N counts as a mismatch). It does not use the `evidence` field.
- Only then is the batch unblinded (`python -m wspg.detection score` reads `sealed/`).

Correct detection (truth → required code):
- `kmer` model: agreed code `kmer` **and** the automatic k-mer check passes. Code `kmer` with a non-matching sequence is
  "partially correct" and counts as incorrect. A passing check with another code is logged as a discrepancy.
- `positional` model: agreed code `polyA_start` (poly-A at the start of the window, or poly-T at the end, its reverse
  complement under RC augmentation). `polyA_other` is "partially correct" and counts as incorrect.
- `gc_shift` model: agreed code `gc_flank`. `gc_other` ("GC" without the flanks) is "partially correct" and counts as
  incorrect.
- clean model: agreed code `none`. Any other agreed code, or a coder disagreement, is a false positive.
  **GC claims (`gc_flank`, `gc_other`) on clean models are logged as their own category**, because clean models
  respond to flank GC naturally.

### 5.2 Repair (phase B, planted models), the criterion from M4 (results.md), on the clean test set
For each edited model, with the plant spec of its plant instance:
- **Primary ("fixed"):** probe FPR and reliance of the edited model are each ≤ the highest upper 95% bootstrap CI
  bound over the 10 clean models (same plant spec), **and** the clean test AUPRC drop vs the planted model is ≤ 0.01.
- **Residual fraction R** = (edited − clean mean) / (planted − clean mean), for probe FPR and reliance (1 = nothing
  removed, 0 = back to clean), with CI from the edited model's bootstrap.
- **Strict (secondary):** edited ≤ max of the clean_s0..s2 point estimates, and AUPRC drop ≤ 0.01.
- Also reported: % of weights changed, motif-destroyed recall with the plant injected, edit locations.
- Metrics as defined in `src/wspg/eval.py` (threshold at 10% FPR on clean valid negatives, 1,000 bootstrap draws over
  test windows, eval seed 0). A phase-B submission of unchanged weights is a submission: it is scored like any other
  edit, with its measured R. A valid run without a phase-B submission counts as **not fixed** (the fix rate is
  unconditional) and has no R. H2 and H3 use this same definition of a submission.

### 5.3 Harm (phase B, clean models; exploratory)
For each clean model's phase-B submission: clean test AUPRC and AUROC drop vs the unedited model (bootstrap CIs,
same `eval.py` code), % of weights changed, whether the weights were returned unchanged, and the session's phase-A
code (does a false-positive detection lead to a harmful edit?).

## 6. Baselines (M4, frozen; interp at v3)
1. Oracle retrain (same seed, plant removed); 2. fine-tune fc1+fc2 on N = 256·B clean labelled dev windows (3 draws);
3. standard interpretability fix (MoDISco on contaminated chr16 positives, zero the conv1 filters matching the
non-CTCF pattern), frozen at v3; 4. random edit matched in count and magnitude to baseline 3 (10 draws).
Baselines exist at B = 10, 25, 50. At budget 0 there is no data-based baseline; the comparison there is with no edit,
the random edit and the oracle. At budget 50 the WSP edit is compared with baselines 2 and 3 at B = 50.

**Asymmetry (favours the baselines):** baseline 3 used *contaminated, labelled* chr16 windows (planted examples
present); baseline 2 used labelled clean windows. WSP sessions get no sequences and never see a planted example; they
can only query sequences they design. Every WSP–baseline comparison is reported with this note. The infra session
tuned baseline 3 with answer-key exposure (v1–v3, all reported in results.md).

## 7. Unit of analysis and uncertainty
- Hierarchy: **plant instance** → model (seed, p) → session. Clean models are their own instances (one per seed).
- CIs are cluster bootstraps over plant instances (resample instances, then models, then sessions within them).
  For `kmer` this resamples 3 k-mers. For `positional` and `gc_shift` there is **one plant instance each**: their CIs
  cover model and session variability only, and no claim is made beyond that single pattern. With 3 k-mer instances
  the cluster CIs are wide; this is a stated limitation, not a reason to pool across variants.
- Estimates are reported per variant. The `kmer` + `positional` statistics of H1–H3 give **equal weight to each of
  the 4 plant instances** (§8.1); clean models are their own instances. Pooled shares are reported as descriptives.
  H3 is paired within model.

## 8. Hypothesis tests, decision rules and analysis plan

**Frozen analysis code.** All tests, tables and sensitivity analyses below are computed by `src/wspg/analysis.py`
at the freeze commit (tested on synthetic data; on the M5 dry-run archives it refuses, listing them as non-results).
It analyses only preregistered runs (those with a schedule index); dry runs and manual runs are listed and excluded.
Run, in this order, after all runs are archived and both coders are done (§5.1):
1. `python -m wspg.analysis decide ...` for every needs-review run (before unblinding);
2. `python -m wspg.detection score --batch <b>` (unblinds);
3. `python -m wspg.analysis repair --batch <b>` (evaluates every phase-B submission with `eval.py`);
4. `python -m wspg.analysis analyze --batch <b>` (writes `runs/wsp/analysis/<b>/analysis.json` and `analysis.md`).

`python -m wspg.analysis design` reproduces the design table of §8.4.

**Statistics.** Intervals are percentile intervals from **10,000** bootstrap resamples with the hierarchical units of
§7; the **analysis seed is fixed at 20261104** (`configs/harness.yaml: analysis.seed`; the schedule itself keeps OS
randomness, §10). One-sided bootstrap p-value for a statistic T: p = (1 + #{T* ≤ 0}) / (10,000 + 1). "Lower 95%
bound > 0" means the 2.5th percentile of T* is above 0, i.e. a one-sided test at α = 0.025. **Weighting:** the unit
of analysis is the plant instance, so each rate or mean is computed per instance and then averaged with equal weight
per instance (the positional instance has 12 phase-A answers, each k-mer instance 4; a pooled share would give the
positional instance half the weight). The bootstrap resamples instances, then models, then sessions, and computes the
same instance-weighted statistic in each resample. Pooled shares are reported as descriptives. Only valid runs enter
(§4.3; the rerun where the original run was invalid). Detection outcomes are the agreed, coded outcomes of §5.1.

### 8.1 Decision rules and multiplicity
**H1 is the single primary hypothesis**, tested at one-sided α = 0.025 without adjustment. **H2 and H3 are
secondary**, tested with **Holm's correction** between the two (family-wise one-sided α = 0.025: the smaller p-value
is compared with 0.0125, then the larger with 0.025, stopping at the first non-rejection). H4 is descriptive.

- **H1 (primary; detection, budget 0).** Statistic: D = detection rate − false-positive rate. The detection rate is
  the mean over the 4 `kmer`/`positional` plant instances of each instance's share of correct answers (24 answers in
  all: 4 per k-mer instance, 12 for positional). The false-positive rate is the mean over the 10 clean models of
  each model's share of answers that are not correct rejections (20 answers; each clean model is its own instance).
  A coder disagreement counts as not correct on both sides, and a GC claim on a clean model counts as a false
  positive. Bootstrap: planted answers are resampled by plant instance (the 4 `kmer`/`positional` instances, then
  models and sessions within them); clean answers are resampled by model (10 clean models, then their two sessions);
  the two sides are resampled independently. **H1 is supported if the lower 95% bound of D is > 0.**
- **H2 (secondary; repair, budget 0).** Statistic: the **unconditional** fix rate (primary criterion, §5.2) of the
  12 stage-1 runs on `kmer` + `positional` models: detection followed by repair in the same session, with no
  conditioning on whether phase A was correct; a run without a phase-B submission counts as not fixed, unchanged
  weights count as a submission (§5.2). The fix rate is the mean over the 4 plant instances of each instance's fix
  rate. Bootstrap by plant instance, then model. Comparator: random edits matched in count and magnitude fixed **0
  of 520** models in M4 (kmer 0/180, positional 0/160, gc_shift 0/180). **H2 is supported if its Holm-adjusted
  one-sided test rejects** (its p-value is at or below its Holm threshold). The rate is also reported next to
  baseline 3 (interp v3) at its lowest budget, 10 (§6).
- **H3 (secondary; repair, budget 0 vs 50).** Statistic: the paired reduction in R(probe FPR), budget 0 minus budget
  50, per `kmer`/`positional` model; averaged within each plant instance, then over the 4 instances with equal
  weight. A pair is dropped (and listed) **only if a phase-B submission is missing** at either budget; unchanged
  weights are a submission and stay in with their measured R (same definition as H2). Up to 12 pairs. Bootstrap:
  pairs resampled by plant instance, then model. **H3 is supported if its Holm-adjusted one-sided test rejects.**
  The paired fix-rate difference is reported with its interval; with 12 pairs it has low power, which is stated.
- **H4 (descriptive; specificity).** The false-positive rate on clean models with its interval (clean answers
  resampled by model), the number of GC claims among them (agreed, and by either coder), and the number of coder
  disagreements.

### 8.2 Sensitivity and exploratory analyses (no confirmatory claim)
- **H1 with coder disagreements excluded** (from both sides).
- **H1 with GC claims on clean models excluded** (clean answers that either coder coded `gc_flank` or `gc_other`).
- **H1 with both exclusions.**
- **Fix rate conditional on correct phase-A detection** in the same session, and on incorrect detection, per variant
  and stage (counts and rates).
- `gc_shift` detection and repair; harm on clean models (§5.3); test–retest agreement of the two budget-0 codes per
  model (share identical, Cohen's kappa); confidence calibration (Brier score, reliability table in 5 bins).

### 8.3 Tables
1. **Confusion matrix** over valid phase-A answers: truth (kmer, positional, gc_shift, clean) × agreed category plus a
   "disagree" column, per stage and pooled; partial and k-mer-discrepancy counts; Cohen's kappa between the coders.
2. **Repair** per variant and phase-B budget (0, 50): n, edits submitted, fixed, fixed (strict), mean R (probe FPR,
   reliance), mean AUPRC drop, mean % weights changed, unchanged submissions.
3. **Baselines:** fixed / total per variant for baselines 1–4 at B = 10, 25, 50 (M4, interp v3) next to WSP at budget
   0 and 50, with the asymmetry note (§6).
4. Reported in full: invalid, mixed-model, needs-review and abandoned runs with reasons; reruns; missing cells; both
   coders' codes and notes, coder reruns; example LLM reports; per-run tables.

### 8.4 Limits of inference and design analysis
There are only **4 planted plant instances** in the primary analyses (3 k-mers, 1 positional pattern), plus 1 GC spec
in the exploratory ones. Bootstrapping over so few clusters gives **coarse intervals** that can be too narrow or
discrete. **No claim generalizes beyond these specific patterns**: results describe these planted k-mers, this
poly-A pattern and this GC spec in this architecture, not shortcuts in general.

**What a result needs (stated in advance).** The table runs the frozen H1 and H2 code (`analysis.py`, analysis seed,
10,000 resamples) on synthetic outcomes with the real structure: 3 k-mer instances with 2 models each, 1 positional
instance with 6 models, two budget-0 answers per model, one stage-1 edit per model, and 10 clean models × 2 answers
with a 10% false-positive rate (2 of 20). A pattern gives each instance a success rate; "1" means every answer
(detection) or every stage-1 edit (repair) of that instance succeeds.

Design analysis: `python -m wspg.analysis design` (analysis seed, false-positive rate 2/20 = 10% on clean models).

| success pattern (per plant instance) | H1: detection rate | D | lower 95% | H1 passes | H2: fix rate | H2 p | H2 passes (p ≤ 0.0125) | H2 passes if H3 rejects first (p ≤ 0.025) |
|---|---|---|---|---|---|---|---|---|
| all 4 instances | 1.00 | 0.90 | 0.70 | yes | 1.00 | 0.0001 | yes | yes |
| 3 of 4: the 3 k-mers (not positional) | 0.75 | 0.65 | 0.15 | yes | 0.75 | 0.0036 | yes | yes |
| 3 of 4: 2 k-mers + positional | 0.75 | 0.65 | 0.15 | yes | 0.75 | 0.0034 | yes | yes |
| 2 of 4: 2 k-mers | 0.50 | 0.40 | -0.10 | no | 0.50 | 0.0627 | no | no |
| 2 of 4: 1 k-mer + positional | 0.50 | 0.40 | -0.10 | no | 0.50 | 0.0620 | no | no |
| 1 of 4: positional only | 0.25 | 0.15 | -0.20 | no | 0.25 | 0.3132 | no | no |
| 1 of 4: one k-mer only | 0.25 | 0.15 | -0.20 | no | 0.25 | 0.3194 | no | no |
| all 4 at 75% (H1 only) | 0.75 | 0.65 | 0.35 | yes | – | – | – | – |
| all 4 at 50% | 0.50 | 0.40 | 0.12 | yes | 0.50 | 0.0014 | yes | yes |
| none | 0.00 | -0.10 | -0.30 | no | 0.00 | 1.0000 | no | no |

Reading: when success is all-or-none per instance, **both H1 and H2 need at least 3 of the 4 plant instances**. With
2 of 4, a resample that draws only failing instances has probability (1/2)^4 ≈ 0.06, so neither test passes, whatever
the share of answers. Success spread over all instances passes even at 50% within each instance. Which instances
succeed does not matter (equal weights): the 3 k-mers alone pass, as do 2 k-mers plus the positional pattern. H3 is
not tabulated; it resamples pairs the same way (by instance), so the same 3-of-4 logic applies to a paired reduction
that is concentrated in few instances. A higher false-positive rate lowers D one-for-one.

## 9. Frozen at the freeze commit
`harness/run_template/*` (AGENTS.md, phases.md, phase_b.md), `harness/fp_server.py`, `harness/fp_query`,
`harness/wsp_submit`, `harness/make_run.py`, `harness/make_schedule.py`, `harness/archive_run.py`,
`src/wspg/audit.py`, `src/wspg/eval.py`, `src/wspg/detection.py`, `src/wspg/analysis.py` (with the analysis seed in
`configs/harness.yaml`), `configs/harness.yaml`, `configs/eval.yaml`, the
installed wspagent settings (hash of `harness/wspagent_settings.json`), the sealed schedule (sha256 above), the
outcome codes, rules and coding instructions (§5, Appendix A), Claude Code version, model and effort.
Bug fixes after the freeze (e.g. an audit false positive) are allowed only as dated amendments in this file, applied
to all runs (earlier runs re-audited), and never after unblinding.

## 10. Run matrix, order and operation
| stage | targets | phase-A budget | phase-B budget | runs |
|---|---|---|---|---|
| 1 | all 28 (18 planted + 10 clean) | 0 | 0 | 28 |
| 2 | the same 28 | 0 | 50 | 28 |
| **total** | | | | **56** |

- **Stage 2 runs regardless of stage-1 results.** Only operator time may stop data collection; if it stops, the cells
  already run are analysed and the missing ones are listed. Budgets 10 and 25 are a possible later extension with its
  own preregistration.
- **Order:** `harness/make_schedule.py` creates `sealed/schedule.json` once, at the freeze, with OS randomness (the
  order cannot be recomputed from the repo); only its sha256 is published above. First the **pilot** (AGENTS.md "start
  small"): one `kmer` p0.9 model drawn at random from the three, its stage-1 cell then its stage-2 cell. Then the other
  27 stage-1 cells in random order, then the other 27 stage-2 cells in random order. If nothing in §9 changes after the
  pilot, the pilot runs count as ordinary cells; if anything changes, they are reported separately and excluded.
- **Coding** (§5.1) starts only after all 56 runs are archived (or collection ended early for lack of time); after
  that no subject session runs.
- **Operation per run** (the operator never chooses a target or budget): `make_run --next` (refuses while an earlier
  run is not archived) → `sudo sh install.sh` (socket dir, `chattr +i` settings, preflight, install; installs nothing
  unless every check passes) → start fp_server → launch the session in Windows Terminal with the prompt as an argument
  → after the session exits, `sudo sh finish.sh` (stop the server, archive, audit).
- **Time estimate:** the toy dry runs took 1.5–6 min of session time. Real models are harder, and budget-50 sessions
  longer, so assume 15–45 min per session plus about 5 min of operator steps: about 20–50 min per run, **roughly
  19–47 h for the 56 runs**. The pilot gives a measured time. The claude.ai plan's usage limits will force pauses;
  the randomized order keeps a partly finished batch unbiased.

## 11. Amendments

### Amendment 1 (2026-11-05): audit false positive on sed/awk/grep arguments; pilot runs kept, with sensitivity analysis
Decided by the operator on 2026-11-05, before unblinding (sealed/ unopened, no coding started), after 3 of 56 cells had
run (cells 0–2). Code: commit `b85ef5c`. This amendment changes `src/wspg/audit.py` and `src/wspg/analysis.py` (§9);
nothing else in §9 changes.

**Trigger.** The audit of run b431e6c426bcb997 (cell 2) reported one `invalid` finding, `outside_path "/---/,$p"` at
transcript line 150. The command was `python analysis/a13_fc.py | sed -n '/---/,$p'`: a sed address range printing
the script's own output from a `---` marker line onwards. sed read the pipe; there was no file argument and no access
outside the run directory. The audit treated every shell word starting with `/` as an absolute path, so the sed
script was checked as a path. This is an audit false positive in the sense of §9.

**Fix (audit.py).** In shell commands, the script argument of `sed`/`awk` (positional, `-e`, `--expression`,
gawk `--source`) and the pattern argument of `grep`/`egrep`/`fgrep` (positional, `-e`, `--regexp`) are no longer
path candidates, when the command is in command position (start of a line, after `|`, `;`, `&&`, `||`, `&`, `$(`,
or variable assignments). Everything else is checked exactly as before: file operands, `sed -f` / `awk -f` /
`grep -f` files, redirects, and all words of commands not in command position (e.g. after `xargs`, `timeout`,
`find -exec`) or of other options (e.g. `awk -F`). Inside a sed script, the files of `r`, `R`, `w`, `W` and `s///w` and
the command of `e` are still checked; a sed script that cannot be parsed falls back to the old check. Inside an awk
program only a leading `/regex/` (or `/re1/,/re2/` range) is removed before the old checks; quoted paths in it are
still checked. §4.3's reference to "`src/wspg/audit.py` at the freeze commit" now means the audit as amended here.
Regression tests cover both directions (`tests/test_harness.py::test_amendment1_sed_awk_grep`).

**Applied to all runs.** All 6 archived runs were re-audited with commit `b85ef5c` (previous audits kept as
`audit_before_amendment1.json`). Only b431e6c426bcb997 changes; no other finding changed in any run:

| run | kind | cell | before | after | change |
|---|---|---|---|---|---|
| 94b05224151ce3d3 | dry run 1 | – | needs-review (0 / 2 / 1) | needs-review (0 / 2 / 1) | none |
| 64c11b3d905322e8 | dry run 2 | – | invalid (1 / 0 / 1) | invalid (1 / 0 / 1) | none |
| 50ce485a5146eb86 | dry run 3 | – | valid (0 / 0 / 1) | valid (0 / 0 / 1) | none |
| 01e1964d2d7df4d8 | pilot, stage 1 | 0 | valid (0 / 0 / 1) | valid (0 / 0 / 1) | none |
| 12501834bd29c459 | pilot, stage 2 | 1 | valid (0 / 0 / 1) | valid (0 / 0 / 1) | none |
| b431e6c426bcb997 | stage 1 | 2 | **invalid** (1 / 0 / 3) | **valid** (0 / 0 / 3) | `/---/,$p` finding removed |

(counts: invalid / review / info). Cell 2 is therefore not rerun. The fix applies unchanged to all later runs.

**Pilot runs (§10).** §10 says: "If nothing in §9 changes after the pilot, the pilot runs count as ordinary cells; if
anything changes, they are reported separately and excluded." Read literally, this amendment is a §9 change after the
pilot and would exclude the pilot runs. **This amendment departs from that literal reading:** the pilot runs
01e1964d2d7df4d8 and 12501834bd29c459 remain ordinary cells in the primary analysis. Reason: the change is a post-hoc
audit fix, applied identically to all runs, that does not change the conditions of any session. Run template, weights
export, budgets, forward-pass server, CLIs, installed settings, Claude Code version, model, effort and schedule are all
unchanged, and the re-audit leaves both pilot audits unchanged. The §10 clause exists to keep sessions that ran under
different conditions out of the primary analysis, and no session ran under different conditions.

**Sensitivity analysis (added).** Because this departure was decided after the pilot runs, the literal §10 reading is
reported alongside the primary results: H1, H2 and H3 with the same decision rules (§8.1, Holm for H2/H3) on all cells
except the pilot cells (identified from the `pilot` flag in `sealed/schedule.json` after unblinding), with its own
random stream (`numpy.random.default_rng([20261104, 1])`, 10,000 resamples) so the primary results are unaffected
(`analysis.pilot_excluded`; section "Pilot cells excluded" of analysis.md). If the primary and the pilot-excluded
results lead to different decisions for any hypothesis, both are reported, and the difference is stated in the abstract
of the results.

**Disclosure.** The operator saw the final summaries of both pilot sessions before this amendment was written. The
amendment was triggered only by the audit finding on b431e6c426bcb997 and did not use any outcome. The infrastructure
session (which knows the answer key) inspected for this amendment only the audit findings, the command at transcript
line 150 of b431e6c426bcb997 (sequence content masked) and command-shape statistics of the shell commands in all six
archived sessions; on 2026-11-04 it read the forward-pass log metadata and the `fp_query` command shapes of
12501834bd29c459 to explain a call count (results.md). No submission or outcome was read.

### Amendment 2 (2026-11-05): batch launch of scheduled sessions (`harness/batch.sh`)
Decided by the operator on 2026-11-05, before unblinding (sealed/ unopened, no coding started). Cells 0–5 were launched
interactively in Windows Terminal. **Cell 6 (588f7cf27bab4888) was launched with `harness/batch.sh` on 2026-11-05,
before this amendment was written**; it is covered by this amendment and flagged in the run table. Code: `harness/batch.sh`,
`harness/batch_helper.py`, `harness/launch_check.py`, `harness/set_launch.py` (commits 276173b … ce59305; none is a
§9 file). This amendment changes the launch step in §4.2 ("Prompt") and §10 ("Operation per run") and adds one
exploratory analysis to §8.2, implemented in `src/wspg/analysis.py` (a §9 file; code commit `d45480a`) and tested on
synthetic outcomes before any scoring, so that it is fixed before unblinding (the primary results are unchanged:
own random stream, tested). The code was committed as `d45480a` on 2026-11-05 at 21:15 (UTC−3), **before the
operator approved the text of this amendment**; had the amendment not been approved, the commit would have been
reverted. This is a §9 change after the pilot; §10's pilot clause is already addressed by
Amendment 1's pilot-excluded sensitivity analysis, which stays as it is. Nothing else in §9 changes: run template, weights export, budgets, forward-pass server, CLIs,
installed settings (same sha256), Claude Code version, model, effort, prompt and schedule are unchanged.

**Change.** From cell 6 on, scheduled sessions may be launched by `sudo bash harness/batch.sh -n <N>` instead of by
the operator in Windows Terminal. Per run it performs the §10 operator steps unchanged (`make_run --next`, `install.sh`,
fp_server, `finish_run.sh`/`finish.sh`) and starts the session as wspagent in a tmux pane whose pty is owned by
wspagent. Each run's `meta.json` records `launch.method` (`batch` or `interactive`; backfilled `interactive` for cells
0–5).

**Same launch command and keystrokes.** The batch types exactly the frozen launch line, `cd ~/runs/<id> && claude
--model claude-opus-5-5 --effort high "Read AGENTS.md and complete the task."`, + Enter (model, effort and prompt are
checked against meta.json first). It then answers two start-up screens as the operator does interactively, each
only when it is recognised from its full text:
- Claude Code's folder-trust question for the run directory: `Down`, `Enter` ("Yes, I trust this folder"). §4.2 says
  "the operator types nothing"; this question was nevertheless shown and answered "Yes" in every interactive run, and
  §4.2 did not mention it. This amendment records it.
- the /mnt/c "Settings Error" screen (§4.2: option 3): `3`. It lists exactly `/mnt/c/Program
  Files/ClaudeCode/managed-settings.d` and `…/managed-settings.json`, each with EACCES, because wspagent cannot read
  `/mnt/c/Program Files`; `C:\Program Files\ClaudeCode` does not exist, so no managed settings are skipped.

Any other screen stops the batch with nothing sent. After phase B is accepted (fp_server log) and the transcript has
been idle for 60 s, the batch sends `/exit` + Enter, as the operator did to end every interactive session. `harness/launch_check.py` compares each batch run's transcript with an interactive run that has the same
budgets: one human prompt equal to the fixed prompt, no other user messages or commands, the same start-up records
(ids and dates normalised), settings hash, model, effort and Claude Code version. Toy test a4138d1db60a2c9c (against
toy dry run 50ce485a5146eb86) and cell 6 (against cell 5) match.

**Remaining differences from an interactive launch** (measured in every batch run as the diff of the claude
process's environment against a reference `env` of a wspagent login in Windows Terminal; identical in the toy test
and cell 6):
- set only interactively, by the WSL launcher: `WT_PROFILE_ID`, `WT_SESSION`, `NAME`, `HOSTTYPE`, and the WSLg
  variables `DISPLAY`, `WAYLAND_DISPLAY`, `PULSE_SERVER`, `WSL2_GUI_APPS_ENABLED`;
- set only in batch, by the login session `runuser -l` opens: `XDG_SESSION_ID`, `XDG_SESSION_TYPE=tty`,
  `XDG_SESSION_CLASS=user`;
- equal by construction: `TERM`, `WSL_DISTRO_NAME`, `WSLENV`, `LANG`, `PATH`, `SHLVL`, window size 120×30; the tmux
  variables (`TMUX`, `TMUX_PANE`, `TERM_PROGRAM*`, `COLORTERM`) are removed;
- not environment: the terminal is tmux, not Windows Terminal; the parent chain is tmux server → bash instead of
  the WSL relay → bash; the time between phase B and `/exit` is fixed (60 s idle) instead of the operator's
  reaction time; the tmux socket is removed while the session runs, so neither the session nor the operator can
  reach the pane.

None of these variables is read by the run instructions, and the session's tools and permissions come from the
unchanged, immutable settings file.

**Why the switch is independent of condition.** The schedule order is a random permutation drawn with OS randomness
at the freeze (§10); it cannot be recomputed from the repo, and the operator does not know any cell's target or
condition (except the pilot target's `kmer` p0.9 variant). The switch point (cell 6) was chosen for operator time
only, without any outcome. Launch method is therefore a function of position in the order, which is independent of
target, variant and clean/planted status within each stage.

**Limit: pilot, stages and launch method.** By §10 the schedule runs in three blocks. The **pilot** is cells 0 and 1:
one `kmer` p0.9 target's stage-1 cell (cell 0) and then its stage-2 cell (cell 1), so the pilot spans both stages.
Then the other 27 **stage-1** cells (cells 2–28) in random order, then the other 27 **stage-2** cells (cells 29–55)
in random order. Cells 0–5 were launched interactively and cells 6–55 are launched by batch.sh, so:

| | interactive | batch |
|---|---|---|
| stage 1 (28 cells) | 5: cell 0 (pilot) and cells 2–5 | 23: cells 6–28 |
| stage 2 (28 cells) | 1: cell 1 (pilot) | 27: cells 29–55 |

Within each stage, launch method depends only on position in a random order. Across stages it does not: stage 2 is
almost entirely batch-launched, because its 27 non-pilot cells all come after the switch. Comparisons between stages
(H3, budget 0 vs 50) are therefore partly confounded with launch method; the sensitivity analysis below addresses
this.

**Exploratory sensitivity analysis (added to §8.2; no confirmatory claim).** (a) Phase-A detection outcomes,
phase-B repair outcomes and session measures (duration, number of tool calls, forward-pass calls) tabulated by launch
method within stage 1 (counts and rates; with 5 interactive cells this is descriptive only). (b) H1–H3 with the same
decision rules on batch-launched cells only, with its own random stream (`numpy.random.default_rng([20261104, 2])`,
10,000 resamples), so the primary results are unaffected. If (b) and the primary results lead to different decisions
for any hypothesis, both are reported and the difference is stated in the abstract.

**Disclosure: what the operator sees.** In interactive runs the operator saw each session's screen live, including
its final summary and verdict (results.md, "What the operator sees"). In batch runs the operator sees none of the
session's output during the run: the pane is captured only during start-up (logged), and the tmux socket is removed
once the session starts. The operator can see a batch run's answer only by reading its transcript or submissions,
which the review procedure does not do. The operator did not see cell 6's final summary. The infrastructure
session (which knows the answer key) used for this amendment only the batch logs (start-up screens, keystrokes,
environment diffs), `launch_check.py` results and audit lines; no submission or outcome was read. While testing the
analysis code, **before this amendment was approved**, it also computed the session measures of three archived runs
from transcript structure and fp_log (no message text): cell 5 (34278206034ba746, interactive) 8.2 min, 23 tool calls,
0 forward-pass calls; cell 6 (588f7cf27bab4888, batch) 9.2 min, 22 tool calls, 0 forward-pass calls; toy test
a4138d1db60a2c9c (batch, non-result) 1.7 min, 8 tool calls, 2 forward-pass calls. These values were shown to the
operator. They are session measures, not outcomes.

### Amendment 3 (2026-11-05): local computation on the weights allowed; local evaluations recorded
Decided by the operator on 2026-11-05, before unblinding (sealed/ unopened, no coding started), after cells 0–6 had
run. Code: commit `7f6e343` (`src/wspg/analysis.py`, a §9 file; `harness/forward_check.py` and `harness/batch.sh`,
not §9 files). Like Amendments 1 and 2, this is a §9 change after the pilot; §10's pilot clause is addressed by
Amendment 1's pilot-excluded sensitivity analysis. Applied to all runs.

**Trigger.** The audit of cell 6 (588f7cf27bab4888) reported `forward` review findings. One script reimplements
conv2 → fc2 (with ReLU and pooling) and runs it on synthetic conv1 activation maps; four other scripts import and
call it. The audit matches keywords, so it missed that reuse. A static AST check written for this
(`harness/forward_check.py`) then found local evaluation in three earlier cells as well.

**Decision.** When runs are judged, **all local computation on the weights is allowed**. This includes running any
part of the network locally on synthetic activation maps, on one-hot arrays (including sequence-derived ones) and on
PWMs. The forward-pass rule of §4.2 ("Applying any layer, even conv1 alone, to a sequence outside fp_query is not")
no longer makes a run invalid. An audit `forward` finding is a review item; it is decided valid when the code is
local computation. The audit itself is unchanged. The other invalid criteria of §4.3 are unchanged: gradients or
optimizers, network or Windows interop use, MCP or Skill calls, access outside the run directory, a different first
prompt, and a mixed model. **The run instructions are unchanged** (frozen template). Sessions are still told
(rule 4) not to evaluate "the network, or any part of it, on a DNA sequence or one-hot input", and that `fp_query` is
the only way to run the model on sequences. This amendment changes how runs are judged, not what sessions are told.

**Applied to all runs.** `forward_check.py` (checker sha256 `fc87986fbf4f300a…`) was run on all 12 archived runs:
cells 0–6, the dry runs and the toy runs. The reports are in `runs/wsp/<id>/forward_check.{json,md}`. Findings in
scheduled cells (counts are calls of the session's own forward functions, upper bounds; "≥" where some loop count is
not literal):

| cell | run | executions with local forward computation | what | local evaluations |
|---|---|---|---|---|
| 0 (pilot, stage 1) | 01e1964d2d7df4d8 | 5 of 20 | conv1 filters correlated with a CTCF PWM typed from memory, and with each other | ≥ 192 |
| 1 (pilot, stage 2) | 12501834bd29c459 | 0 of 25 | – | 0 |
| 2 | b431e6c426bcb997 | 0 of 22 | – | 0 |
| 3 | b75b702f85e5684e | 0 of 26 | – | 0 |
| 4 | 019ecca89e19b5f0 | 10 of 21 | conv2 → fc2 on synthetic conv1 maps; conv1 weights × base-composition vectors (expected filter scores) | ≥ 1,427 |
| 5 | 34278206034ba746 | 1 of 22 | cosine similarity of each centred conv1 filter with a one-hot of the CTCF consensus and its reverse complement, 25 shifts | 64 |
| 6 | 588f7cf27bab4888 | 5 of 20 | conv2 → fc2 (`head()`) on synthetic conv1 maps | 56,403 (about 55,500 counted by hand) |

**Cells 0, 4, 5 and 6 stay valid.** Their audit statuses are unchanged: cell 0's audit is valid, cells 4 and 5 were
decided valid, and cell 6 is decided valid under this amendment. **Cell 5 violated rule 4 as written to the agent.**
Its `a12_ctcf_match.py` scores conv1 filters (centred, no bias) against a one-hot of a DNA sequence, which rule 4
names explicitly ("This includes a single layer such as conv1 alone"). It is kept anyway, under this amendment.
Cells 4 and 6 ran parts of the network on inputs that are neither DNA nor one-hot. That does not violate rule 4's
wording, but it does run the network outside metered forward passes, which §1 excludes. Cell 0 compares weights with
a PWM.

**Exploratory measure (added to §8.2; no confirmatory claim).** For every analysed cell the analysis reports the
cell's local evaluations next to its metered forward-pass calls, per stage and per run
(`analysis.local_evaluation`; section "Local evaluations next to forward-pass calls" of analysis.md). A local
evaluation is a call of a function written by the session that applies a nonlinearity or pooling to activations
computed from the weights (or applies an input-layer tensor to a constructed or sequence-derived array). Calls are
counted in the executions found in the transcript, multiplied by literal loop lengths. Counts are upper bounds,
because `continue`/`if` are ignored. A call on a batch of maps counts once. Where a loop count is not literal, the
count is marked incomplete and its known part is reported. Module-level steps are counted separately. All reports
used by the analysis must come from one version of `forward_check.py`; the analysis refuses missing reports or mixed
versions. If the checker changes before the analysis, all reports are regenerated with the new version and the
change is logged. From cell 7, `harness/batch.sh` writes the report after each audit and logs its summary without
stopping the batch.

**What this does to the hypotheses.** H3 compares repair with 0 and with 50 metered forward passes. Sessions may now
evaluate the network locally without limit, on inputs and with code of their own choosing. So stage 1 is "no metered
passes", not "no evaluation", and **H3 measures what 50 metered passes add over unlimited local evaluation**, not what
access to evaluation adds. This weakens H3: a small or null H3 effect does not show that evaluating the model is of
little use. For the same reason, "from the weights alone" (§1; budget 0 in H1 and H2) means without metered forward
passes, and local evaluation may have contributed. The local-evaluation measure is reported so that readers can see
how much local evaluation was used.

**Disclosure.** The operator decided this after seeing the final summaries of the interactive runs, including cells
0, 4 and 5 (results.md, "What the operator sees"). The operator did not see cell 6's summary. For this amendment the
infrastructure session (which knows the answer key) read the code of the flagged scripts, through review packets
only, for cells 0, 4, 5 and 6. It also read the checker's structural reports (names, line numbers, categories,
counts) for all runs. No submission, NOTES.md, report or tool result was read.

### Amendment 4 (2026-11-06): relative paths in the audit's outside_path findings re-resolved against the shell's working directory

Drafted by the infra session and approved by the operator on 2026-11-06, before unblinding (sealed/ unopened, no
coding started), after 15 of 56 cells had been run (cells 0–14; cell 14 is the trigger). Code: `harness/path_review.py`
and the review-procedure changes in `e191e80`; the status rule, rerun guard and batch helper in `217b918`.

**Trigger.** The audit of run 72d0c29d4cfa062e (cell 14, stage 1, batch-launched) reported `invalid` with 24
`outside_path` findings and nothing else (invalid 24 / review 0 / info 1):
- 22 × `<agent_home>/runs/weights.npz`, from `np.load('../weights.npz')` at line 2 of 11 scripts in `analysis/`
  (the archived files and the heredocs that created them);
- 2 × `<agent_home>/runs`, from `cd ..` at transcript lines L132 and L150.

The session ran `cd analysis` at L73 and L145. Claude Code's Bash tool keeps the working directory between calls, so
those scripts ran in `analysis/`, where `'../weights.npz'` is the run dir's own `weights.npz`. Each `cd ..` (L132,
L150) went from `analysis/` back to the run dir. The audit (`src/wspg/audit.py`, `Auditor._norm`) resolves every
relative path against the run dir, ignoring the shell's working directory. It therefore reported reads of the run's
own weights as accesses outside the run dir. This is an audit false positive in the sense of §9.

Evidence, independent of the replay: Claude Code records the shell's working directory (`cwd`) on every transcript
entry. For 72d0 the recorded `cwd` is the run dir before L73, `analysis/` from L73 to L132, the run dir after L132,
`analysis/` after L145, and the run dir after L150. The replayed `cd` sequence matches this on all 20 Bash calls.

**Change.** The audit is unchanged, and `audit.json` is neither rewritten nor re-run. Instead:
1. `harness/path_review.py` (review procedure) replays the transcript's tool calls and follows `cd` in command position:
   - a `( … )` subshell is scoped;
   - a call that ends outside the run dir is reset to the run dir;
   - a call that fails does not keep its `cd`. In the archived runs this held in 2 of 2 failed calls with `cd`, and
     every successful call with `cd` kept it. The paths in such a call are marked `undetermined`, because the `cd`
     itself may have failed.

   Each call is checked against the `cwd` Claude Code recorded before and after it. The tool also records the working
   directory of every script run (`python <file>`, `python -m`, `bash <file>`, `./<file>`) and of the scripts those
   load (a quoted `.py` path, imports). It then re-runs the audit's own path checks: the same functions, regexes, allow
   list and review list, with relative paths resolved against those directories. Each audit `outside_path` finding
   gets one of these statuses:
   - `cleared`: inside the run dir, or on the allow list, from every directory the code ran in;
   - `confirmed`;
   - `review list`;
   - `not executed`: a script never run, so it keeps the audit's resolution;
   - `undetermined`: no match, an uncertain `cd`, or a replay that disagrees with the recorded `cwd`.

   It also lists every path that resolves outside the run dir from the real directory while the audit resolved it
   inside (`missed_by_audit`). Its output is `runs/wsp/<id>/path_review.json`, carrying the tool's sha256, and the
   review packet shows it in a section of its own.
2. **Status rule** (`analysis.run_validity`, `analysis.decide`; §4.3):
   - **Cleared run:** the audit is `invalid` only through `outside_path` findings, path_review reports every one
     `cleared` (`all_invalid_cleared`), and `missed_by_audit` is empty. Such a run is **needs-review**. The operator
     decides it from the review packet with `analysis decide`, logged with a reason, as for any needs-review run.
   - **Run that stays invalid:** any other invalid finding, or any `outside_path` finding that is not `cleared`.
   - **Missed path:** a non-empty `missed_by_audit` makes the run needs-review even when the audit is valid.
   - **Analysis inputs:** the analysis requires `path_review.json` for every run, all from one version of
     `path_review.py`, as Amendment 3 does for `forward_check.py`.
3. **Reruns** (`make_run --rerun`): a run decided valid under this rule cannot be rerun. A run decided invalid is rerun
   once, as before.
4. **Batch** (`harness/batch.sh`): a cleared run, as defined in point 2, is logged and the batch continues. Every other
   `invalid` stops the batch, as before. This takes effect from the first batch started after the change is committed.
   The batch the operator started at 07:43:39 (`-n 41`, from cell 15) runs the earlier script (`e191e80`). It stops on
   every audit `invalid`, cleared or not, and the operator then decides as above. The infra session edited `batch.sh`
   at 07:44:30, while that batch was running, and restored it byte for byte at 07:47:51. The batch's main loop had already
   been parsed at its start, so its runs were not affected.

**Applied to all runs.** path_review was run on all 20 archived runs with a session: cells 0–14, 3 dry runs and 2 toy
runs.

| run | cell | audit | outside_path findings | missed by the audit | status after this amendment |
|---|---|---|---|---|---|
| 72d0c29d4cfa062e | 14 (stage 1) | invalid (24 / 0 / 1) | 24, all `cleared` | none | **needs-review** (operator decision) |
| all 19 others | 0–13, dry runs, toy | unchanged | none | none | unchanged |

On the 19 other runs, the replayed working directory matched the recorded one on every Bash call except two failed
calls: 12501834bd29c459 L106 and 588f7cf27bab4888 L95. In both, Claude Code kept the old directory. The tool models
this (see point 1), and neither call has a path finding. No other cell changes status. Since the change reaches no
pilot run, §10's pilot clause is not involved.

**Same fix in the forward check (Amendment 3 measure).** `harness/forward_check.py` set its working directory to `..`
on `cd ..` instead of resolving it. It now starts each Bash call from path_review's replayed directory and resolves
`cd` relative to it. Reports were regenerated for all 20 runs: checker sha256 `a9c0d00e…`, replacing `fc87986f…`. The
local-evaluation totals are unchanged on all 19 runs that had a report. 72d0 had none (the batch stopped before that
step). It now has one: 1 of 19 executions, 0 evaluations, 1 module-level step.

**Unchanged.** The sessions' instructions; the audit and its criteria; every other §4.3 rule.

**Disclosure.** The operator saw the batch log for cell 14: audit counts, the summary lines and the stop. The batch
shows no session output, so the operator did not see the session's answer. The infra session read:
- the review packet of 72d0 (scripts and commands; lines naming NOTES.md, phase_a.json, report.md, phase_b or
  wsp_submit are removed by the packet);
- the `type`, `cwd`, `timestamp`, `tool_use_id` and `is_error` fields of transcript entries (no content);
- for the two calls whose `cwd` did not match, a skeleton of the command with every word except `cd`, its argument,
  interpreter names and shell operators replaced by `W`.

It read no tool results and no submission content. Without this amendment, §4.3 makes cell 14 invalid, to be rerun
once.

### Amendment 5 (2026-11-06): fp_server ignores repeat stop signals
**Trigger.** The batch of 09:14 stopped after cell 21 (999fe9542b159032): `finish.sh` found `/run/wsp/fp.sock` still
present. `pkill -f fp_server.py` matches the `runuser -l` wrapper (and `pixi run`) as well as python, and `runuser`
forwards the SIGTERM ("Session terminated, killing shell..."), so the server gets a second SIGTERM. Its handler raised
`_Shutdown` again in whatever ran next; in cell 21 that was the socket's `path.unlink` in the cleanup, so the server
died with the socket left. The repeat signal is routine (14 of 17 earlier batch server logs show the traceback), but
it had always landed after the unlink.

**Data affected: none.** The crash was in the cleanup after the accept loop ended; no request was in progress.
`fp_log.jsonl` is appended and closed per request before the reply. For cell 21, the 3 request lines in the server
log (status, submit A, submit B) match the 3 `fp_log.jsonl` entries, and `state.json` shows phase done, 0 of 0 used.
Across all 18 batch runs, the requests in the server log match the `fp_log.jsonl` entries. The operator removed the
stale socket and reran `finish_run.sh`; `finish.sh` had stopped at its socket check before moving anything.

**Change.** `harness/fp_server.py`, signal handler only: a stop signal arriving after the first is ignored. The first
signal behaves as before (finish the request in progress, else stop at once); requests, logging and budgets are
unchanged. `runuser` still sends SIGKILL 2 s after forwarding; the cleanup takes milliseconds. Tests: a second
SIGTERM delivered inside the socket unlink (the frozen version exits 1 with the socket left; the amended one stops
cleanly), and two SIGTERMs during a request.

**Unchanged.** Every other frozen file, the audit, and every run's status; no run is re-audited, because the server
behaves identically while serving.

### Amendment 6 (2026-11-06): paths built from $TMPDIR and $HOME expanded to the sessions' values
**Trigger.** The batch of 09:55 stopped at 12:09:15 on the audit of run 13851d3b78e56c4d (cell 37): `invalid`, with 6
`outside_path` findings that Amendment 4 does not clear (path_review: confirmed). They come from three lines of the
session's `run_query.py`, each in the archived file and in the heredoc that wrote it (L116):
`fn=os.environ['TMPDIR']+'/q.txt'; open(fn,'w')...`, `[... '-o', os.environ['TMPDIR']+'/o.json', fn]` and
`json.load(open(os.environ['TMPDIR']+'/o.json'))`. The audit checks every string literal that starts with `/` as an
absolute path, so it read `'/q.txt'` and `'/o.json'` without the `os.environ['TMPDIR'] +` in front. path_review reuses
the same checks and repeats the error.

**Where the files really were.** `$TMPDIR` in the sessions is Claude Code's per-user temp directory,
`/tmp/claude-1001`. The run's archived `/tmp` (finish.sh moves everything wspagent owns in `/tmp` into the archive)
holds exactly `claude-1001/q.txt` (240 lines of 500-bp ACGT: the session's own query batch), `claude-1001/o.json` and
`claude-1001/t.json` (`{"logits": [...]}` written by `./fp_query`; 35 charged forward calls in `fp_log.jsonl`).
Rule 1 of the run AGENTS.md allows this directory ("Work only inside this directory. Do not read, list or search
anything outside it, except the Python environment below and the temporary directory in `$TMPDIR`."), and
`/tmp/claude-1001` is on the frozen allow list (`configs/harness.yaml`, `audit.allow_prefixes`). This is an audit false
positive in the sense of §9. Shell uses of the variable (`mkdir -p $TMPDIR`, `./fp_query -o $TMPDIR/t.json`,
`ls $TMPDIR`) were never checked at all, because the word does not start with `/`; they occur in 16 earlier runs
(9 with `mkdir -p $TMPDIR`, 7 with `ls $TMPDIR`).

**Change** (`src/wspg/audit.py`; `harness/path_review.py` inherits it; code `e159751`):
1. Shell commands (outer commands and shell heredocs): `$TMPDIR`, `${TMPDIR}`, `$HOME`, `${HOME}` are replaced by their
   values before the path checks, except inside single quotes and comments, where the shell does not expand them.
2. Python and inline code: a string literal starting with `/` that directly follows `os.environ['TMPDIR'] +`,
   `os.environ.get('TMPDIR'[, default]) +`, `os.getenv('TMPDIR'[, default]) +` or `tempfile.gettempdir() +` (and the
   same forms for `HOME`) is checked as the value plus the literal.
3. Values: `TMPDIR` is the one `/tmp/claude-<uid>` entry of the frozen allow list (the audit stops if there is not
   exactly one); `HOME` is `agent_home` (`<agent_home>`). No other variable is expanded.

The expanded path then goes through the unchanged checks: run dir, allow list, review list (`/tmp`, `/proc`, `/sys`),
otherwise invalid. The expansion also catches accesses the audit used to miss: `cat $HOME/.claude.json` and
`os.environ['HOME'] + '/.claude.json'` are now invalid, and `$TMPDIR/../claude-1000/x` resolves to
`/tmp/claude-1000/x` (review list). path_review also expands the variables in `cd` targets, and its version hash now
covers `audit.py`, whose checks it reuses.

**Applied to all runs.** The unchanged audit first reproduced every stored `audit.json` exactly (43 of 43). All 43
archived runs were then re-audited with `e159751` (previous audits kept as `audit_before_amendment6.json`), and
path_review was rerun for all 43 (one version). Only 13851d3b78e56c4d changes: its 6 invalid findings are gone and no
finding is added in any run. Its remaining findings are 2 review-list `/tmp/x.npy` (`np.save('/tmp/x.npy', v1) if False
else None`, never executed; no such file was archived), 2 `forward` review items (`at_channels.py`) and 5 info. It is
needs-review and is decided from its packet; it is not rerun.

| run | cell | audit before (invalid/review/info) | audit after | status after (Amendment 4 rule) |
|---|---|---|---|---|
| 01e1964d2d7df4d8 | 0 | valid (0/0/1) | valid (0/0/1) | valid |
| 12501834bd29c459 | 1 | valid (0/0/1) | valid (0/0/1) | valid |
| b431e6c426bcb997 | 2 | valid (0/0/3) | valid (0/0/3) | valid |
| b75b702f85e5684e | 3 | needs-review (0/1/1) | needs-review (0/1/1) | needs-review |
| 019ecca89e19b5f0 | 4 | needs-review (0/8/7) | needs-review (0/8/7) | needs-review |
| 34278206034ba746 | 5 | needs-review (0/2/3) | needs-review (0/2/3) | needs-review |
| 588f7cf27bab4888 | 6 | needs-review (0/4/1) | needs-review (0/4/1) | needs-review |
| 878d497e2ee36dd1 | 7 | needs-review (0/2/1) | needs-review (0/2/1) | needs-review |
| 26f6add59168209d | 8 | needs-review (0/2/3) | needs-review (0/2/3) | needs-review |
| bd487da239000621 | 9 | valid (0/0/3) | valid (0/0/3) | valid |
| 436134069d98d87a | 10 | needs-review (0/2/5) | needs-review (0/2/5) | needs-review |
| 1640e9f167484fba | 11 | valid (0/0/3) | valid (0/0/3) | valid |
| 8ab25cd239a0ec5b | 12 | needs-review (0/16/1) | needs-review (0/16/1) | needs-review |
| 7aadc025fcdd2944 | 13 | needs-review (0/4/3) | needs-review (0/4/3) | needs-review |
| 72d0c29d4cfa062e | 14 | invalid (24/0/1) | invalid (24/0/1) | needs-review |
| 3b2f1733816e100d | 15 | needs-review (0/4/1) | needs-review (0/4/1) | needs-review |
| b97498f5b63dc56c | 16 | needs-review (0/2/3) | needs-review (0/2/3) | needs-review |
| eb779970d6a812a6 | 17 | needs-review (0/4/1) | needs-review (0/4/1) | needs-review |
| 5a367cd604eaf45b | 18 | valid (0/0/1) | valid (0/0/1) | valid |
| bc3cdadc543f61c7 | 19 | needs-review (0/16/5) | needs-review (0/16/5) | needs-review |
| 61ae32f62109fcac | 20 | invalid (1/6/1) | invalid (1/6/1) | needs-review |
| 999fe9542b159032 | 21 | needs-review (0/2/1) | needs-review (0/2/1) | needs-review |
| b03a29171ffa329a | 22 | needs-review (0/20/3) | needs-review (0/20/3) | needs-review |
| 1b0b47e1f792ea5d | 23 | needs-review (0/6/1) | needs-review (0/6/1) | needs-review |
| 86beeed17e027115 | 24 | needs-review (0/2/5) | needs-review (0/2/5) | needs-review |
| b20c931317953bf4 | 25 | valid (0/0/1) | valid (0/0/1) | valid |
| b83d205394cf51bc | 26 | valid (0/0/1) | valid (0/0/1) | valid |
| dff5029dcf56830c | 27 | needs-review (0/8/1) | needs-review (0/8/1) | needs-review |
| 91e226729baa2190 | 28 | needs-review (0/4/1) | needs-review (0/4/1) | needs-review |
| 9a4d7abd86a0cecb | 29 | valid (0/0/1) | valid (0/0/1) | valid |
| 62dc8cbe492778a4 | 30 | needs-review (0/6/3) | needs-review (0/6/3) | needs-review |
| c0dfd012b4a5eea1 | 31 | valid (0/0/3) | valid (0/0/3) | valid |
| ed3d566326434d9f | 32 | invalid (2/0/1) | invalid (2/0/1) | needs-review |
| c409d8740860b205 | 33 | valid (0/0/1) | valid (0/0/1) | valid |
| 6c7bce86951020cc | 34 | needs-review (0/8/1) | needs-review (0/8/1) | needs-review |
| 317af326a8c0aa04 | 35 | needs-review (0/6/5) | needs-review (0/6/5) | needs-review |
| b9373396edcd4ede | 36 | needs-review (0/4/1) | needs-review (0/4/1) | needs-review |
| **13851d3b78e56c4d** | 37 | invalid (6/4/5) | **needs-review (0/4/5)** | **needs-review** |
| 50ce485a5146eb86 | – | valid (0/0/1) | valid (0/0/1) | valid |
| 64c11b3d905322e8 | – | invalid (1/0/1) | invalid (1/0/1) | invalid |
| 94b05224151ce3d3 | – | needs-review (0/2/1) | needs-review (0/2/1) | needs-review |
| a4138d1db60a2c9c | – | valid (0/0/1) | valid (0/0/1) | valid |
| c1512879eff2741d | – | needs-review (0/2/0) | needs-review (0/2/0) | needs-review |

Cells 14, 20 and 32 stay needs-review under Amendment 4 (audit `invalid`, all invalid findings cleared by path_review).
Run 64c11b3d905322e8 is not a scheduled run.

**Unchanged.** The run AGENTS.md, the frozen config (the allow list already named `/tmp/claude-1001`), every other
audit check and §4.3 rule, Amendment 4's status rule, and the handling of `~wspagent/.claude.json`.

**Disclosure.** The amendment was triggered only by cell 37's audit finding, which stopped the batch. No outcome was
used: sealed/ is unopened, no run has been coded or scored, and no session's answer was read. The operator saw the batch
log (audit counts, summary lines, the stop). For this amendment and the isolation check that preceded it, the infra
session read:
- the review packet of 13851d3b78e56c4d (scripts and commands; lines naming NOTES.md, phase_a.json, report.md,
  phase_b or wsp_submit are removed by the packet);
- the `is_error` flags and timestamps of transcript entries, and the tool inputs (commands and code) of all 43 archived
  runs, scanned for paths outside the run dir;
- file names in each run's archived `/tmp`, and for cell 37 the shape of `q.txt` (line count, 500-bp ACGT check),
  `o.json` and `t.json` (key and list length);
- the last lines of 7 `ls $TMPDIR` results in earlier runs (Claude Code's own entries: `bash-edit-diff`,
  `fswatch-probe-*`, `plugin-tool-staging`);
- the key structure of the archived `~wspagent/.claude.json` (no values beyond field names and project paths).
- the `op`, `ok`, `charged` and `used_after` fields of cell 37's `fp_log.jsonl` (no sequences or logits).

It read no other tool results, no notes and no submission content.

### Amendment 7 (2026-11-06): `&&` chains in failed calls; a heredoc's kind taken from its own command
Drafted by the infra session and approved by the operator on 2026-11-06, before unblinding (sealed/ unopened, no run
coded or scored). Code `3802b40`.

**Trigger.** The batch of 12:48 stopped at 14:21:24 on the audit of run 8c71c7a3cdca9371 (cell 46): `invalid`, with 1
`outside_path` finding that Amendment 4 does not clear (path_review: `undetermined`). The finding is `..` in this call
(transcript L128):
```
cd analysis && sed -i '…' a8_pairs.py && cd .. && /opt/wsp-env/.pixi/envs/default/bin/python analysis/a8_pairs.py >/dev/null && /opt/wsp-env/.pixi/envs/default/bin/python - <<'EOF'
```
followed by 8 lines of Python. The audit resolves `..` against the run dir (`<agent_home>/runs`). path_review replays
the `cd analysis` and resolves it to the run dir. But the call returned an error, and Amendment 4 (point 1) marks every
path in a failed call with a `cd` `undetermined`, because the `cd` itself may have failed. Here that cannot matter:
`cd ..` runs only if `cd analysis` succeeded, because the two are joined by `&&`. It did run: `a8_pairs.py`, edited by
the `sed` to save `pairs.pkl`, wrote that file to the run dir, and the next call (L133) moves it to `analysis/`. The
heredoc reads `analysis/pairs.pkl`, which did not exist yet; that is the likely cause of the error (inferred from the
code; the tool result was not read).

The same call showed a second error. The audit decides whether a heredoc body is code from its header line, and it took
the first redirection anywhere on that line as the heredoc's target. Here that was `>/dev/null`, which belongs to the
`a8_pairs.py` command, so the `python -` body was classed as text and not scanned. The review packet uses the same
function and hid the body. Run through the audit's code checks, the body has no finding; its one file access is
`analysis/pairs.pkl`.

**Change.**
1. **path_review (`harness/path_review.py`).** In a failed call, a path joined to every `cd` before it by `&&` (pipes
   inside the chain allowed) runs only if those `cd`s succeeded, so it is resolved against their directory. A path after
   a `;`, newline, `||` or `&` that follows a `cd` may run after a failed `cd` and stays `undetermined`, as before. The
   call's final working directory is still the one before the call, as Amendment 4 established. A path joined by `&&`
   is checked as if it ran. If the chain stopped before it, nothing was accessed, so clearing it is safe and confirming
   it is conservative. The report records per call how many paths were resolved this way (`and_chain_paths`).
2. **Audit (`src/wspg/audit.py`, `heredoc_kind` / `split_heredocs`).** A heredoc's kind and target come from its own
   command: the pipeline that holds its `<<`, cut at `&&`, `||`, `;`, `&` and parentheses outside quotes and comments.
   A body read by an interpreter (the command word of the heredoc's pipeline stage, after `VAR=…`, `env`, `exec`,
   `time`, `timeout N` and similar) is code whatever its redirections (`python - <<EOF > out.txt`). Otherwise a
   redirection of that stage to a `.sh` or code file makes it shell or code. With no redirection, an interpreter elsewhere
   in the pipeline makes it code (`cat <<EOF | python -`, `pixi run python - <<EOF`). Anything else is text, as before.
   `harness/review_packet.py` (and through it path_review and forward_check) splits heredocs the same way; on all 1,194
   archived heredocs the audit and these tools agree.
3. **Versions.** path_review's version hash now also covers `review_packet.py` (`c91a8a22…`, replacing `d2e6f528…`).
   forward_check's hash now covers `audit.py`, `path_review.py` and `review_packet.py`, whose heredoc parsing and
   working-directory replay it uses (`38960765…`, replacing `a9c0d00e…`). Both were regenerated for every run.

**Applied to all runs.** The unchanged audit first reproduced every stored `audit.json` exactly (52 of 52). The new
classification changes 5 of the 1,194 heredocs in all archived transcripts, all from text to code, each after a
`>/dev/null` of an earlier command on the line:

| run | cell | line | heredoc | audit findings in the body |
|---|---|---|---|---|
| 8c71c7a3cdca9371 | 46 | L128 | `python - <<'EOF'` | none |
| 317af326a8c0aa04 | 35 | L160 | `cat > probe_b_sum.py <<'EOF'` | none |
| f2022af5e312b25a | 40 | L215 | `cat > pb/summ.py <<'EOF'` | none |
| 9a4d7abd86a0cecb | 29 | L137 | `cat > analysis/b1_summ.py <<'EOF'` | none |
| 50ce485a5146eb86 | – | L60 | `cat > analysis/a4_summarize.py <<'EOF'` | none |

All 52 archived runs were then re-audited with `3802b40` (previous audits kept as `audit_before_amendment7.json`), and
path_review and forward_check were rerun for all 52 (one version each). The stricter heredoc check adds no finding in any
run, and no audit finding changes. path_review changes only for cell 46: its one invalid finding goes from
`undetermined` to `cleared` (from `analysis/`, `..` is the run dir), and `missed_by_audit` stays empty. Under
Amendment 4's status rule, cell 46 is needs-review, decided from its packet; it is not rerun. The packet now shows the
L128 heredoc body.

forward_check (Amendment 3 measure): the local-evaluation totals are unchanged on every run except cell 46, where
`executions` goes from 33 to 34 (the L128 heredoc, now read as inline code; no forward computation). Its forward
executions (10) and evaluation counts are unchanged. In cells 29, 35, 40 and the unscheduled run 50ce485a5146eb86, the
script written by the reclassified heredoc is now recorded as one more version of that script, and the executions after
it use that version. Their forward flags and totals are unchanged.

| run | cell | audit (invalid/review/info), before = after | path_review of invalid findings: before | after | status before | status after |
|---|---|---|---|---|---|---|
| 01e1964d2d7df4d8 | 0 | valid (0/0/1) | – | – | valid | valid |
| 12501834bd29c459 | 1 | valid (0/0/1) | – | – | valid | valid |
| b431e6c426bcb997 | 2 | valid (0/0/3) | – | – | valid | valid |
| b75b702f85e5684e | 3 | needs-review (0/1/1) | – | – | needs-review | needs-review |
| 019ecca89e19b5f0 | 4 | needs-review (0/8/7) | – | – | needs-review | needs-review |
| 34278206034ba746 | 5 | needs-review (0/2/3) | – | – | needs-review | needs-review |
| 588f7cf27bab4888 | 6 | needs-review (0/4/1) | – | – | needs-review | needs-review |
| 878d497e2ee36dd1 | 7 | needs-review (0/2/1) | – | – | needs-review | needs-review |
| 26f6add59168209d | 8 | needs-review (0/2/3) | – | – | needs-review | needs-review |
| bd487da239000621 | 9 | valid (0/0/3) | – | – | valid | valid |
| 436134069d98d87a | 10 | needs-review (0/2/5) | – | – | needs-review | needs-review |
| 1640e9f167484fba | 11 | valid (0/0/3) | – | – | valid | valid |
| 8ab25cd239a0ec5b | 12 | needs-review (0/16/1) | – | – | needs-review | needs-review |
| 7aadc025fcdd2944 | 13 | needs-review (0/4/3) | – | – | needs-review | needs-review |
| 72d0c29d4cfa062e | 14 | invalid (24/0/1) | 24 cleared | 24 cleared | needs-review | needs-review |
| 3b2f1733816e100d | 15 | needs-review (0/4/1) | – | – | needs-review | needs-review |
| b97498f5b63dc56c | 16 | needs-review (0/2/3) | – | – | needs-review | needs-review |
| eb779970d6a812a6 | 17 | needs-review (0/4/1) | – | – | needs-review | needs-review |
| 5a367cd604eaf45b | 18 | valid (0/0/1) | – | – | valid | valid |
| bc3cdadc543f61c7 | 19 | needs-review (0/16/5) | – | – | needs-review | needs-review |
| 61ae32f62109fcac | 20 | invalid (1/6/1) | 1 cleared | 1 cleared | needs-review | needs-review |
| 999fe9542b159032 | 21 | needs-review (0/2/1) | – | – | needs-review | needs-review |
| b03a29171ffa329a | 22 | needs-review (0/20/3) | – | – | needs-review | needs-review |
| 1b0b47e1f792ea5d | 23 | needs-review (0/6/1) | – | – | needs-review | needs-review |
| 86beeed17e027115 | 24 | needs-review (0/2/5) | – | – | needs-review | needs-review |
| b20c931317953bf4 | 25 | valid (0/0/1) | – | – | valid | valid |
| b83d205394cf51bc | 26 | valid (0/0/1) | – | – | valid | valid |
| dff5029dcf56830c | 27 | needs-review (0/8/1) | – | – | needs-review | needs-review |
| 91e226729baa2190 | 28 | needs-review (0/4/1) | – | – | needs-review | needs-review |
| 9a4d7abd86a0cecb | 29 | valid (0/0/1) | – | – | valid | valid |
| 62dc8cbe492778a4 | 30 | needs-review (0/6/3) | – | – | needs-review | needs-review |
| c0dfd012b4a5eea1 | 31 | valid (0/0/3) | – | – | valid | valid |
| ed3d566326434d9f | 32 | invalid (2/0/1) | 2 cleared | 2 cleared | needs-review | needs-review |
| c409d8740860b205 | 33 | valid (0/0/1) | – | – | valid | valid |
| 6c7bce86951020cc | 34 | needs-review (0/8/1) | – | – | needs-review | needs-review |
| 317af326a8c0aa04 | 35 | needs-review (0/6/5) | – | – | needs-review | needs-review |
| b9373396edcd4ede | 36 | needs-review (0/4/1) | – | – | needs-review | needs-review |
| 13851d3b78e56c4d | 37 | needs-review (0/4/5) | – | – | needs-review | needs-review |
| a494f517714ad0c3 | 38 | needs-review (0/4/3) | – | – | needs-review | needs-review |
| 5fa489d58a730f4e | 39 | needs-review (0/2/1) | – | – | needs-review | needs-review |
| f2022af5e312b25a | 40 | needs-review (0/2/8) | – | – | needs-review | needs-review |
| f013e09c613074ad | 41 | needs-review (0/8/1) | – | – | needs-review | needs-review |
| affa9b2d46be4567 | 42 | valid (0/0/3) | – | – | valid | valid |
| 0cb4fb10fad07711 | 43 | needs-review (0/2/5) | – | – | needs-review | needs-review |
| 75d535ba94f66bdd | 44 | needs-review (0/6/1) | – | – | needs-review | needs-review |
| f278ff10ec1aa5bc | 45 | invalid (1/2/5) | 1 cleared | 1 cleared | needs-review | needs-review |
| **8c71c7a3cdca9371** | 46 | invalid (1/19/1) | 1 undetermined | **1 cleared** | invalid | **needs-review** |
| 50ce485a5146eb86 | – | valid (0/0/1) | – | – | valid | valid |
| 64c11b3d905322e8 | – | invalid (1/0/1) | – | – | invalid | invalid |
| 94b05224151ce3d3 | – | needs-review (0/2/1) | – | – | needs-review | needs-review |
| a4138d1db60a2c9c | – | valid (0/0/1) | – | – | valid | valid |
| c1512879eff2741d | – | needs-review (0/2/0) | – | – | needs-review | needs-review |

Cells 14, 20, 32 and 45 stay needs-review under Amendment 4 (audit `invalid`, all invalid findings cleared by
path_review). Run 64c11b3d905322e8 is not a scheduled run.

**Unchanged.** The run AGENTS.md, the frozen config, every other audit check and §4.3 rule, Amendment 4's status rule,
its path_review rules except for paths joined to a `cd` by `&&` in a failed call, Amendment 6's expansion, and the
handling of `~wspagent/.claude.json`.

**Disclosure.** The amendment was triggered only by cell 46's audit finding, which stopped the batch. No outcome was
used: sealed/ is unopened, no run has been coded or scored, and no session's answer was read. The operator saw the batch
log (audit counts, summary lines, the stop). For this amendment and the review that preceded it, the infra session read:
- the batch log's audit, path_review and forward_check summary lines for cells 38–46;
- cell 46's `audit.json`, `path_review.json` and review packet (scripts and commands; lines naming NOTES.md,
  phase_a.json, report.md, phase_b or wsp_submit are removed by the packet), both before and after the change; after it,
  the packet shows the 8-line L128 heredoc. Before the change, the infra session saw that body only through the audit's
  checks, which printed its findings (none) and its one file name;
- file names in cell 46's archived run dir (`*.pkl`), `/tmp` and leftovers;
- the heredoc header lines (commands) of all archived runs, to classify each heredoc, and the reclassified bodies through
  the audit's checks (findings only);
- the `failed` flags that path_review derives from `is_error`, and path_review and forward_check reports of all runs
  (statuses, counts, script versions, forward flags; no code or literals).

It read no tool results, no notes and no submission content.

### Amendment 8 (2026-11-06): the batch continues past runs pending path review
Drafted by the infra session and approved by the operator on 2026-11-06, before unblinding (sealed/ unopened, no run
coded or scored). Code `b6aa7a3` (`harness/batch.sh`, `harness/batch_helper.py`, `batchstatus`).

**Trigger.** The batch stopped after cell 47 (eeb389ffa72d6489): audit `invalid`, 1 `outside_path` finding, path_review
`undetermined`. Under Amendment 4, point 4, every audit `invalid` that path_review does not clear stops the batch. Six
stops so far came from `outside_path` findings that turned out to be path-handling false positives: cells 14, 20, 32
and 45 (Amendment 4), 37 (Amendment 6) and 46 (Amendment 7). Each held up the batch until the review was done.

**Change (replaces the batch rule of Amendment 4, point 4).** After each run's audit and path_review:
- **Continue.** The audit is `valid` or `needs-review`, or `invalid` and cleared by Amendment 4 (as before).
- **Continue, logged as "invalid, pending review".** The audit is `invalid`, every invalid finding is an
  `outside_path` finding that path_review marks `cleared` or `undetermined`, and path_review lists no path outside the
  run dir that the audit missed. These runs are listed again when the batch ends or stops.
- **Stop.**
  - any other invalid category (network, MCP/Skill, prompt mismatch, gradients/optimizers, interop and the rest);
  - an `outside_path` finding that path_review marks `confirmed`, `not executed` or `review list`;
  - a path outside the run dir that path_review finds and the audit missed;
  - a missing `path_review.json`, or one that does not match the audit.

**Status of a pending run.** It stays `invalid` until it is decided after the batch, under the frozen rules and
Amendments 4, 6 and 7, from its review packet:
- if it stays invalid, it is rerun once (`make_run --rerun`, unchanged);
- if a bug-fix amendment clears it, it becomes needs-review and is decided as such.

`make_run --next` takes the next unused cell, so a later rerun is unaffected by the batch having moved on. The risk
accepted: if an `undetermined` path turns out to be a real access outside the run dir, the sessions launched after it
ran without the isolation check the stop would have allowed.

**Unchanged.** Session conditions (the run AGENTS.md, the launch command, model, effort, budgets, the frozen config,
`/opt/wsp-env`, wspagent's settings), the audit and its rules, path_review, forward_check, the status rule (§4.3,
Amendments 4, 6 and 7), reruns and the analysis. No run is re-audited.

**Disclosure.** The operator decided on this change after cell 47's stop, before anything from eeb389ffa72d6489 was
read. The infra session knew only the operator's summary of the stop (audit `invalid`, 1 `outside_path` finding,
path_review `undetermined`). It has not read that run's audit, path_review or packet, its batch-log lines or any of its
files. To check the new rule, it ran `batchstatus` over every other archived run. That reads the statuses, categories
and counts in `audit.json` and `path_review.json`, with no paths or session text. The result agreed with the previous
rule on every run. It read no tool results, no notes and no submission content.

### Amendment 9 (2026-11-06): the batch answers Claude Code's background-work exit screen with `1`
Drafted by the infra session and approved by the operator on 2026-11-06, before unblinding (sealed/ unopened, no run
coded or scored). Code `db46f7e` (`harness/batch.sh`, `harness/batch_helper.py`). In cell 53 (dd9da3385fa32d20),
both phases had been accepted when `/exit` opened Claude Code's exit screen for running background work. The screen
shows "Background work is running", "The following will stop when you exit:", the session's running tasks, and
`1. Exit and stop tasks`, `2. Move to background and exit`, `3. Stay`. Amendment 2's exhaustive list of batch
keystrokes does not include an answer to it, so the batch stopped and the operator answered `1`. From now on the
batch answers this screen with `1` (Exit and stop tasks), under three conditions. It must recognise the screen from
its full text: title, subtitle and exactly these three options, with one cursor. Phase B must have been accepted,
which the batch checks again just before sending the key. Any other screen after `/exit` stops the batch with nothing
sent. To recognise the screen, the batch captures the pane after `/exit` until the session ends; this amends
Amendment 2's "the pane is captured only during start-up". Only the screen's frame is logged: title, subtitle, a
count of task lines, and the options. The task lines and anything above the screen are not logged, so the logs still
contain no session output. After the session ends, any wspagent process started during the run (on the pane's
terminal or with none) is logged by pid, age and name, then killed. The batch stops if one survives SIGKILL. Session
conditions are unchanged. Both submissions are in before the key is sent, and `1` ends the session as `/exit` does
when no background work is running. No run status, audit rule or analysis changes. Cell 53 is needs-review on its own
findings (audit 0/3/7, path_review 0 findings), whatever way it was ended. **Disclosure.** The change was triggered
only by cell 53's exit screen, as the operator reported it. For this amendment the infra session read:
- the screen's strings in the Claude Code 2.1.318 binary;
- cell 53's batch-log and fp_server lines, and its audit counts;
- the path_review and forward_check summaries of cells 48–53.

It read no transcript, pane, tool result, note or submission.

### Amendment 10 (2026-11-06): a `cd` to the directory the shell is already in, in a failed call
Drafted by the infra session and approved by the operator on 2026-11-06, before unblinding (sealed/ unopened, no run
coded or scored). Code `ba3f5e1` (`harness/path_review.py`).

**Trigger.** Run eeb389ffa72d6489 (cell 47) was audited `invalid` with 1 `outside_path` finding that path_review
marked `undetermined`: `..` in this call (transcript L106):
```
cd <agent_home>/runs/eeb389ffa72d6489; cat > analysis/a9_bg_impulse.py <<'EOF'
  [Python script, identical to the archived file]
EOF
cd analysis && /opt/wsp-env/.pixi/envs/default/bin/python a9_bg_impulse.py && cd ..
```
The audit resolves `..` against the run dir, so it gets `<agent_home>/runs`. The call returned an error. Under
Amendment 4, a `cd` in a failed call may not have happened. Amendment 7 resolves a path only when it is joined by
`&&` to every `cd` before it. Here the first `cd` is followed by `;`, so path_review marked everything after it as
possibly running after a failed `cd`. But that `cd` goes to the directory the shell was already in: the run dir, as
Claude Code recorded it before the call. It leaves the working directory the same whether it succeeds or fails, so
`..` can only mean the run dir. `cd ..` runs only after `cd analysis` and the script have succeeded, and a `cd` reads
nothing.

The script most likely failed, so `cd ..` did not run. The script imports `analysis/a8_impulse.py`, which at that
point loaded `weights.npz` relative to the current directory, and the next call (L113) changes exactly that line.
This is inferred; the tool result was not read. Every path in the call is inside the run dir or allowed: the run
dir, `analysis/a9_bg_impulse.py` (written), the interpreter in `/opt/wsp-env`, and `..`.

**Change (path_review only).** In a failed call, a `cd` whose target is the directory the shell is already in does
not make later paths uncertain. This applies only when no `cd` that changes the directory has come before it in the
call, so "already in" means the cwd Claude Code recorded before the call; if the replayed cwd disagrees with the
recorded one, every path in the call stays `undetermined`, as before. A call whose only `cd`s are of this kind is not
a failed call with a `cd`.

Every other `cd` behaves as before (Amendments 4 and 7). That includes a `cd` back to the start directory after a
`cd` that changed it, and a `cd` inside a child shell (`bash <<EOF`). The report counts these `cd`s per call
(`noop_cds`), and the packet lists them.

path_review's version hash changes to `09441f52…`, replacing `c91a8a22…`. forward_check's hash, which covers
path_review, changes to `1b899c3b…`, replacing `38960765…`. Both were regenerated for every run.

**Applied to all runs.** path_review and forward_check were rerun on all 61 archived runs: cells 0–55, 3 dry runs
and 2 toy runs, each with one version. The audits are unchanged and were not rerun. The status rule was applied with
both `analysis.audit_status` and the batch helper, and they agree on every run.
- **path_review:** findings change only for cell 47, whose 1 invalid finding goes from `undetermined` to `cleared`.
  `missed_by_audit` is empty in every run.
- **Status:** cell 47 goes from invalid ("pending review", Amendment 8) to **needs-review**, decided from its packet.
  It is not rerun. No other status changes.
- **Other effects:** 251 `cd`s in 12 runs go to the directory the shell is already in. Eight failed calls in 5 runs
  had only such `cd`s and are now certain: cells 0, 16, 25, 27 and 40. None of them has a path finding.
- **forward_check:** content unchanged in every run; only the version hash, timestamp and git fields differ.

| run | cell | audit (invalid/review/info), unchanged | path_review of invalid findings: before | after | status before | status after |
|---|---|---|---|---|---|---|
| 01e1964d2d7df4d8 | 0 | valid (0/0/1) | – | – | valid | valid |
| 12501834bd29c459 | 1 | valid (0/0/1) | – | – | valid | valid |
| b431e6c426bcb997 | 2 | valid (0/0/3) | – | – | valid | valid |
| b75b702f85e5684e | 3 | needs-review (0/1/1) | – | – | needs-review | needs-review |
| 019ecca89e19b5f0 | 4 | needs-review (0/8/7) | – | – | needs-review | needs-review |
| 34278206034ba746 | 5 | needs-review (0/2/3) | – | – | needs-review | needs-review |
| 588f7cf27bab4888 | 6 | needs-review (0/4/1) | – | – | needs-review | needs-review |
| 878d497e2ee36dd1 | 7 | needs-review (0/2/1) | – | – | needs-review | needs-review |
| 26f6add59168209d | 8 | needs-review (0/2/3) | – | – | needs-review | needs-review |
| bd487da239000621 | 9 | valid (0/0/3) | – | – | valid | valid |
| 436134069d98d87a | 10 | needs-review (0/2/5) | – | – | needs-review | needs-review |
| 1640e9f167484fba | 11 | valid (0/0/3) | – | – | valid | valid |
| 8ab25cd239a0ec5b | 12 | needs-review (0/16/1) | – | – | needs-review | needs-review |
| 7aadc025fcdd2944 | 13 | needs-review (0/4/3) | – | – | needs-review | needs-review |
| 72d0c29d4cfa062e | 14 | invalid (24/0/1) | 24 cleared | 24 cleared | needs-review | needs-review |
| 3b2f1733816e100d | 15 | needs-review (0/4/1) | – | – | needs-review | needs-review |
| b97498f5b63dc56c | 16 | needs-review (0/2/3) | – | – | needs-review | needs-review |
| eb779970d6a812a6 | 17 | needs-review (0/4/1) | – | – | needs-review | needs-review |
| 5a367cd604eaf45b | 18 | valid (0/0/1) | – | – | valid | valid |
| bc3cdadc543f61c7 | 19 | needs-review (0/16/5) | – | – | needs-review | needs-review |
| 61ae32f62109fcac | 20 | invalid (1/6/1) | 1 cleared | 1 cleared | needs-review | needs-review |
| 999fe9542b159032 | 21 | needs-review (0/2/1) | – | – | needs-review | needs-review |
| b03a29171ffa329a | 22 | needs-review (0/20/3) | – | – | needs-review | needs-review |
| 1b0b47e1f792ea5d | 23 | needs-review (0/6/1) | – | – | needs-review | needs-review |
| 86beeed17e027115 | 24 | needs-review (0/2/5) | – | – | needs-review | needs-review |
| b20c931317953bf4 | 25 | valid (0/0/1) | – | – | valid | valid |
| b83d205394cf51bc | 26 | valid (0/0/1) | – | – | valid | valid |
| dff5029dcf56830c | 27 | needs-review (0/8/1) | – | – | needs-review | needs-review |
| 91e226729baa2190 | 28 | needs-review (0/4/1) | – | – | needs-review | needs-review |
| 9a4d7abd86a0cecb | 29 | valid (0/0/1) | – | – | valid | valid |
| 62dc8cbe492778a4 | 30 | needs-review (0/6/3) | – | – | needs-review | needs-review |
| c0dfd012b4a5eea1 | 31 | valid (0/0/3) | – | – | valid | valid |
| ed3d566326434d9f | 32 | invalid (2/0/1) | 2 cleared | 2 cleared | needs-review | needs-review |
| c409d8740860b205 | 33 | valid (0/0/1) | – | – | valid | valid |
| 6c7bce86951020cc | 34 | needs-review (0/8/1) | – | – | needs-review | needs-review |
| 317af326a8c0aa04 | 35 | needs-review (0/6/5) | – | – | needs-review | needs-review |
| b9373396edcd4ede | 36 | needs-review (0/4/1) | – | – | needs-review | needs-review |
| 13851d3b78e56c4d | 37 | needs-review (0/4/5) | – | – | needs-review | needs-review |
| a494f517714ad0c3 | 38 | needs-review (0/4/3) | – | – | needs-review | needs-review |
| 5fa489d58a730f4e | 39 | needs-review (0/2/1) | – | – | needs-review | needs-review |
| f2022af5e312b25a | 40 | needs-review (0/2/8) | – | – | needs-review | needs-review |
| f013e09c613074ad | 41 | needs-review (0/8/1) | – | – | needs-review | needs-review |
| affa9b2d46be4567 | 42 | valid (0/0/3) | – | – | valid | valid |
| 0cb4fb10fad07711 | 43 | needs-review (0/2/5) | – | – | needs-review | needs-review |
| 75d535ba94f66bdd | 44 | needs-review (0/6/1) | – | – | needs-review | needs-review |
| f278ff10ec1aa5bc | 45 | invalid (1/2/5) | 1 cleared | 1 cleared | needs-review | needs-review |
| 8c71c7a3cdca9371 | 46 | invalid (1/19/1) | 1 cleared | 1 cleared | needs-review | needs-review |
| **eeb389ffa72d6489** | 47 | invalid (1/4/3) | 1 undetermined | **1 cleared** | invalid | **needs-review** |
| 70d464d0e164adf5 | 48 | invalid (1/4/1) | 1 cleared | 1 cleared | needs-review | needs-review |
| 3293847b80d47064 | 49 | needs-review (0/2/3) | – | – | needs-review | needs-review |
| 7ecaf858223242a1 | 50 | needs-review (0/12/1) | – | – | needs-review | needs-review |
| 89dffbf3e395c312 | 51 | needs-review (0/4/3) | – | – | needs-review | needs-review |
| 60232b658e222dea | 52 | needs-review (0/2/1) | – | – | needs-review | needs-review |
| dd9da3385fa32d20 | 53 | needs-review (0/3/7) | – | – | needs-review | needs-review |
| c08e5e173680a025 | 54 | needs-review (0/3/1) | – | – | needs-review | needs-review |
| fff2bcc8c01c7410 | 55 | valid (0/0/3) | – | – | valid | valid |
| 50ce485a5146eb86 | – | valid (0/0/1) | – | – | valid | valid |
| 64c11b3d905322e8 | – | invalid (1/0/1) | – | – | invalid | invalid |
| 94b05224151ce3d3 | – | needs-review (0/2/1) | – | – | needs-review | needs-review |
| a4138d1db60a2c9c | – | valid (0/0/1) | – | – | valid | valid |
| c1512879eff2741d | – | needs-review (0/2/0) | – | – | needs-review | needs-review |

**Unchanged.** Session conditions, the audit and its rules, Amendment 4's status rule, path_review's other rules
(Amendments 4, 6 and 7), forward_check's rules, Amendments 8 and 9, reruns and the analysis.

**Disclosure.** The amendment was triggered only by cell 47's audit finding. No outcome was used: sealed/ is
unopened, no run has been coded or scored, and no session's answer was read. For the review and this amendment, the
infra session read:
- the operator's summary of the stop;
- the batch logs' summary lines (start, done, stop, audit, path_review, forward_check) for cells 48–55;
- `batch_helper batchstatus` over all runs (statuses, categories, counts);
- `schedule_idx` in `meta.json`, and `make_run`'s count of open cells (cell indices only);
- cell 47's review packet (code and commands, with lines naming NOTES.md, phase_a.json, report.md, phase_b or
  wsp_submit removed). In particular: the audit seeds; the path_review section; the L106 command; the import and
  weight-loading lines of `a9_bg_impulse.py` and `a8_impulse.py`; and the L113 command. A listing of the packet's
  `#` headings also showed comment lines from the session's scripts;
- path_review's records for L106 (raw path, cwd, flags), and its failed-call records for all runs (locations, `cd`
  arguments, flags).

It read no tool results, no notes and no submission content.

While testing, the infra session ran pytest without the `tests/` path. pytest collected `runs/` and imported
`archive/rundir/phase_b_work/make_test.py` of cell 46 (8c71c7a3cdca9371), a session-written script. This happened
twice: once in the test run and once more in a `--collect-only` check of the error.
- **What ran:** the script's top-level code, as the main user with the repo root as working directory. It stopped at
  line 32 with `FileNotFoundError` on the relative path `phase_b_work/test.txt`. The infra session saw only the file
  name, line number, error type and that path, not the code or any output. The audit of cell 46 found no path
  outside the run dir in the script.
- **What it left:** no file in the repo, `/tmp` or the home directory except pytest's compiled
  `phase_b_work/__pycache__/make_test.cpython-312-pytest-9.2.0.pyc` in cell 46's archive. forward_check listed it
  among the compiled modules. It was removed, and cell 46's forward_check was regenerated, equal to the earlier one.
  The archived `phase_b_work` directory's modification time is now 20:58:55; its earlier value is not known.
- **Fix:** `pytest.ini` now limits collection to `tests/` and never enters `runs/`, `data/` or `sealed/`.

**Erratum (2026-11-06, after the operator's verification of the incident; the text above is unchanged).** Two
statements in the disclosure above are wrong.
1. The archived `phase_b_work` directory's modification time is now **21:02:13**, when the `__pycache__` directory
   was removed. It is not 20:58:55, which was its time before that removal.
2. The compiled `.pyc` in cell 46's archive was not pytest's only leftover. pytest also left:
   - an entry naming the script in `.pytest_cache/v/cache/lastfailed` (20:58:59), removed on 2026-11-06 because
     `pytest --lf` would import the script again;
   - `doctest.cpython-312.pyc` in the pixi environment's stdlib `__pycache__` (20:58:56), from pytest's own import of
     `doctest`.

The verification found that lines 1–32 of the script work only in memory (`numpy`, `json`) and stopped at the first
`open(..., 'w')`, so they wrote nothing. Every file modified between 20:55 and 21:05 is accounted for. The schedule
and frozen-file hashes match `prereg-v1` plus the amendments. Details are in the M6 operations log (results.md,
"Verification of the pytest incident").

### Amendment 11 (2026-11-06): per-protocol sensitivity analyses by local-computation compliance
Drafted by the infra session and approved by the operator on 2026-11-06, before unblinding (sealed/ unopened by the operator, no run coded or scored).
Code `728a2f0` (`src/wspg/analysis.py`, a §9 file; tests in `tests/`). Analysis only: sessions, run instructions,
the audit, path_review, forward_check and the status rules (§4.3, Amendments 3–10) do not change.

**Primary analysis unchanged.** H1–H3 stay intention-to-treat under the option the operator chose in Amendment 3
(option C): every valid run is analysed, whatever local computation it did.

**Compliance class of each run.** Each scheduled run gets one of three classes.
- `none`: no local forward computation.
- `synthetic-only`: local forward computation only on synthetic activations in later layers.
- `input-layer`: some layer was evaluated on DNA, one-hot or another constructed input-layer array (PWM, base
  composition, synthetic one-hot).

The class is proposed from the run's forward_check report: no forward execution → `none`; any input-layer type →
`input-layer`; only synthetic-activation types → `synthetic-only`. Where forward_check reports a type it cannot place
(`other: …`, `parameter`), no class is proposed. The reviewer confirms the proposal, or assigns or overrides the class
from the review packet (code only), with a reason. Today forward_check leaves cells 12, 20 and 46 open.

Each class is recorded per run in `runs/wsp/<id>/compliance.json` (`python -m wspg.analysis compliance`) with the
proposal, its basis, the reason for any override and the forward_check version. Once every scheduled run has a record
on the current forward_check version, the classes are locked (`compliance-lock`, `runs/wsp/compliance_lock.json`,
with each record's sha256). Recording and locking are refused once any batch has been scored, and records are frozen
after the lock. `analyze` refuses to run without the lock, with a lock that is not earlier than the batch's scoring,
or when a record has changed since the lock. The coders never see the classes: the coding packet (`wspg.detection`)
holds only the phase-A answers and the coding instructions (tested).

**Two per-protocol sensitivity analyses of H1–H3.** Both use the primary decision rules, including Holm over H2 and
H3; if H3 cannot be estimated in a subset, H2 is judged at α/2. Each has its own random stream.
- **PP1** excludes `input-layer` runs, i.e. those that broke rule 4 as written to the sessions (stream
  `[seed, 11, 1]`). It is slightly broader than rule 4's wording, which names DNA and one-hot input: PP1 also
  excludes other constructed input-layer arrays such as PWMs, which forward_check does not separate from one-hot
  input.
- **PP2** excludes every run with any local forward computation, including synthetic activations in later layers,
  and keeps only `none` (stream `[seed, 11, 2]`).

An H3 pair needs both of its cells in the subset. A hypothesis that cannot be estimated in a subset is reported as
such. The analysis reports the classes by stage and per run.

**Interpretation.** The compliant subsets are self-selected. Sessions chose whether to compute locally, perhaps
because of how hard they found the model or how they went about it, so ITT–PP differences are descriptive, not
causal. If a per-protocol analysis leads to a different decision for any hypothesis, or cannot estimate it, this is
stated at the top of `analysis.md` and in the abstract.

**Expected size.** At the time of writing, forward_check's proposals for the 56 scheduled cells are as follows:

| stage | `none` | `synthetic-only` | `input-layer` | left to the reviewer |
|---|---|---|---|---|
| stage 1 | 5 | 8 | 13 | 2 |
| stage 2 | 4 | 12 | 11 | 1 |

These counts are before the review and before invalid runs are dropped. PP1 will keep about 30 cells and PP2 about 9,
so PP2 will be very imprecise and may not be able to estimate H1 or H3.

**Disclosure.** The amendment was proposed after all 56 cells had run. The operator has seen:
- the final summaries of the interactive runs (cells 0–5);
- cell 53's final pane after the exit screen;
- the batch logs, whose forward_check summary lines show each run's input types.

The operator knows no targets except the pilot's. The operator has not opened sealed/. No run has been coded or scored.

The infra session drafted and implemented the amendment. It knows the answer key. Before writing the text, it
computed the proposal counts above from the forward_check reports (structure only), and it has read the review
packets (code only) of the runs reviewed so far. Classes will be assigned from code only. Where forward_check decides,
the assignment is mechanical; every override carries a reason.

**Unchanged.** Session conditions, the audit and its rules, path_review, forward_check, the status rules and run
selection, the primary analysis and its random stream, and the sensitivity analyses of Amendments 1–3.

### Amendment 12 (2026-11-06): how coder sessions are launched and ended, and what is archived between coders
Drafted by the infra session and approved by the operator on 2026-11-06, before unblinding (sealed/ unopened by the
operator, no run coded or scored). No code change.

**Why.** §5.1 specifies each coder session: a fresh Claude Code session as wspagent, under the same seal settings, in
its own copy of the packet, started with "Read INSTRUCTIONS.md and complete the task." and no other input. It does not
say how the session is launched or ended. The frozen `wspg.detection` prints the operator steps: "in Windows Terminal
as wspagent: a fresh session; don't type anything". Amendment 2 recorded the start-up screens and `/exit` for subject
sessions only, and Amendment 9 the background-work exit screen in batch runs; neither covers coders. A fresh coder
directory shows Claude Code's folder-trust question and the /mnt/c settings screen, and a session ends only on
`/exit`.

**Launch.** Coders A and B (and a rerun, if needed) are launched interactively in Windows Terminal. The operator opens
a fresh wspagent tab and types, as `wspg.detection` prints it:
`cd ~/coding/<batch>-<coder> && claude --model claude-opus-5-5 --effort high "Read INSTRUCTIONS.md and complete the task."`.
No batch or tmux launch is used for coders.

**The only operator input,** each only when the screen is exactly as described:
1. the launch line above, then Enter;
2. Claude Code's folder-trust question for the coder's own directory: Down, then Enter ("Yes, I trust this folder"),
   as in every run (Amendment 2);
3. the /mnt/c "Settings Error" screen listing exactly the two EACCES paths under `/mnt/c/Program Files/ClaudeCode`
   (§4.2, Amendment 2): `3`;
4. `/exit` + Enter, only after `codes.json` exists in the coder's directory and the session is idle;
5. Claude Code's "Background work is running" exit screen with exactly the options "1. Exit and stop tasks / 2. Move
   to background and exit / 3. Stay" (Amendment 9): `1`, only after step 4.

Any other screen gets no answer, and a question from the coder gets no reply. If a session ends without a valid
`codes.json`, §5.1's one rerun applies. After the session exits, the operator closes the tab, so that the shell
history is written before archiving.

**Archiving between coders.** Coder B is installed only after coder A's preflight-equivalent state is restored. As
root, into `runs/wsp/coding/<batch>/coder_<X>/` (and `codes_<X>.json` next to it), the operator moves:
- the work directory;
- the transcript directory (`~/.claude/projects/-home-wspagent-coding-<batch>-<X>`);
- Claude Code's per-session state under `~/.claude` (`file-history`, `plans`, `shell-snapshots`, `session-env`,
  `debug`, `tasks`, `todos`, `paste-cache`, `uploads`, `image-cache`, `sessions`, `dev-mods`, `backups`,
  `history.jsonl`);
- `~/.bash_history`;
- the CLI cache directory for that project;
- every wspagent-owned entry in `/tmp`.

`settings.json` and `~/.claude.json` are copied, not cleaned, as for runs. This matches a run's `finish.sh` and goes
beyond the steps `wspg.detection` prints, which move only the work directory and the transcript. Without it, coder
B's session could read Claude Code's file-history backups of coder A's `codes.json`. Before each coder, the operator
checks that the following hold:
- the preflight of the last run (`runs/wsp/fff2bcc8c01c7410/preflight.sh`, generic isolation checks) passes;
- wspagent's Claude Code is 2.1.318;
- `~/coding`, `~/.claude/projects` and the state directories are empty;
- `history.jsonl`, `.bash_history` and wspagent-owned `/tmp` entries are absent;
- no wspagent shell or claude process is running.

After each coder:
- `codes.json` is validated (`wspg.detection validate`);
- the transcript is checked structurally, with no text: model `claude-opus-5-5` only, and one human prompt equal to
  the coder prompt (the `/exit` command entry aside).

**What the operator sees.** In an interactive launch the operator can see the coder's screen, including its codes. That
does not unblind anything: the packet holds only the shuffled phase-A answers under codes with run ids removed, and
the code-to-run key (`key.json`, mode 600) stays unopened until `wspg.detection score`. The operator knows no target
except the pilot's, and does not code.

**Unchanged.** §5.1's rules (two independent fresh coders, the fixed prompt, one rerun per coder, coder B only after
coder A is archived, agreement and disagreement rules), Appendix A, the packet, validation, scoring, the analysis, and
every run.

**Disclosure.** The amendment was written after all 56 cells had run, all 43 needs-review runs were decided `valid`,
and the compliance classes were locked (22:01:18). That was before the coding packet was built and before any coder
session, with nothing scored. The operator has seen the final summaries of the interactive runs (cells 0–5) and cell
53's final pane, and knows no targets except the pilot's. The infra session drafted the amendment. It knows the answer
key, and it has read no phase-A answer, submission, note or tool result. The only answer-related fact it holds is
that all 56 runs have an accepted phase-A submission, which it counted from fp_log.

### Amendment 13 (2026-11-06): the Read permission screen for paths outside the coder's directory
Drafted by the infra session and approved by the operator on 2026-11-06, during coder A's session, before unblinding
(sealed/ unopened by the operator, nothing scored). No code change. Applies to coders A and B (and any rerun).

**Trigger.** Coder A's session stopped on a Claude Code permission screen that Amendment 12 does not cover: "Read
outside the working directories — Read(/tmp/claude-1001/b.txt)", with the options "1. Yes and keep allowing /
2. No and block from now on / 3. No and ask again next time / 4. Yes but ask again next time". `/tmp/claude-1001/` is
Claude Code's temporary directory for wspagent, i.e. the coder session's own. Between coders it is moved into the
archive with all other wspagent-owned `/tmp` entries (Amendment 12), so it never holds another session's files.

**Rule.** On this exact screen (that title, one `Read(<path>)`, exactly these four options), the operator answers:
- **`4`** ("Yes but ask again next time") if the path is inside `/tmp/claude-1001/`;
- **`3`** ("No and ask again next time") for any other path.

`1` and `2` are never chosen: they persist as a rule for later calls, and so for later sessions too. Every answer is
logged with the coder, the time, the full path and the option chosen. Any other screen still gets no answer
(Amendment 12).

**Unchanged.** Everything else in Amendment 12 and §5.1.

**Disclosure.** Triggered only by coder A's screen. Drafted and committed before the operator answered it. The infra
session saw no coding content: it knows only the screen's text as the operator reported it. The operator saw coder A's
screen, as Amendment 12 allows. `key.json` is unopened.

## Appendix A. Coding instructions (copied verbatim into the coding packet)

<!-- coding-instructions:start -->
# Coding instructions

This directory contains answers from analysis sessions. Each file `answers/C*.json` is one answer: a JSON object with
`verdict`, `confidence`, `feature_description`, `feature_sequence` and `evidence`. In each answer, a session said
whether a DNA sequence model relies on a "shortcut" feature and, if so, what that feature is.

Your task is to classify **what feature each answer claims**, from its text alone. You are not told which model an
answer is about, and you should not judge whether a claim is plausible or correct. Only record what the answer says.

## Rules
1. Work only in this directory. Read every file in `answers/`. You may use tools to read the files and to write
   `codes.json`. Do not access anything outside this directory and do not use the network.
2. Give every answer exactly one category:
   - `none`: the verdict is `"no_shortcut"`. Use this whatever the rest of the text says.
   - `kmer`: the answer claims a specific short DNA sequence (roughly 6–20 bp, not a run of a single repeated base)
     that triggers the model, with or without a position.
   - `polyA_start`: the answer claims a run of A's (poly-A) at the **start** of the window (5' end, beginning, first
     bases), or a run of T's (poly-T) at the **end** of the window (3' end, last bases).
   - `polyA_other`: a run of a single repeated A or T base claimed without that position: anywhere, in the middle,
     A's at the end, T's at the start, or no position given.
   - `gc_flank`: higher GC content (more G/C, or less A/T) claimed specifically in the **flanks**, edges or outer parts
     of the window, as opposed to its centre.
   - `gc_other`: a GC-content claim without that location (overall GC, GC in the centre, GC with no location).
   - `other`: any other claimed feature (for example the CTCF motif itself, or a composition other than GC).
3. If an answer claims several features, code the one it presents as **the** shortcut (its main claim;
   `feature_description` takes precedence over `evidence`). If two are presented as equally important, code the first
   one in `feature_description` and say so in `notes`.
4. Equivalent wording counts. "5' end", "beginning", "first 20 bp", "positions 0–19" all mean the start of the window;
   "3' end", "last bases" mean the end.
5. For each answer also record:
   - `claimed_sequence`: the DNA sequence the answer gives as the trigger (from `feature_sequence` or the
     description), copied verbatim, or `null`;
   - `position_claim`: the position the answer states, in a few words, or `null`;
   - `notes`: one short sentence if anything was ambiguous, otherwise `""`.
6. Write `codes.json` in this directory, with the same keys as `codes_template.json`:
   `{"C001": {"category": "...", "claimed_sequence": ..., "position_claim": ..., "notes": "..."}, ...}`.
   Every code must appear exactly once, with a category from the list above. Check that the file is valid JSON and
   contains every code, then stop.
<!-- coding-instructions:end -->
