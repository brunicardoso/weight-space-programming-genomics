# Weight-space programming on a regulatory-genomics CNN: public release 1

This is the curated public release of a preregistered, blinded experiment. We tested whether an LLM agent (Claude
Opus 5.5 in Claude Code) can **detect** and **remove** a planted, non-biological shortcut in a small CNN that predicts
CTCF binding from DNA. The agent reasons over the network's weights and gets no gradients and no training.

The preprint is in `reports/preprint/main.pdf`, with its source next to it.

## Results in brief
- **56 blind sessions** on 18 planted and 10 clean models. Each session first named a shortcut with no forward passes
  (phase A), then submitted repaired weights with 0 or 50 metered forward passes (phase B).
- **H1 (detection, primary): supported, narrowly.** D = 0.396 [0.021, 0.792], one-sided p = 0.0177. Arithmetic on
  the preregistered bootstrap shows the support rests on one correct poly-A answer. There were no false alarms on clean
  models.
- **Budget 0 was not "weights alone".** 47 of 56 sessions ran parts of the network locally, on synthetic activations or
  constructed inputs. All 7 correct detections came from such sessions, and the per-protocol analyses did not support
  detection.
- **Repair: H2 and H3 not supported.** 2 of 36 planted-model sessions met the fix criterion, and no edit harmed
  clean performance.

Full results, the operations log and every amendment are in `reports/results.md` and `reports/preregistration.md`.

## Layout
| path | contents |
|---|---|
| `reports/preregistration.md` | preregistration (frozen at tag `prereg-v1`), 13 dated amendments and the erratum |
| `reports/results.md` | milestone results and the full M6 operations log, including invalid runs, slips and disclosures |
| `reports/preprint/` | preprint source (`main.tex`, `refs.bib`), figure/table generator, figures and `main.pdf` |
| `results/analysis/` | outputs of the frozen analysis: `analysis.md`, `analysis.json`, `repair.json` |
| `results/coding/` | `confusion.md` and `detection_summary.json` (coded outcomes per run, no answer text) |
| `results/compliance/` | locked compliance classes (`compliance_lock.json`) and the per-run compliance records |
| `src/wspg/` | data pipeline, model, training, planting, evaluation, audit, detection scoring and analysis code (in `plant.py` the two functions that draw the plants are withheld; see `REDACTIONS.md`) |
| `harness/` | blind-session harness: run template, forward-pass server, CLIs, batch launcher, audit helpers |
| `configs/`, `tests/` | configurations and the test suite |
| `VERIFICATION.md` | commit hashes and SHA-256 sums that tie this release to the private repository |
| `REDACTIONS.md` | what is withheld and why |
| `MANIFEST.md` | every file with its size and SHA-256 |

## Running the tests
```bash
pixi install          # environment from pixi.toml / pixi.lock (Linux, CUDA 12.8 PyTorch build)
pixi run pytest -q tests
```
Some tests use the harness's shell scripts and need a Linux shell with `bash`. The test suite needs no withheld data.

## Running the analysis
The frozen analysis (`python -m wspg.analysis analyze --batch main`) needs the archived runs, the coders' codes and
the sealed answer key. All of these are withheld (see `REDACTIONS.md`). Without them, `wspg.analysis analyze` and
`wspg.analysis repair` stop with a message saying what is missing. `wspg.detection score` and `wspg.detection packet`
stop with a `FileNotFoundError` naming the first missing file. The analysis code is frozen at the preregistration, so
this behaviour was not changed. Their outputs are included in `results/`, and `reports/preprint/make_figures.py`
documents how every figure and table was derived from them. In this release that script cannot run as is, because it
also reads withheld files.

## Placeholders
Local paths and identifiers were replaced:

| placeholder | stands for |
|---|---|
| `<repo>` | the project checkout |
| `<home>` | the main user's home directory |
| `<user>` | the main user's login name |
| `<agent_home>` | home directory of the sandboxed agent user (`wspagent`) |
| `<winuser>` | the Windows user name |
| `<host>` | the machine's host name |

Code, configs and tests use runnable stand-ins instead: `/home/mainuser` and `mainuser` for the main user, and
`/home/mainuser/repo` for the checkout. They keep the home directory of `wspagent`, the sandbox account the harness
creates. To run the
harness itself, set `main_user`, `agent_user` and the paths in `configs/harness.yaml` (and the matching variables at the
top of `harness/batch.sh` and `harness/finish_run.sh`) to your own values.

## Licence
Code is under the MIT licence; documents and data are under CC BY 4.0. See `LICENSE`. To cite this work, use
`CITATION.cff`.
