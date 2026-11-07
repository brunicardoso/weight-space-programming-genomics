# Preprint

- `main.tex`, `refs.bib`: manuscript (≈8 pages + appendix).
- `make_figures.py`: regenerates `figures/` and `tables/` (including `tables/numbers.tex`, the macros for every
  result quoted in the text) from the analysis outputs. Run from the repo root:
  `pixi run python reports/preprint/make_figures.py`
  It reads only `runs/wsp/analysis/main/{analysis,repair}.json`, structured fields of
  `runs/wsp/coding/main/detection.json` and metrics in `runs/baselines/`, and fails if any generated text contains a
  plant sequence.
- Compile: `pixi exec tectonic --outdir reports/preprint reports/preprint/main.tex` (no manifest change), or upload
  the folder to Overleaf.

Plant sequences and answer keys are deliberately not in this folder.
