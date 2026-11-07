# What this release withholds, and why

This is public release 1. It contains the code, the preregistration with all amendments, the results and operations
log, the preprint, and redacted analysis outputs. Everything else is withheld. The table lists each category.

| withheld | why |
|---|---|
| **Answer keys and plant identities** (`sealed/`: which models are planted, the planted k-mer sequences, positions, the run schedule) | The same 28 models are planned for a blinded follow-up. Publishing the plants would unblind it. The preprint calls the k-mers "A", "B" and "C". |
| **How the plants were drawn** (the bodies of `choose_kmer` and `make_spec` in `src/wspg/plant.py`) | The plant generator is deterministic, and the seeds are part of the model names, so this code would let anyone regenerate the planted k-mers and the contaminated windows. The two functions are kept as stubs that raise `NotImplementedError`; the screening rules (constants at the top of the file, and the preregistration) and the injection code (`inject`, `apply_spec`) are published. |
| **Model weights and built datasets** (`*.npz`, `*.pt`, `data/`) | The weights encode the plants, and the datasets mark which windows were contaminated. The underlying ENCODE data are public (accessions in the preprint and `results.md`). |
| **Run archives** (`runs/`): session transcripts, the sessions' notes and scripts, phase-A answers and phase-B submissions | They contain the planted sequences as the sessions found them. They also hold archived Claude Code state with account data. |
| **Forward-pass logs** (`fp_log`) | Query sequences can contain plant sequences, and the logits leak model identity. |
| **Coding material**: the coding packet, `key.json`, the coders' `codes_*.json` and transcripts | Answer text and k-mer check windows contain plant sequences. A detection summary without answer text is in `results/coding/detection_summary.json`. |
| **Review packets, review reports, the fitting check, batch logs, audit and forward_check reports** | They quote session code and commands, which can reveal plants or answers. Their conclusions are in `results.md` and in the preregistration's amendments. |
| **Claude Code state and credentials** of the main user and of `wspagent` (`.credentials.json`, `.claude.json`, settings copies with account data, `history.jsonl`, `.bash_history`) | Account data and tokens. They are never published. |
| **Personal paths and identifiers** (home directories, user names, host name, Windows user name) | Replaced by placeholders in the documents (`<repo>`, `<home>`, `<user>`, `<agent_home>`, `<winuser>`, `<host>`) and by runnable stand-ins in the code (`/home/mainuser`, `mainuser`). The author's name and institutional e-mail appear only in the preprint. |
| **The export script** (`harness/export_public.py`, `harness/public_release/`) | It lists the local identifiers it replaces. It stays in the private repository, so the export can be reproduced there. |

Within the files that are included:
- `results/analysis/repair.json` and `analysis.json` are published unchanged except for placeholder substitution. They
  hold metrics only.
- `results/coding/detection_summary.json` drops the automatic k-mer check's matched windows and fields. It keeps each
  run's coded category, outcome, verdict and confidence.

**A second release is planned after the follow-up study.** It would add the withheld run archives (transcripts,
submissions, logs), the answer keys and the full `plant.py`, once publishing them can no longer unblind anything.
