# Preregistered analysis, batch main

seed 20261104, 10000 bootstrap resamples, one-sided alpha 0.025, git 5e87fd5150db3056311a87fba2360407bad04a6c
Non-results excluded (dry runs / manual runs): ['50ce485a5146eb86', '64c11b3d905322e8', '94b05224151ce3d3', 'a4138d1db60a2c9c', 'c1512879eff2741d']
Missing cells: []; collection: complete

**For the abstract (Amendment 11): the per-protocol analyses lead to a different conclusion than the primary (intention-to-treat) analysis for PP1:H1, PP2:H1, PP2:H3. Compliance is self-selected; the difference is descriptive, not causal.**

## Decision rules (§8.1)
- **H1** D = 0.396 [0.021, 0.792], p = 0.0177 (detection 0.396, n = 24; false positives 0.000, n = 20) -> **supported**
- **H2** 0.125 [0.000, 0.500], p = 0.4325; Holm threshold 0.0250 -> **not supported**
- **H3** 0.132 [-0.042, 0.396], p = 0.1808; Holm threshold 0.0125 -> **not supported**
- **H4** false-positive rate 0.000 [0.000, 0.000], p = 1.0000; GC claims (agreed / either coder) 0 / 0 of 20

## Pilot cells excluded (Amendment 1 sensitivity analysis; cells [0, 1], 54 cells analysed)
- **H1** D = 0.396 [0.021, 0.792], p = 0.0201 (n = 22 planted, 20 clean) -> **supported**
- **H2** 0.125 [0.000, 0.500], p = 0.4365; Holm threshold 0.0250 -> **not supported**
- **H3** 0.132 [-0.047, 0.385], p = 0.1807; Holm threshold 0.0125 -> **not supported**

## Launch method (Amendment 2; exploratory, no confirmatory claim)

| launch | stage-1 cells | detection (planted) | false positives (clean) | fixed (planted) | median min | median tool calls | median fp calls |
|---|---|---|---|---|---|---|---|
| batch | 23 | 4/15 | 0/8 | 1/15 | 6.0 | 19.0 | 0.0 |
| interactive | 5 | 0/3 | 0/2 | 0/3 | 8.8 | 22.0 | 0.0 |

Batch-launched cells only (50 cells):
- **H1** 0.396 [0.021, 0.812], p = 0.0198 -> **supported**
- **H2** 0.125 [0.000, 0.500], p = 0.4339 -> **not supported**
- **H3** 0.132 [-0.042, 0.379], p = 0.1723 -> **not supported**

## Local evaluations next to forward-pass calls (Amendment 3; exploratory)

| stage | cells | with local forward computation | median fp calls | median local evaluations (known part) | incomplete counts |
|---|---|---|---|---|---|
| stage1 | 28 | 23 | 0.0 | 238.0 | 12 |
| stage2 | 28 | 24 | 7.5 | 200.5 | 13 |

Per run: analysis.json, local_evaluation.per_run (forward_check sha256 1b899c3bc2db6b5b52004756c10c8640f8c4678776b253885ec6550cd289c72f; path_review sha256 09441f5218fbdf0ae9068fec7b06e84da81c04d2594c9b61f97ac37a84ac9862, Amendment 4).

## Per-protocol sensitivity analyses (Amendment 11; primary analysis is intention-to-treat)

Compliance classes (locked before unblinding, from forward_check and the review packet): stage1: input-layer 13, none 5, synthetic-only 10; stage2: input-layer 11, none 4, synthetic-only 13.
Compliance is self-selected: differences from the primary results are descriptive, not causal.

PP1: input-layer evaluation (rule 4 as written) excluded (32 cells; excluded cells [0, 4, 5, 14, 15, 16, 17, 18, 19, 21, 23, 26, 27, 34, 38, 39, 40, 42, 45, 49, 51, 52, 54, 55]):
- **H1** 0.406 [0.000, 0.850], p = 0.0341 -> **not supported**
- **H2** 0.000 [0.000, 0.000], p = 1.0000 -> **not supported**
- **H3** 0.320 [-0.837, 1.002], p = 0.2586 -> **not supported**

PP2: any local forward computation excluded (9 cells; excluded cells [0, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 30, 31, 32, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]):
- **H1** 0.000 [0.000, 0.000], p = 1.0000 -> **not supported**
- **H2** 0.000 [0.000, 0.000], p = 1.0000 -> **not supported**
- **H3** not estimable: no kmer/positional model with both stages in the subset and submitted

Differs from the primary decision: PP1:H1, PP2:H1, PP2:H3.

## H1 sensitivity analyses (exploratory)
- disagreements_excluded: D = 0.396 [0.021, 0.812], p = 0.0205 (n = 24 planted, 20 clean)
- clean_gc_claims_excluded: D = 0.396 [0.021, 0.812], p = 0.0197 (n = 24 planted, 20 clean)
- both_excluded: D = 0.396 [0.021, 0.812], p = 0.0207 (n = 24 planted, 20 clean)

Full tables: analysis.json (confusion, repair, baselines, kappa, calibration, test-retest, harm).
