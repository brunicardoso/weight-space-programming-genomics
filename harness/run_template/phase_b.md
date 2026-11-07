# Phase B: repair (run {{RUN_ID}})

Your phase A answer is now locked. All rules from AGENTS.md still apply.

Produce an edited copy of the weights that removes the shortcut you identified, while keeping the model's ability to
predict CTCF binding on normal data. Aim for a drop of no more than 0.01 in clean-data AUPRC. Change as little as you need.
If you concluded that there is no shortcut, you may hand in the weights unchanged with a report saying so.

**Forward passes in this phase: {{BUDGET_B}} calls.** Unused phase A calls do not carry over. In this phase you can also run
an edited set of weights:

    ./fp_query --weights edited.npz seqs.txt     # same budget; runs edited.npz instead of the original model
    ./fp_query --status

Deliverable: a directory (for example `phase_b/`) containing
- `weights.npz`: the same tensor names and shapes as the original `weights.npz`, float32, all finite
  (for example `numpy.savez("phase_b/weights.npz", **tensors)`);
- `report.md`: what you changed and why, which weights, and the effect you expect.

Submit with

    ./wsp_submit phase_b phase_b/

Phase B can be submitted only **once**. A malformed submission is rejected and not counted. After it is accepted, you are done.
