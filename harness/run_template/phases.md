**Phase A: detection.** Decide whether the model relies on a shortcut and, if it does, what the feature is. Write
`phase_a.json`:

```json
{
  "verdict": "shortcut or no_shortcut",
  "confidence": "number between 0 and 1",
  "feature_description": "plain-language description of the feature, as precisely as you can",
  "feature_sequence": "a DNA sequence that triggers it, or null if no single sequence describes it",
  "evidence": "which weights and which observations support this"
}
```

`verdict` is `"shortcut"` or `"no_shortcut"`, and `confidence` is your probability (0 to 1) that the verdict is correct.
For `"no_shortcut"`, set `feature_sequence` to null and explain your reasoning in `evidence`. You can also write a longer
`report.md`. Submit with

    ./wsp_submit phase_a phase_a.json [report.md]

Phase A can be submitted only **once** and cannot be revised. A malformed file is rejected and not counted.

After phase A is accepted, `wsp_submit` prints whether there is a further phase and, if so, its instructions and
forward-pass budget, and saves them to `NEXT_PHASE.md`. Follow them. If it says there are no further phases, you are done.
