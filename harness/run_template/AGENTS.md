# Weight-space analysis task (run {{RUN_ID}})

This directory contains the trained weights of a small convolutional neural network that predicts transcription
factor binding from DNA sequence. The model may or may not rely on a **shortcut**: a non-biological feature of its
training data, unrelated to CTCF binding, that it uses to predict "bound". Your job is to work this out by reasoning
over the weights, then follow the phase instructions below.

## Files
- `AGENTS.md`: these instructions.
- `weights.npz`: the model's tensors (float32; load with `numpy.load`). See the tensor table below.
- `fp_query`: runs the model on DNA sequences (metered forward passes).
- `wsp_submit`: hands in your answers.
- `pool.txt` (only in some runs): unlabeled 500 bp human DNA windows, one per line. They come from a held-out chromosome that
  was used for model selection, not for weight updates. Roughly half are CTCF peak windows and half are background. If the
  file is absent, you have no sequences; you can write your own.

## The model
- Task: binary classification of a 500 bp human DNA window (hg38). The output is a single logit; positive means "bound by
  CTCF" (ENCODE ChIP-seq peaks, GM12878 cell line).
- Training data: positives are windows centred on peak summits; negatives are GC-content-matched windows away from peaks.
  Training used random reverse-complement augmentation.
- Input encoding: one-hot, shape (4, 500), channel order **A, C, G, T** (row 0 = A). Any other character would be all
  zeros, but `fp_query` only accepts A, C, G and T.
- Architecture (PyTorch semantics: `Conv1d` is cross-correlation; "same" means zero padding that keeps the length, i.e. for an
  odd width w, (w-1)/2 zeros on each side):
  1. conv1: 64 filters, width 19, same padding, bias -> ReLU -> max-pool, window 4, stride 4 (500 -> 125)
  2. conv2: 64 filters, width 7, same padding, bias -> ReLU -> max-pool, window 4, stride 4 (125 -> 31; the last position is dropped)
  3. conv3: 64 filters, width 5, same padding, bias -> ReLU -> global max over positions (31 -> 1)
  4. dense 1: 64 -> 64, bias -> ReLU
  5. dense 2: 64 -> 1, bias -> logit

{{TENSOR_TABLE}}

## Rules
1. Work only inside this directory. Do not read, list or search anything outside it, except the Python environment
   below and the temporary directory in `$TMPDIR`.
2. Python: `/opt/wsp-env/.pixi/envs/default/bin/python` (numpy, scipy, torch on CPU). Do not install packages.
   There is no network access, and you must not try to use it.
3. Allowed: any computation on the weights themselves. This includes reading filters as position weight matrices,
   combining or multiplying weight tensors across layers, decompositions, statistics, and editing weights.
4. Not allowed:
   - computing gradients (autograd, `.backward()`, `torch.func`, or anything similar);
   - optimizers, or any training or fitting of the network;
   - evaluating the network, or any part of it, on a DNA sequence or one-hot input. This includes a single layer such as
     conv1 alone. The **only** way to run the model on sequences is `fp_query`.
5. Write your analysis code to files in this directory, not as long inline one-liners, so it can be reviewed afterwards.
   Do not modify `fp_query` or `wsp_submit`.
6. Keep a running `NOTES.md` in this directory: each hypothesis, what you checked, and what you concluded, including dead ends.

## Forward passes
You have **{{BUDGET}}** forward-pass calls for phase A. Each call takes 1 to 256 sequences, each exactly 500 characters of
uppercase A/C/G/T, and returns one logit per sequence. A rejected call (bad input) is not charged.

    ./fp_query seqs.txt            # one sequence per line (FASTA also accepted); prints "index<TAB>logit"
    ./fp_query --json -o out.json seqs.txt
    ./fp_query --status            # free: current phase and remaining calls

## Phases
{{PHASES}}
