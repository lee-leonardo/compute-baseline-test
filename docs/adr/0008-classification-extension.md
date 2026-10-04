# ADR 0008: Classification metrics and cross-runtime checkpoints

## Decision

Extend the existing classification-v1 lifecycle to native MLX CPU/GPU in FP32.
Adapters provide materialized predictions, loss scoring, CPU class indices, and export
to a canonical CPU PyTorch model. MLX export copies matching linear-layer layouts and
never changes the initialization reference used for trial reset. Existing checkpoints
remain loadable; no optimizer-state portability is promised.

Compute classification-quality-v1 metrics from held-out confusion counts outside timed
batches. Preserve accuracy/cross-entropy and add per-class results, F1 variants, balanced
accuracy, and a majority baseline selected from training counts. Undefined precision or
recall is null. Balanced accuracy averages supported labels; macro F1 covers all known
labels with zero for undefined F1. Record missing evaluation classes explicitly.

## Consequences

One checkpoint supports matched PyTorch/MLX inference comparisons. Quality differences
remain visible rather than being hidden behind speedup. CSV schema validation fails
before training on duplicate columns/features or malformed rows. Split semantics and
prediction timing are unchanged; richer evaluation contributes to full-job wall time.
Historical source revisions should not be pooled as identical jobs. The sklearn full-fit
baseline remains separate. Embedding objectives are outside this completed tranche.
