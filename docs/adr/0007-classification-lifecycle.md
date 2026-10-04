# ADR 0007: Focused CLI and classification checkpoints

## Decision

Expose a single command index with focused `classification train` and `classification
infer` help. Retain flat workload commands and existing scripts for compatibility.
Move micro trial orchestration out of cli.py into micro.py. Manifests continue to use
the flat Namespace contract; checkpoint inference restores model-owned options.

Add a PyTorch tabular classification lifecycle using existing deterministic MLP data
builders. Save CPU state tensors and primitive metadata, loaded with weights_only=True.
Restore preprocessing and the exact evaluation split. Fail on changed CSV data, missing
artifacts, incompatible runtimes, overwritten output paths, or mismatched weight hashes.
Use classification-v1 for its training and prediction timing contract.

## Consequences

Matched inference compares the same trained weights across nodes. Training resets after
warmup and between trials and saves the final measured trial. Prediction timing excludes
loss computation; held-out scoring remains outside timed loops. Checkpoint writing is
outside reported workload wall time but inside resource sampling. No optimizer resume,
unlabeled serving, or MLX checkpoint conversion is implied. The sklearn baseline and
legacy initialized-weight learning examples remain separate tasks.

MLX checkpoint support was subsequently added by [ADR 0008](0008-classification-extension.md).
The original PyTorch-only restriction above records the initial tranche's scope.
