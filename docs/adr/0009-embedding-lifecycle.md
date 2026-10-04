# ADR 0009: Paired-text embedding lifecycle and explicit artifact preparation

## Decision

Finish classification preparation with an explicit idempotent CLI command: validate an
existing artifact or train a missing one. Never hide training inside an inference run.

Introduce a separate embedding task with train/infer modes. Preserve legacy plural
embeddings behavior for compatibility. Use normalized mean-pooled encoders, symmetric
in-batch contrastive cross-entropy, fixed temperature/AdamW settings, and paired CSV data.
Reject empty/repeated text and batch sizes that would duplicate negatives. Optional AG News
title/description preparation is an unmeasured, weakly supervised data transformation.

Keep device operations in EmbeddingRuntime and data/trial/artifact behavior in embedding.py.
Checkpoint directories contain encoder/tokenizer assets plus metadata and fingerprints.
Trial resets reload initial weights and reset optimizer/RNG after warmup. Evaluate retrieval
against all held-out positives with pessimistic tie ranking, outside timed batches.

## Consequences

Embedding training is not classifier fine-tuning. Training throughput counts pairs, while
inference counts anchors. Use embedding-v1 timing and explicit standard/diagnostic profiles;
never pool with legacy protocols. Existing compare/export tools accept the new reports.
PyTorch CPU/CUDA/MPS paths are implemented; native validation must be reported separately.
Full public checkpoints and semantic quality are not established by tiny offline fixtures.
MLX embeddings, resumable optimizer state, automatic quality targets and serving remain
outside this tranche. Checkpoint writing is excluded from workload wall time but included
in resource sampling. Inference neither downloads nor silently regenerates artifacts.
