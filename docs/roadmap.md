# Workload development tranches

The current tranche establishes manifests, run labels, diagnostic NLP phases, failure
retention, and per-trial export. Review that workflow before expanding workloads.

The first workload tranche now includes [classification training and saved-checkpoint
inference](classification.md) on PyTorch CPU/CUDA/MPS. Review it before extending adapters
or objectives.

1. **Embedding training:** reproducible paired-text data, contrastive objective,
   held-out evaluation, and saved checkpoints reused by embedding inference.
2. **Classification extensions:** evaluate additional data/quality metrics and MLX
   checkpoint support; retain sklearn as a separate CPU reference.
3. **Sequence experiments:** refine the existing Transformer task and evaluate whether
   an RNN or autoencoder baseline adds a useful, matched learning question.

Each tranche needs offline correctness fixtures, reset/provenance checks, an explicit
runtime compatibility table, and separate native-device validation. Synthetic fixtures
establish harness correctness; meaningful public data supports learning assessments.
Time-to-quality requires a defined evaluation target and a recorded not-reached outcome;
it should follow reliable checkpoint/evaluation support rather than being inferred from
fixed-step throughput. Visualization and scheduling are downstream consumers, not part
of this tranche.
