# Workload development tranches

The current tranche establishes manifests, run labels, diagnostic NLP phases, failure
retention, and per-trial export. Review that workflow before expanding workloads.

The classification tranche is complete for the benchmark scope: synthetic/labeled CSV
training, held-out checkpoint inference, portable PyTorch/MLX weights, class-distribution
and imbalance-aware metrics, CSV validation, and example sweeps. See the
[classification guide](classification.md) for supported paths and deliberate limits.
Native CPU, Apple MPS, and MLX checks are separate from unverified CUDA configurations.

Classification preparation now explicitly validates or regenerates artifacts before inference.
The first [embedding tranche](embedding.md) implements paired-text contrastive training,
held-out retrieval evaluation, and saved encoder/tokenizer inference on PyTorch.

The [Transformer lifecycle](transformer.md) now supports from-scratch sequence classification,
synthetic endpoint-order data or AG News CSV, held-out quality and portable checkpoints.
It reuses the shared harness and PyTorch adapter. RNN/autoencoder comparisons remain optional
future baselines; they are not part of this implemented Transformer tranche.

Each tranche needs offline correctness fixtures, reset/provenance checks, an explicit
runtime compatibility table, and separate native-device validation. Synthetic fixtures
establish harness correctness; meaningful public data supports learning assessments.
Time-to-quality requires a defined evaluation target and a recorded not-reached outcome;
it should follow reliable checkpoint/evaluation support rather than being inferred from
fixed-step throughput. Visualization and scheduling are downstream consumers, not part
of this tranche.
