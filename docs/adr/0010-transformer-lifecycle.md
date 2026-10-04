# ADR 0010: From-scratch Transformer sequence classification

## Decision

Add a distinct transformer train/infer lifecycle using a bidirectional padding-aware
encoder, two blocks, four heads and zero dropout. Reuse the micro trial harness,
TorchRuntime, and classification quality aggregation instead of duplicating timing loops.
Keep legacy news/stateful behavior intact. Use transformer-v1 to distinguish measurements.

Provide a reproducible position-dependent synthetic task with unique train/test sequences,
and bounded AG News CSV classification with deterministic word hashing. Reject duplicate
truncated/tokenized rows and missing training classes. Persist model/data settings and
weights in portable weights-only checkpoints; inference restores the original split and
architecture and validates data/weight fingerprints.

## Consequences

The workload exercises attention/sequence capacity without pretrained downloads. It measures
sequence classification, not autoregressive decoding. Padding is excluded from attention
keys and pooling. Training-length sweeps change the model/data problem; inference restores
checkpoint length. Report padded-position throughput with an explicit definition. No RNN
baseline or MLX adapter is silently substituted. Native validation remains separate from
CPU correctness tests; no hardware performance claims follow from synthetic task quality.
