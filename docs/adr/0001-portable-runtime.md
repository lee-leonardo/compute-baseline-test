# ADR 0001: uv and a shared PyTorch runtime

Status: Accepted

## Context

We need a fast-to-start, portable harness that lets an external system clone the repo,
install a reproducible runtime, execute comparable tasks, and collect small reports.
The question is which workload configurations benefit from CUDA versus CPU and Apple GPU execution.

## Decision

Pin Python 3.12.12 and lock packages with uv. Use PyTorch 2.9.1 as the common model API,
explicit CPU/CUDA wheel extras on Linux/Windows, and PyPI MPS wheels on macOS.
CUDA uses the 12.8 index to support a modern GPU generation while retaining older-device
FP32/FP16 experiments. Actual driver and operator support must be validated per host.
No toolkit compilation or custom GPU extensions are required.

Implement coffee MLP classification, AG News Transformer classification, and a stateful
memory-window Transformer delayed-copy task. Keep downloads outside measured runs.
Use explicit numeric coffee predictors because the dataset schema and intended target
must be inspected before choosing labels; never invent a dataset schema.

Use synchronized wall-clock timing, fixed work budgets, reset trials, quality checks,
and separate transfer measurements. Default to FP32 and expose CUDA mixed precision as
separate experiments. Write human-readable text plus versioned JSON provenance.
Ignore local env files as opaque files; ignore rules must not depend on env keys or values.

## Consequences

One code path makes backend comparisons practical, but framework kernel differences
remain part of measured results. A small table may not saturate a GPU. Synthetic stateful
data measures sequence mechanics, not real-world language quality. Mixed precision,
compilation, and other backends need independent validation. No performance/cost conclusion
is valid without matching empirical runs. uv installs do not install host GPU drivers.
