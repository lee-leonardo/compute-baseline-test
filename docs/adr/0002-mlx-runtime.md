# ADR 0002: an optional Apple MLX runtime

Status: Accepted

## Context

PyTorch MPS provides a useful shared-framework baseline, but Apple MLX is another
runtime of interest. Comparing it with CUDA requires equivalent work and deliberate
handling of MLX's lazy execution, optimizer defaults, and unified memory.

## Decision

Add optional MLX 0.32.3 with a uv dependency marker for native Apple Silicon macOS.
Keep one harness and separate `TorchRuntime` / `MLXRuntime` adapters. MLX initially
supports the coffee and smoke MLPs in FP32, on explicit CPU or Metal GPU execution.
MLX `auto` selects GPU and fails if unavailable; it never falls back to CPU.
Other workloads and precision modes fail clearly.

Build the canonical model and data on CPU with the existing PyTorch seed, then copy
its exact weights to MLX. Use exact GELU and identical AdamW hyperparameters, including
explicit MLX bias correction. Fingerprint the initial weights in both reports.
Keep common batches, splitting, training/evaluation budgets, warmup and trial resets.
Force MLX loss, parameters, and optimizer state evaluation inside the timed step and
synchronize before stopping the timer. Use uncompiled Python dispatch in both runtimes.

Reuse text/JSON reports and permit matched cross-runtime comparisons. Label the runtime,
MLX version and allocator metrics explicitly. Array materialization is recorded under
the common preparation/transfer timing field; it is not claimed to be a PCIe copy.

## Consequences

The runtime variant remains optional; existing CPU/CUDA/MPS usage continues to work.
PyTorch stays installed and performs untimed canonical initialization/preprocessing for
MLX, so this is not a minimal MLX-only deployment. MLX manages execution threads itself;
`--threads` controls the shared CPU preparation, not MLX's thread pool.

Native tests check forward outputs and multiple AdamW steps against PyTorch and verify
trial resets and reports. Cross-framework speedups describe the complete hardware/runtime
combination, not isolated hardware or Tensor Core utilization. Numerical equality within
tolerance does not imply identical kernels or identical long training trajectories.
MLX allocator memory is not total process or total unified memory consumption.
