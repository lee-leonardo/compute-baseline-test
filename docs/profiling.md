# GPU telemetry and performance evidence

The shared profiler is `src/runtime_bench/profiling/`. The harness supplies configuration
and consumes its summary; vendor APIs stay inside collectors. A collector exposes
metric specifications and implements `open()`, `sample()` and `close()`. Each observation
contains a numeric value or an unavailable reason. Collectors can be injected into
`Profiler(..., collector=...)` for tests and future platform adapters.

The `resources.metrics` report contract is `telemetry-v1`:

| Field | Meaning |
|---|---|
| `unit`, `scope`, `source` | What was measured, where, and by which API |
| `status`, `reason` | Available, partial, or unavailable; missing is never zero |
| `samples`, `failed_samples` | Successful and unsuccessful observations |
| `mean`, `sampled_peak` | Aggregates of successful observations |

Process RAM/CPU and available host memory use the same metric metadata; available
host memory reports a sampled minimum. Legacy fields remain in the resource summary. Runtime
allocator measurements remain separate from device-wide sampled memory. GPU
sampling spans setup, warmup, execution and evaluation; its mean is a whole-job
activity measure, not utilization exclusively during timed iterations.

| Collector | Implemented counters | Limitation |
|---|---|---|
| NVIDIA NVML | GPU activity, device used memory, board watts, temperature | Device-wide; optional CUDA extra; native verification required |
| Apple MPS / MLX | Runtime allocation metrics elsewhere in the report | GPU activity collector pending; utilization remains null |
| AMD GPU | None | Runtime and collector pending |

To explain hardware value, collect three forms of evidence:

1. Equal-work runs: identical data/model fingerprints, batch, sequence length,
   precision and training steps. Use repeated synchronized timings and report
   both iteration throughput and complete job duration, with quality checks.
2. Same-machine CPU versus GPU: estimates the benefit of the accelerator and its
   backend. Across-machine comparisons measure the complete hardware/software system.
3. Capacity runs: increase batches, sequence lengths or model sizes and retain
   failures. Record what becomes feasible and how long it takes.

Use `runtime-bench-compare baseline.json candidate.json` for compatible equal-work
results, or `--capacity` for differing configurations. A 10× throughput ratio means
one order of magnitude; activity percentages alone do not imply that ratio.
Report hours saved for a stated number of jobs alongside capacity gains to explain
replacement value. No purchase value is inferred from GPU branding.

Keep deep traces separate from timing runs. Future Apple collectors should use
an explicitly enabled OS source with documented privileges and units; they must
not substitute CPU counters for GPU activity. See [ADR 0005](adr/0005-telemetry-protocol.md).

## Opt-in NLP phase timing

`--profile diagnostic` separates CPU tokenization, synchronized device transfer, and
synchronized model execution within each timed NLP trial. Standard mode preserves the
original timing behavior. Compare only matching profiling modes. See
[experiment workflow](experiments.md#standard-versus-diagnostic-measurement) for scopes,
limitations, run labels, and per-trial export.

The singular embedding lifecycle also supports diagnostic tokenization/transfer/execution
phases. Its `embedding-v1` report counts pairs during training and anchors during inference;
compare only matching modes and work budgets. Retrieval evaluation is outside timed batches.
