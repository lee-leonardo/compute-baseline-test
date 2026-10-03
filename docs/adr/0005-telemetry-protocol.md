# ADR 0005: Shared telemetry protocol

Status: accepted

## Decision

The harness consumes `profiling.Profiler`; it never queries vendor counters directly.
Collectors implement `open`, `sample`, and `close`, expose metric specifications,
and return observations with a numeric value or an explicit unavailable reason.
Collectors observe hardware and must never select a different execution device.

`telemetry-v1` records source, unit, scope, successful and failed sample counts,
availability, mean and sampled peak. Missing observations remain null; partial
availability retains valid samples and the failure reason. Legacy resource fields
remain for report consumers. Sampling covers the whole job, including setup,
warmup, timed work and evaluation, and is distinct from synchronized trial timing.
A sampled peak is not an allocator high-water mark. Utilization is device activity,
not Tensor Core occupancy or a performance speedup.

NVIDIA collection uses the actual CUDA device UUID and independent NVML queries
for utilization, device memory, board power and temperature. Unsupported counters
cannot invalidate other measurements. Apple utilization and AMD GPU telemetry
remain explicitly unavailable until native collectors are implemented and verified.
The contract accepts injected collectors without changing workloads or runtimes.
Apple unified memory must not be labeled dedicated VRAM; estimated Apple power
must not be compared directly with NVIDIA board power.

## Consequences

One module owns collection, sampling and aggregation. Runtime adapters continue
owning execution, synchronization and allocator metrics. The comparison layer
owns equal-work speed and throughput ratios, while capacity comparisons retain
configuration differences and failures. Detailed tracing is a separate diagnostic
run because its overhead can affect measurements.
