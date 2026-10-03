# Measurement protocols

The primary workflow is the [hardware workload ladder](ladder.md): test Apple Silicon,
NVIDIA RTX and CPU configurations such as AMD/Intel using representative tasks. It uses
each machine's available resources, real pretrained models, elapsed time and workload limits. The details below apply to the original coffee/news/stateful/smoke micro
workloads, whose compute timings deliberately isolate different costs. Reports label
`operational-v1` and `micro-v2`; do not mix their timing meanings.

## Optional micro workload protocol

Use the same git revision, locked dependencies, dataset SHA-256, feature list, target,
seed, mode, batch, width, sequence length, warmup, steps, repeats, and CPU thread count.
Close competing workloads, record power settings in experiment notes, and run one process
at a time. Repeat fresh-process runs as well as in-process trials. Maintain identical
batch/sequence sizes when comparing devices; conduct capacity sweeps separately.

Start with FP32 CPU and GPU runs. Then compare CUDA FP16/BF16 against its own FP32 result;
those are separate precision experiments. Do not attribute a backend-level result solely
to matrix hardware. Backend kernels, bandwidth, memory layout, drivers, precision, CPU,
and software all contribute. Apple neural-engine use is outside this harness.

The timer synchronizes CUDA/MPS before and after each measured operation. Warmup is
excluded, and weights/state are reset afterward. Compute timing covers forward + loss
and, in train mode, backward + AdamW update. Transfer timing covers host batch → runtime-ready array preparation
and synchronization: CPU-to-device copies for PyTorch, array materialization/conversion
in unified memory for MLX. These operations have different physical transfer semantics. Loop throughput additionally includes CPU batch indexing,
synchronization, Python overhead and memory sampling. Setup covers data preprocessing and
model construction, not uv installation or Python/import startup. Wall time includes warmup,
all trials, and evaluation, but excludes report writing and hardware discovery.

Reports include per-trial median and nearest-rank p95 latency, samples/second,
transfer time, final loss, and evaluation accuracy/cross-entropy. For the stateful task,
one sample is a chunk of `length` tokens; multiply sample throughput by length for tokens/second.
Stateful loss includes an initial chunk without context; evaluation skips that first chunk.
Stateful quality uses the same generated stream and is a diagnostic, not held-out generalization.
Coffee/news quality uses the held-out row split from the last trial. Infer mode uses randomly
initialized weights; it characterizes architecture execution, not a trained deployment model.
Legacy micro examples do not save/load checkpoints. The new classification lifecycle
saves trained weights and restores them for held-out inference; see [classification](classification.md).
Time to a quality threshold is not measured.

GPU warmup may fail due to unsupported operators or OOM; this writes a failed report
and is not a valid performance result. CPU fallback on MPS is rejected when enabled. Inspect
quality across runs: faster execution with degraded/nonfinite outputs is not an advantage.
Nonfinite training losses are rejected. Seeded initialization improves repeatability but does
not promise bitwise equality across backends.

Memory fields have different meanings: per-trial process RSS is sampled at step boundaries; whole-job RSS is also sampled periodically, CUDA
memory is the allocator peak, and MPS memory is a current allocation snapshot. MPS allocation
is not a peak and cannot be summed with RAM to estimate total unified-memory consumption.
Shared memory is reported for the selected Apple MPS backend; unknown fields remain unknown.
Hardware discovery is best effort and never guesses missing GPU capacity.

For matching trials compute `speedup = baseline median latency / candidate median latency`
and `throughput gain = candidate samples/s / baseline samples/s`. Retain each trial's
spread rather than publishing only a best run. Small coffee tables can be CPU/launch-bound;
a GPU's advantage may emerge with larger batches, widths, and sequence lengths. Use a
matrix (batch 1/32/128, length 32/128/512, width 128/256) with the same matrix on each device,
and label capacity failures. These measurements can establish speed and capacity advantages;
these are measurements for the tested configuration, not a universal ranking.
Energy efficiency requires separately measured power.

Timing references: [CUDA synchronization](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html),
[MPS synchronization](https://docs.pytorch.org/docs/stable/generated/torch.mps.synchronize.html).
Installation follows [uv's PyTorch index guidance](https://docs.astral.sh/uv/guides/integration/pytorch/).

Compare two matching JSON reports:

```sh
uv run --locked --extra cpu runtime-bench-compare results/baseline.json results/candidate.json
```

The comparison rejects different workload settings, dataset fingerprints, or git revisions.
It prints candidate/baseline throughput gain and baseline/candidate latency speedup,
plus quality changes. Dirty checkouts require manual source verification. Software/backend
versions are retained in each report and must be reviewed; ratios do not establish causality.

For MLX, use matching coffee/smoke FP32 configurations. Reports include exact canonical
initial-weight fingerprints. Cross-runtime comparisons include software/kernel differences.
MLX peak allocation, active allocation and cache bytes are allocator-specific and are not
total unified memory. MLX execution manages its own threads. See the [MLX example](mlx-example.md).

Caught configuration/runtime failures now produce a failed text/JSON report and nonzero
exit code. `--capacity` displays unequal configurations or failures side by side without
a speedup claim. The sampler records process/host resource context; see [ladder metric
semantics](ladder.md). Original reports without a protocol field use `micro-v1`.
