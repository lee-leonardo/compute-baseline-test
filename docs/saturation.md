# Gated hardware saturation

Use `runtime-bench saturate` to explore the operating range of each explicit
runtime/device/precision profile. The existing `suite` command continues to run
independent cases even after failures; saturation runs an **ordered ladder** and
stops escalating a profile at its first failed checkpoint.

```sh
uv run --locked --extra cpu runtime-bench saturate plan experiments/saturation.toml
uv run --locked --extra cpu runtime-bench saturate run experiments/saturation.toml \
  --node node-a --condition idle
uv run --locked --extra cpu runtime-bench saturate status results/saturation-<id>/index.json
```

Choose the committed uv extra for the actual backend (`cuda`, or `cpu` plus `mlx`).
Edit the manifest to include only the profiles intended for this node. Device
`auto` is rejected. Unavailable accelerators stop their profile without CPU fallback.
`plan` validates and displays commands without starting benchmark processes; workload
asset/backend compatibility is still checked by each child before execution.

## Designing the ladder

`experiments/saturation.toml` is a synthetic MLP training example, not a hardware
measurement or a claim that these thresholds suit every machine. Tune its limits
before running. Each stage merges `[defaults]`, `[stages.options]`, then the compute
profile. Stages are executed in their listed order, independently for each profile.
Use separate stages to change batch size, sequence length, model width or step count;
change one dimension at a time to make the resulting boundary interpretable. Stage
settings do not inherit the preceding stage's overrides.

A useful progression is a baseline followed by bounded batch/length/width stages.
Existing flat workload options are available, including transformer training and
local pretrained NLP workloads. Fetch assets beforehand. Training stages must omit
`checkpoint`; lifecycle workloads create unique checkpoints in their stage output
folders. Inference may use an explicitly prepared checkpoint. Profiles own runtime,
device, precision and optional CPU thread count. There are no implicit model, runtime,
precision or device substitutions.

This workflow runs sequential workloads. See [bounded frontier workflows](frontiers.md)
for crossover analysis, one-axis boundary refinement and representative text pipelines.
Concurrent services and throughput plateau detection remain outside this scope.

## Checkpoints and live guards

`[limits]` requires positive `timeout_seconds` and `min_host_available_bytes`.
Optional `max_process_rss_bytes` bounds the child process's resident memory. The parent
polls every 100 ms and kills/waits for the child when a guard is crossed. The timeout
includes interpreter startup, setup, warmup, measurement, evaluation and reporting.
Host memory is also checked before launch. These are sampled, best-effort controls,
not OS resource reservations; a sudden allocation can fail between polls. The RSS
guard covers the benchmark child, not a process tree. Current workloads execute their
compute in that child. Device-memory gates are evaluated after a stage, not live.

`[gates]` supports these independent report measurements:

| Gate | Stop condition (inclusive) | Measurement scope |
| --- | --- | --- |
| `wall_seconds` | >= threshold | Workload report wall time, excluding interpreter startup |
| `batch_p95_ms` | >= threshold | Worst reported trial's end-to-end request p95 |
| `compute_p95_ms` | >= threshold | Worst reported trial's synchronized compute p95 |
| `process_rss_bytes` | >= threshold | Sampled whole-job process RSS peak |
| `host_available_bytes` | <= threshold | Sampled whole-host available RAM minimum |
| `host_swap_growth_bytes` | >= threshold | Whole-host swap-used change over the job |
| `cuda_allocated_bytes` | >= threshold | Maximum trial CUDA allocator peak |
| `gpu_device_used_bytes` | >= threshold | Sampled whole-device use, including other processes |

All configured gates must pass. Missing, null or non-finite required metrics stop
escalation as insufficient evidence. Only configure metrics the workload/backend
reports; operational workloads may not expose micro-workload compute p95. Memory
measurements are never added together or relabeled as VRAM. High memory pressure can
reflect other processes; hold node conditions consistent when interpreting results.
Add a `[profiles.gates]` table after an individual profile to extend or override the
common gates for that profile, for example `cuda_allocated_bytes` only on CUDA.

Any nonzero child exit, OOM, malformed/missing report, or live guard also stops the
profile. Other profiles proceed through their own gates. Existing adapter timing,
warmup resets, seeds, train-only preprocessing and data fingerprints remain unchanged.
The parent's supervision adds some overhead outside the workload's timing protocol.

## Reading and reusing evidence

Every stage has a separate directory with original reports and console logs. An
atomically replaced `index.json` records the full plan, commands, decisions, last
passing stage and first stopped stage. Later stages are explicitly skipped. An
interruption preserves a running-stage checkpoint and terminates the active child.

A threshold stop identifies a **policy boundary under these conditions**. An OOM is
capacity evidence for that configuration. A timeout is a censored observation. An
execution error or missing metric does not establish a hardware limit. `exhausted`
means all planned stages passed: the boundary remains undiscovered. Do not infer an
absolute maximum or compare speedups across unequal work from these labels.

To avoid repeating already disqualifying baselines/stages on the same node:

```sh
uv run --locked --extra cpu runtime-bench saturate run experiments/saturation.toml \
  --node node-a --condition idle --prior-index results/saturation-<id>/index.json
```

The complete plan, working directory, node and condition must match. Previously
stopped or interrupted profiles remain blocked with a link and copy of their prior
evidence. Previously successful profiles run a fresh baseline; historical success
never authorizes escalation. This is stop carry-forward, not resume. Node labels are
user assertions, not machine identity verification. Omit `--prior-index` explicitly
to reassess after resolving a failure or changing hardware/conditions. Historical
stops are conservative evidence, not a fresh hardware measurement.

You can also supply an existing standalone report with
`--baseline torch-cpu-fp32=results/baseline.json` (repeat for other profiles). Its
resolved configuration must match the first stage, node and condition, except output
locations and generated training checkpoints. Configured gates and applicable live
memory/time thresholds are evaluated against that report. A failure, threshold crossing
or missing required metric stops the profile before any new compute. Passing evidence
still runs a fresh baseline. The index stores its source path, SHA-256 and decisions.
Keep the source report with the campaign. This import does not validate current hardware
or asset contents and only conservatively disqualifies work; it never qualifies a run.
The original reports remain exportable with `runtime-bench export`;
the campaign index is orchestration evidence, not a workload report.

Exit codes: 0 when every profile exhausts the ladder, 1 when any profile stops, 2 for
invalid input/orchestration errors, and 130 for user interruption. A planned threshold
stop returning 1 is not necessarily a benchmark implementation failure.
