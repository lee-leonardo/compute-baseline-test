# Runtime Bench

A small local ML workload profiler for CPU, NVIDIA CUDA, Apple MPS, and selected
Apple MLX workloads. Run the same experiment on different nodes, repeat it under
normal background load, and inspect execution time, throughput, memory, and quality.
Results describe a workload/runtime/device combination, not a universal hardware score.

Architecture: **harness → runtime adapter → workload → report**. No model server,
cluster, dashboard, or cloud account is required. Downloads happen before measured runs.

## Build a machine performance profile

[`run.sh`](run.sh) is the high-level matrix runner. It validates the selected
hardware profile, obtains only the required local assets before timing, and expands
every supported runtime/device/precision permutation for the requested workload
families. Each lifecycle inference case consumes the checkpoint produced by its
matching training case; all artifacts are isolated in one new result directory.

```sh
# Inspect the exact matrix without changing the environment or downloading assets.
./run.sh --profile cpu --dry-run

# Profile every supported CPU permutation with public-safe context labels.
./run.sh --profile cpu --node cpu-lab-a --condition idle

# Restrict a native CUDA profile to lifecycle tests and keep all supported precisions.
./run.sh --profile cuda --families classification,transformer,embedding --precisions all
```

Use `./run.sh --help` for the work-budget, optional coffee CSV, model-cache, and
diagnostic-NLP timing parameters. `--profile apple` runs PyTorch CPU/MPS and the
supported MLX CPU/GPU paths; `--profile cuda` runs CPU/CUDA paths and tests BF16 only
when the native CUDA device reports support. The script never downgrades a requested
accelerator to CPU. Static TOML suites and comparison/export stay explicit commands,
because they require a user-selected manifest or report paths.

## Train a classifier and reuse it

```sh
uv run --locked --extra cpu runtime-bench classification train --device cpu \
  --checkpoint results/classifier.pt
uv run --locked --extra cpu runtime-bench classification infer --device cpu \
  --checkpoint results/classifier.pt
```

See the [classification walkthrough](docs/classification.md) for CSV inputs and timing,
and the [CLI guide](docs/cli.md) for help, defaults, and debugging.

## Start with an offline experiment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone the repo,
and run from its root:

```sh
uv sync --locked --extra cpu
uv run --locked --extra cpu runtime-bench suite experiments/smoke.toml \
  --node node-a --condition idle
```

This runs two synthetic MLP batch sizes, each with three reset trials. The suite
prints an index path under `results/suite-*/`; each case has logs and text/JSON reports.
Use a public-safe node label. Generated data, local configuration, and results are ignored.

Run the same manifest on another node. Change its explicit device to `mps` or `cuda`
for that device, retaining the workload settings. CUDA installations use `--extra cuda`
instead of `--extra cpu` on every command. For native Apple MLX, add `--extra mlx` and
select `runtime = "mlx"`, `device = "gpu"` for supported tasks.

Repeat with `--condition services-running` while you manage your usual services.
The label records context; the harness does not start services or measure their latency.

## Compare and export

```sh
# Supply actual successful report paths with matching work and source revisions.
uv run --locked --extra cpu runtime-bench compare BASELINE.json CANDIDATE.json

# Capacity outcomes, including failed runs, without a speedup claim.
uv run --locked --extra cpu runtime-bench compare --capacity BASELINE.json CANDIDATE.json

# One row per trial; failures get a row without invented measurements.
# The numeric case directory prevents including suite index.json files.
uv run --locked --extra cpu runtime-bench export \
  results/suite-*/*/*.json --output results/trials.csv
```

Reports retain quality, trial variability, configuration, provenance, and distinct
memory metrics. A matching experiment ID groups equivalent configured work and assets;
it does not certify identical code, software, or learning quality. The comparison
command performs additional compatibility checks. Different precisions and profiling
modes are separate experiments.

## Current workload coverage

| Family | Implemented | Boundary |
| --- | --- | --- |
| Classification | MLP training + saved-checkpoint inference; separate sklearn baseline | PyTorch CPU/CUDA/MPS and MLX CPU/GPU; portable checkpoints and per-class quality |
| NLP / embeddings | Paired-text contrastive training + checkpoint inference; legacy encoding, fill-mask and classifier fine-tuning | PyTorch CPU/CUDA/MPS; held-out retrieval evaluation |
| Sequence learning | From-scratch Transformer training + checkpoint inference on synthetic sequences or news CSV | PyTorch CPU/CUDA/MPS; legacy news/stateful examples retained |

Use `embedding train/infer` for contrastive learning and saved encoder reuse.
The legacy `embeddings` task only computes vectors; `finetune` trains a news classifier. BERT is an optional larger encoder, not a required
Python package. MLX supports `classification`, `smoke`, and `coffee` MLPs in FP32.

## Guides

- [Transformer sequence training and inference](docs/transformer.md)
- [Embedding training and inference](docs/embedding.md)
- [Classification and checkpoint preparation](docs/classification.md)
- [Experiment manifests and profiling workflow](docs/experiments.md)
- [Gated saturation and compute-profile boundaries](docs/saturation.md)
- [Crossover, boundary refinement and representative pipelines](docs/frontiers.md)
- [Installation, downloads, and individual commands](docs/installation.md)
- [Model × task and runtime × device compatibility](docs/models-and-runtimes.md)
- [Timing and telemetry interpretation](docs/profiling.md)
- [Architecture](docs/architecture.md) and [contribution rules](AGENTS.md)
- [Workload development tranches](docs/roadmap.md)

## Development

```sh
uv run --locked --extra cpu --extra pipeline pytest
uv run --locked --extra cpu --extra pipeline ruff check .
uv run --locked --extra cpu --extra pipeline ruff format --check .
# Native Apple MLX verification, separately from CPU checks:
uv run --locked --extra cpu --extra pipeline --extra mlx pytest --run-mlx
```

No automatic CUDA/MPS verification is implied by CPU tests. Tiny-model checks run
offline; full public checkpoints and native devices require separate verification.
