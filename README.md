# Runtime Bench

A small local ML workload profiler for CPU, NVIDIA CUDA, Apple MPS, and selected
Apple MLX workloads. Run the same experiment on different nodes, repeat it under
normal background load, and inspect execution time, throughput, memory, and quality.
Results describe a workload/runtime/device combination, not a universal hardware score.

Architecture: **harness → runtime adapter → workload → report**. No model server,
cluster, dashboard, or cloud account is required. Downloads happen before measured runs.

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
| Classification | MLP training + saved-checkpoint inference; separate sklearn baseline | Checkpoint lifecycle supports PyTorch CPU/CUDA/MPS; labeled held-out evaluation |
| NLP / embeddings | Pretrained embedding inference; fill-mask inference; classification fine-tuning | Embedding training is not implemented |
| Sequence learning | Small news Transformer; synthetic stateful memory Transformer | Learning examples, not pretrained language-model serving |

The `embeddings` task computes vectors. `finetune` trains a news classifier; it is
not contrastive embedding training. BERT is an optional larger encoder, not a required
Python package. MLX currently supports only the `smoke` and `coffee` MLPs in FP32.

## Guides

- [Experiment manifests and profiling workflow](docs/experiments.md)
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
