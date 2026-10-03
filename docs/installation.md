# Installation and task reference

For the simplified command index and checkpoint workflow, start with the
[CLI guide](cli.md) and [classification walkthrough](classification.md). The commands
below remain supported as the advanced/legacy task interface.

A small uv project to **test hardware differences using representative ML workloads**.
Compare Apple Silicon, NVIDIA RTX systems, and CPU configurations such as AMD and Intel
by measuring runtime, throughput, memory use, and the workloads each system can fit.
Architecture: **harness → runtime adapter → workload → report**.

| Step | Command/task | Representative work |
| --- | --- | --- |
| 1 | `classify` | sklearn CPU classification, synthetic or coffee CSV |
| 2 | `embeddings` | Text cleanup/tokenization + pretrained MiniLM embeddings |
| 3 | `infer` | Pretrained DistilBERT/BERT fill-mask inference |
| 4 | `finetune` | Full-model fine-tuning on news labels |

Start with the same data/model/work budget, then increase batch, sequence length or model
size to discover capability boundaries. Use each computer's real resources. The harness
imposes no RAM ceiling and needs no extra infrastructure. Results describe the tested
hardware, runtime and settings together; they do not establish a universal hardware ranking.

## Task and runtime compatibility

For the complete model × task and runtime × device matrices, see
[choosing packages, model assets, and runtimes](models-and-runtimes.md).
BERT is optional; MiniLM supplies embeddings, while DistilBERT is the default for
pretrained inference and fine-tuning.

Installing an extra makes its packages available; it does **not** select that runtime.
`--runtime` defaults to `torch`. Select MLX explicitly with `--runtime mlx`.

| Task | sklearn CPU | PyTorch CPU / CUDA / MPS | MLX Apple CPU / GPU | Extra needed for the task |
| --- | --- | --- | --- | --- |
| `classify` | Yes — always CPU | No — `--runtime torch` routes this task to sklearn | No | `pipeline` |
| `embeddings` | No | Yes | No | `pipeline` |
| `infer` | No | Yes | No | `pipeline` |
| `finetune` | No | Yes | No | `pipeline` |
| `coffee` | No | Yes | Yes | `mlx` when selecting MLX |
| `smoke` | No | Yes | Yes | `mlx` when selecting MLX |
| `news` | No | Yes | No | No additional task extra |
| `stateful` | No | Yes | No | No additional task extra |

For PyTorch, use `--runtime torch` with `--device cpu`, `cuda`, or `mps`.
Install `--extra cuda` for NVIDIA CUDA, or `--extra cpu` for CPU/MPS; the `cpu` extra
includes the Apple MPS-capable wheel on macOS. For MLX, install `--extra mlx` and use
`--runtime mlx --device gpu` or `--device cpu` on native Apple Silicon macOS.
MLX examples support FP32 only; PyTorch mixed precision currently requires CUDA.
An implemented path still requires compatible hardware, drivers and native verification.

`classify` is the sklearn CPU baseline even with `--device auto`. There is no
`--runtime sklearn` option. Installing `--extra mlx` does not change that task into an
MLX neural network. For a classification workload implemented in MLX, choose `smoke`
(synthetic MLP) or `coffee` (MLP with explicit CSV target/features).

```sh
# Installing MLX does not change classify: this still runs sklearn on CPU.
uv run --locked --extra mlx --extra pipeline runtime-bench classify --device auto

# Explicitly select an implemented MLX task and the Apple GPU.
uv run --locked --extra mlx runtime-bench smoke --runtime mlx --device gpu
```

`auto` is task/runtime-specific: sklearn `classify` uses CPU; PyTorch selects CUDA,
then MPS, then CPU; MLX selects the Apple GPU and fails if it is unavailable.

## Hardware coverage

| Configuration | Current execution path |
| --- | --- |
| Apple Silicon GPU | PyTorch MPS; optional MLX for coffee/smoke MLPs |
| NVIDIA RTX GPU | PyTorch CUDA; optional NVIDIA resource counters |
| AMD, Intel or Apple CPU | CPU workloads with the available packages on that host |
| AMD GPU | Future extension; ROCm is not configured or validated in this repo |
| Other GPUs/accelerators | Require an explicitly implemented and verified runtime adapter |

AMD CPU support is separate from AMD GPU support. The CUDA extra is NVIDIA-specific.
See [hardware coverage and comparison](hardware.md) for supported paths and limits.

## Install and run, step by step

Run these commands from the cloned repository directory. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) first. uv manages the
project's pinned Python 3.12.12, creates `.venv`, and installs the packages from
`uv.lock`; you do not need to install Python packages with pip separately.

There are three different kinds of dependencies:

| Dependency | How it is obtained | Why it is needed |
| --- | --- | --- |
| Python packages | `uv sync` | PyTorch executes neural networks; `pipeline` adds sklearn and Transformers; the CUDA extra adds NVIDIA telemetry |
| Dataset | `runtime-bench-fetch news` or your own local CSV | Supplies the text/examples the workload processes |
| Pretrained model assets | `runtime-bench-fetch minilm`, `distilbert`, or `bert` | Supplies learned weights, tokenizer vocabulary and configuration; these are separate from the Transformers Python package |

**BERT is a model, not another Python package to install.** Installing Transformers
provides the code that loads models; it does not download their weights. You only
need the model used by your chosen workload. Full BERT is an optional larger test;
start with MiniLM and DistilBERT.

### 1. Install the packages for your hardware

Choose one of these commands:

```sh
# Apple Silicon (CPU or PyTorch MPS GPU), or any supported CPU host
uv sync --locked --extra cpu --extra pipeline

# NVIDIA RTX on Linux/Windows, including WSL with CUDA exposed by the host driver
uv sync --locked --extra cuda --extra pipeline
```

`sync` prepares the environment. `--locked` requires the committed dependency pins.
`--extra pipeline` enables the four representative tasks listed above. `--extra cpu`
and `--extra cuda` select the PyTorch package source and are mutually exclusive;
use the same extras on subsequent commands. On macOS, the CPU extra includes MPS
support. The CUDA packages do not install the host NVIDIA graphics driver; a working
CUDA-capable driver is required before GPU execution. No Docker setup is required.

### 2. Verify the harness without downloading data or models

```sh
uv run --locked --extra cpu --extra pipeline runtime-bench classify --device cpu
```

`uv run` executes a command in the project environment. This task fits a CPU
classifier on generated, explicitly labeled synthetic data and writes a report.
It checks the basic installation; it does not test your GPU. On a CUDA installation,
replace `--extra cpu` with `--extra cuda` even for this CPU task.

### 3. Download the inputs for the NLP workloads

The following examples use the Apple/CPU installation. On NVIDIA, replace
`--extra cpu` with `--extra cuda` in every command.

```sh
# Text and labels shared by embeddings, inference and fine-tuning
uv run --locked --extra cpu --extra pipeline runtime-bench-fetch news

# Small pretrained encoder for the embeddings workload
uv run --locked --extra cpu --extra pipeline runtime-bench-fetch minilm

# Pretrained encoder for inference and fine-tuning
uv run --locked --extra cpu --extra pipeline runtime-bench-fetch distilbert
```

Fetch commands require internet access. The news CSV is saved as
`data/ag-news.csv`; pinned model assets are cached under `data/models/`. These
local files are ignored by Git. Fetch each input once per machine/cache; measured
runs load local assets and do not download missing models. Downloads are deliberately
separate so internet speed is excluded from hardware execution measurements.

| Workload | Local inputs required | What it does |
| --- | --- | --- |
| `classify` | None for its synthetic default | Fits a CPU classifier |
| `embeddings --model minilm` | News CSV + MiniLM | Converts text into numeric vectors |
| `infer --model distilbert` | News CSV + DistilBERT | Predicts masked tokens without updating model weights |
| `finetune --model distilbert` | News CSV + DistilBERT | Updates pretrained weights and a new four-class head using news labels |
| `infer` / `finetune --model bert` | News CSV + BERT | Runs the corresponding workload with the larger encoder |
| `smoke` / `stateful` | None | Runs synthetic learning examples |
| `news` | News CSV; no pretrained model | Trains a small Transformer from scratch |
| `coffee` | Your coffee CSV, target and numeric feature names | Trains a small MLP |

### 4. Run the NLP workloads on an explicit device

These examples use the Apple GPU through PyTorch MPS:

```sh
uv run --locked --extra cpu --extra pipeline runtime-bench embeddings \
  --device mps --data data/ag-news.csv --model minilm --batch 8 --length 128

uv run --locked --extra cpu --extra pipeline runtime-bench infer \
  --device mps --data data/ag-news.csv --model distilbert --batch 8 --length 128

uv run --locked --extra cpu --extra pipeline runtime-bench finetune \
  --device mps --data data/ag-news.csv --model distilbert \
  --batch 8 --length 128 --steps 100 --repeats 3
```

For CPU execution, use `--device cpu`. For NVIDIA, use `--extra cuda` and
`--device cuda`. An explicitly requested unavailable device fails; it is not
silently replaced by CPU. `--batch` sets examples processed together, `--length`
sets the padded token sequence length, `--steps` sets batches executed per trial,
and `--repeats` sets the number of measured trials. Start small, then increase
these settings to explore speed and memory limits. Keep settings identical when
claiming a speedup between devices.

To try the larger BERT model, download it first, then change the model argument:

```sh
uv run --locked --extra cpu --extra pipeline runtime-bench-fetch bert
uv run --locked --extra cpu --extra pipeline runtime-bench infer \
  --device mps --data data/ag-news.csv --model bert --batch 8 --length 128
```

A missing-model error means the selected model is not in the expected local cache;
run its fetch command first. If you customize the location, use `--cache PATH` on
fetch and the same `--model-cache PATH` on execution. MLX is optional and currently
supports `smoke` and `coffee`, not these pretrained NLP tasks; see the compatibility
table above and the [MLX walkthrough](mlx-example.md).

Follow the [workload ladder](ladder.md) for scaling and report comparisons.
For real coffee data, obtain the [Coffee Quality & Pricing Benchmark](https://www.kaggle.com/datasets/razanihababdellatif/coffee-quality-and-pricing-benchmark)
CSV separately and review its categorical target and numeric predictors before running.

Reports are a small `.txt` plus a `.json` sidecar in ignored `results/`: wall time,
throughput, sampled RAM/CPU/GPU use, runtime allocation metrics, hardware, quality and
provenance. Caught OOM/error runs retain failed reports and exit nonzero. Apple GPU
utilization and other unavailable counters are explicitly unknown. Performance is measured,
never inferred from a product name. See the [shared telemetry protocol](profiling.md)
for counter coverage and how to interpret speed, throughput and capacity evidence.

Optional: copy `.env.example` to `.env` and run `uv run --env-file .env ...`.
The harness reads process defaults; gitignore treats local env files as opaque files and
never derives ignore rules from env keys or values. Keep credentials out of tracked files.

## Learning examples

The original `coffee` MLP, `news` Transformer trained from scratch, `stateful` streaming
memory Transformer and `smoke` classifier remain available. These use a separate micro
measurement protocol. The [MLX example](mlx-example.md) runs the same coffee/smoke MLPs
with matched initialization on Apple CPU/GPU; it is a useful variant alongside PyTorch MPS.

Read [measurement details](benchmarking.md), [architecture](architecture.md),
and the [hardware scope decision](adr/0004-hardware-comparison-scope.md).

## Development

```sh
uv sync --locked --extra cpu --extra pipeline --group dev
uv run --locked --extra cpu --extra pipeline pytest
uv run --locked --extra cpu --extra pipeline ruff check .
uv run --locked --extra cpu --extra pipeline ruff format --check .
```

Pytest does not select a runtime from installed extras either:

| Test invocation | What runs |
| --- | --- |
| `pytest` with `--extra pipeline` | CPU harness and offline tiny-model pipeline checks; native MLX tests are skipped |
| `pytest --run-mlx` with `--extra pipeline --extra mlx` | The same checks plus native Apple MLX GPU parity/reset/report tests |

To run the native MLX checks on Apple Silicon:

```sh
uv run --locked --extra cpu --extra pipeline --extra mlx pytest --run-mlx
```

Adding `--extra mlx` without `--run-mlx` leaves native MLX tests skipped.
The pytest suite does not automatically run native CUDA or MPS workloads.
CPU/tiny local model tests run offline. Native CUDA/MPS/MLX verification is reported separately.
Agent contribution rules live in [AGENTS.md](../AGENTS.md).
