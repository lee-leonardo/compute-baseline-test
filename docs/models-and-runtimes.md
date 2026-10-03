# Choose packages, model assets, and an execution runtime

BERT is optional. It is a pretrained Transformer encoder, used here as a larger
alternative to DistilBERT. There is no separate BERT installation step for the
harness, sklearn classification, MLX MLPs, or Transformers trained from scratch.

Three independent choices determine a run:

1. **Packages:** `cpu` or `cuda` selects the PyTorch distribution; `pipeline` adds
   sklearn and Hugging Face Transformers; `mlx` adds the Apple MLX library.
2. **Assets:** fetch the dataset and only the model checkpoint needed by the task.
   A checkpoint contains learned weights, configuration, and tokenizer files.
3. **Execution:** `--runtime` selects the adapter and `--device` selects its device.
   Installing an extra never selects an adapter. Fetching a model never selects a task.

## Model and task permutations

All pretrained tasks require `pipeline`, a local news CSV, and locally cached model
assets. Their runtime is PyTorch on CPU, CUDA, or MPS. They do not run through MLX.

| Checkpoint | `embeddings` | `infer` (fill-mask) | `finetune` (news labels) |
| --- | --- | --- | --- |
| MiniLM | Default; mean-pooled, normalized vectors | Rejected: no pretrained fill-mask head | Encoder plus a newly initialized four-class head |
| DistilBERT | Encoder with mean pooling; not the MiniLM sentence embedding model | Default; pretrained fill-mask head | Default; new four-class head, all weights trained |
| BERT | Encoder with mean pooling | Larger pretrained fill-mask model | Larger encoder plus new four-class head |
| Local Transformers directory | Compatible AutoModel and tokenizer | Complete AutoModelForMaskedLM and tokenizer with a mask token | Compatible AutoModelForSequenceClassification with four labels |

A model loading successfully does not make different model/task pairs equivalent.
Compare the same model, task, data fingerprint, and work budget across machines.
Fine-tuning intentionally initializes a classification head when the checkpoint lacks
one. Fill-mask inference must not initialize missing weights: that would benchmark
random predictions while calling them pretrained inference. The harness rejects that
case, including local encoder-only checkpoints.

Local directories need safetensors weights, configuration and tokenizer assets; loading
is offline with remote code disabled. The selected model must support `--length`.
Local models with incompatible architectures or head dimensions fail at load time.
The offline tests use tiny BERT fixtures; they do not validate every external checkpoint.

## Runtime and device permutations

| Execution subsystem | Tasks | Device arguments | Packages | Precision |
| --- | --- | --- | --- | --- |
| PyTorch classification lifecycle | `classification train/infer` | `cpu`, `cuda`, `mps`, `auto` | `cpu` or `cuda` | FP32; FP16/BF16 on compatible CUDA |
| sklearn | `classify` | `cpu`, `auto` → CPU | `cpu` or `cuda`, plus `pipeline` | FP32 setting only |
| PyTorch learning adapters | `coffee`, `smoke`, `news`, `stateful` | `cpu`, `cuda`, `mps`, `auto` | `cpu` or `cuda` | FP32; FP16/BF16 on compatible CUDA |
| PyTorch Transformers adapter | `embeddings`, `infer`, `finetune` | `cpu`, `cuda`, `mps`, `auto` | `cpu` or `cuda`, plus `pipeline` | FP32; FP16/BF16 on compatible CUDA |
| MLX learning adapter | `coffee`, `smoke` | `cpu`, `gpu`, `auto` → Apple GPU | `cpu` plus `mlx` | FP32 |

`classify` uses the default `--runtime torch` entry point but reports `runtime=sklearn`.
There is no `--runtime sklearn`. MLX requires native Apple Silicon macOS; its `gpu`
argument is not an alias for PyTorch MPS/CUDA. PyTorch `auto` prefers CUDA, then MPS,
then CPU. Explicit unavailable accelerators fail instead of falling back.
On macOS the `cpu` extra retains PyTorch MPS support. `cpu` and `cuda` extras conflict;
`pipeline` and `mlx` may be added together, but this does not extend MLX task support.
PyTorch remains a base dependency even for MLX because it constructs the canonical
reference weights and data used in the matched learning examples.

## Minimal setup paths

For an offline learning check, no BERT, Transformers, or dataset download is needed:

```sh
uv run --locked --extra cpu runtime-bench smoke --device cpu
uv run --locked --extra cpu --extra mlx runtime-bench smoke --runtime mlx --device gpu
```

For pretrained fill-mask inference on CPU:

```sh
uv sync --locked --extra cpu --extra pipeline
uv run --locked --extra cpu --extra pipeline runtime-bench-fetch news
uv run --locked --extra cpu --extra pipeline runtime-bench-fetch distilbert
uv run --locked --extra cpu --extra pipeline runtime-bench infer \
  --device cpu --model distilbert --data data/ag-news.csv
```

For MPS, change the final device to `mps`. For CUDA, use `--extra cuda` in every
command and `--device cuda`. To use BERT, change both the fetched alias and `--model`
to `bert`. For embeddings, fetch `minilm` and select `embeddings --model minilm`.
For fine-tuning, select `finetune --model distilbert`; reuse its existing cached assets.

Custom cache locations must agree: fetch uses `--cache PATH`, execution uses
`--model-cache PATH`. Measured execution does not fetch missing models. Keep the
same extras on subsequent `uv run` commands so environment synchronization retains
needed optional packages.

See the [README walkthrough](installation.md#install-and-run-step-by-step),
[workload ladder](ladder.md), and [adapter boundaries](architecture.md).
