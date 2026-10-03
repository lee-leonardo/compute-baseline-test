# CLI reference and debugging

`runtime-bench --help` is the command index. Each command owns its detailed help:

| Command | Purpose |
| --- | --- |
| `classification train` | Train a tabular MLP, evaluate it, and save a checkpoint |
| `classification infer` | Benchmark saved weights on the same held-out split |
| `run TASK` | Advanced access to individual workloads and legacy examples |
| `suite MANIFEST` | Expand a TOML experiment into isolated sequential runs |
| `fetch ASSET` | Download news or pinned pretrained model assets before measurement |
| `compare BASELINE CANDIDATE` | Compare compatible reports; `--capacity` shows unequal work/failures |
| `export REPORT... --output FILE` | Flatten report files into one CSV row per trial |

Run commands with the same selected uv extras each time. The MLP classification
workflow needs only the base PyTorch installation (`--extra cpu` or `--extra cuda`).
`pipeline` is needed for sklearn and pretrained NLP. An installed extra does not
select a runtime or device.

```sh
uv run --locked --extra cpu runtime-bench --help
uv run --locked --extra cpu runtime-bench classification train --help
uv run --locked --extra cpu runtime-bench classification infer --help
uv run --locked --extra cpu runtime-bench run --help
uv run --locked --extra cpu runtime-bench suite --help
```

## Classification options

The focused classification commands omit NLP, MLX, and sequence-model flags.

| Option | Meaning |
| --- | --- |
| `--device cpu/cuda/mps/auto` | PyTorch execution device; explicit unavailable devices fail |
| `--checkpoint PATH` | Train: new output file (default timestamped file in results); infer: required input |
| `--data PATH` | Labeled numeric-feature CSV; omit for reproducible synthetic data |
| `--target NAME`, `--features A,B` | Required for CSV training; restored from checkpoint for inference |
| `--width N` | Training hidden width; inference restores the checkpoint architecture |
| `--seed N` | Training initialization and split seed; inference restores it |
| `--batch N`, `--steps N` | Examples per batch and measured batches per trial |
| `--warmup N`, `--repeats N` | Discarded warmup batches and independently reset trials (defaults 5, 3) |
| `--precision fp32/fp16/bf16` | FP32 default; mixed precision requires compatible CUDA |
| `--threads N` | PyTorch CPU thread budget; defaults to physical cores |
| `--output PATH` | Reports directory, default `results` |
| `--node LABEL`, `--condition LABEL` | Public-safe labels for comparison/export |
| `--synthetic-data` | Mark a supplied training CSV as synthetic; persisted in checkpoint |

Defaults for batch, steps, width, and seed are 32, 50, 128, and 42. The synthetic
classification fixture contains 1,024 rows and 16 numeric features, with 800 training
and 224 held-out rows; it does not use `--limit`. CSV uses a seeded 80/20 row split.

`BENCH_DEVICE` and `BENCH_OUTPUT` supply device/output defaults when flags are omitted.
Explicit flags take precedence. Use explicit devices when comparing nodes. MPS CPU
fallback must be disabled; the harness rejects `PYTORCH_ENABLE_MPS_FALLBACK=1`.

## Compatibility

Existing `runtime-bench smoke ...`, `runtime-bench infer ...` and the
`runtime-bench-suite`, `runtime-bench-fetch`, `runtime-bench-compare`, and
`runtime-bench-export` entry points still work. New documentation prefers the single
`runtime-bench` command. The old `classify` task remains a sklearn full-fit CPU baseline;
it is distinct from the new `classification` MLP lifecycle.

Manifests still use flat fields: `task = "classification"`, `mode = "train"` or
`"infer"`, and `checkpoint = "results/model.pt"`. Use `runtime = "torch"`.
Inference restores width, seed, feature selection and synthetic labeling from the
checkpoint, even if those fields appear in a flat manifest. Focused inference help
omits those overrides entirely. Legacy examples still allow initialized-weight inference;
use `classification infer` when you need trained weights.

## Debugging a run

1. Inspect the relevant `--help`, or expand a manifest with `suite ... --dry-run`.
2. Read the effective `config` in the JSON report, including resolved defaults.
3. For a failure, read `error`, `reason`, and requested device. Suite failures also
   retain a per-case `console.log` and an incremental `index.json`.
4. Check dataset/model fingerprints and protocol before comparing timings.

Argument parsing errors exit 2 before a report exists. Expected workload errors write
a failed report and exit 2. A suite continues after a failed case and exits 1 if any
case failed; invalid manifests exit 2. An OS-killed process may have only a suite exit
code/log and no report. Unhandled programming errors retain their Python traceback.

For contributors, `cli.py` owns parsing, defaults, validation, dispatch and report
output. `micro.py` owns learning trials and evaluation. `classification.py` owns the
checkpoint/data contract; `torch_runtime.py` owns device operations. `experiments.py`
supervises processes. Follow those boundaries when investigating a bug rather than
adding model-specific behavior to argument dispatch.
