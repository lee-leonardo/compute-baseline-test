# Classification: train, save, and measure inference

This workflow benchmarks a small tabular MLP through training and held-out inference.
It runs through PyTorch on CPU, CUDA, or MPS. No pretrained model, sklearn, or server is
required. MLX checkpoint support is not implemented; existing MLX smoke/coffee examples
remain available separately.

## Synthetic quick start

```sh
uv sync --locked --extra cpu
uv run --locked --extra cpu runtime-bench classification train \
  --device cpu --checkpoint results/classifier.pt --steps 100 --repeats 3
uv run --locked --extra cpu runtime-bench classification infer \
  --device cpu --checkpoint results/classifier.pt --steps 100 --repeats 3
```

The first command fits a synthetic two-class problem and prints report/checkpoint paths.
The second loads the saved weights and benchmarks predictions on held-out examples.
It never trains or updates the checkpoint. Training refuses to overwrite an existing
checkpoint: choose a new path, or omit `--checkpoint` for a timestamped output.

Use `--device mps` on a supported Apple machine. For CUDA, select `--extra cuda` on
all uv commands and `--device cuda`. Copy the *same* checkpoint to each node for matched
inference comparisons. For training comparisons, use identical data, seed, model width,
precision, and work budgets. Do not compare training time with inference time.

## Your own CSV

Use a categorical target and explicit numeric predictors. The target must not be in
features; exclude other columns derived from the target as well.

```sh
uv run --locked --extra cpu runtime-bench classification train \
  --data data/examples.csv --target category --features feature_a,feature_b \
  --device cpu --checkpoint results/csv-classifier.pt
uv run --locked --extra cpu runtime-bench classification infer \
  --data data/examples.csv --checkpoint results/csv-classifier.pt --device cpu
```

CSV parsing, train-only median imputation and standardization occur before steady-state
timing. Checkpoints retain preprocessing statistics, feature order, class labels, data
fingerprint, split seed, architecture settings, and CPU weight tensors. Inference reuses
those statistics; it does not refit preprocessing. CSV inference requires the exact
original file, verified by SHA-256, to reconstruct the same labeled held-out split.
Arbitrary new/unlabeled CSV prediction and deployment serving are outside this benchmark.
Checkpoints do not contain raw dataset rows.

## What is measured

Training batches include forward, cross-entropy, backward and AdamW updates. Inference
compute timing includes only model forward prediction; transfer timing still includes
batch inputs and labels through the shared adapter. The final batch loss and full
held-out accuracy/cross-entropy are computed outside timed inference.

Each trial resets to the same starting weights and optimizer. Warmup work is discarded
by resetting again before measurement. Training saves the final measured trial after
held-out evaluation, never the best trial selected by quality. Saved weights are reusable
for inference, not an optimizer checkpoint for resuming training.

Reports use `classification-v1` so they cannot be pooled with older initialized-weight
micro examples. They retain the existing micro timing fields, explicit timed split,
initial weight fingerprint, and inference checkpoint weight fingerprint. Different
checkpoint weights make inference reports incompatible even if filenames are identical.
File paths do not define experiment identity.

Setup and full-job time include dataset preparation, checkpoint loading for inference,
warmup, resets, measured trials, and evaluation within the workload. Training checkpoint
serialization and report writing are excluded from reported job wall time; whole-job
resource sampling also observes checkpoint serialization. Acquisition and Python process
startup remain outside the reported wall time. Inspect `trials`, `quality`, `dataset`,
and `checkpoint` in the JSON for debugging. CPU/GPU memory metrics keep separate scopes.

The synthetic task checks execution and learning behavior. It is not a representative
quality benchmark for every application. Fixed-step throughput does not measure time to
convergence. An epoch-based trainer, class-imbalance metrics, and automatic tuning are
not part of this initial lifecycle.
