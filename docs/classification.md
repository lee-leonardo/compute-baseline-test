# Classification: train, save, and measure inference

This workflow benchmarks a small tabular MLP through training and held-out inference.
It runs through PyTorch on CPU, CUDA, or MPS, and native Apple MLX on CPU or GPU
(FP32). No pretrained model, sklearn, or server is required. Checkpoints use the same
canonical CPU tensor layout across adapters and can be loaded in either direction.

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

## Apple MLX and portable checkpoints

```sh
uv run --locked --extra cpu --extra mlx runtime-bench classification train \
  --runtime mlx --device gpu --checkpoint results/mlx-classifier.pt
uv run --locked --extra cpu --extra mlx runtime-bench classification infer \
  --runtime torch --device mps --checkpoint results/mlx-classifier.pt
# Also load a PyTorch-trained classifier through MLX:
uv run --locked --extra cpu --extra mlx runtime-bench classification infer \
  --runtime mlx --device gpu --checkpoint results/classifier.pt
```

MLX `--device cpu` and `gpu` share this lifecycle. `auto` selects the Apple GPU and
fails if it is unavailable. MLX requires native Apple Silicon and FP32. The checkpoint
format remains `classification-v1`, including older PyTorch classification checkpoints.
Export copies trained MLX weights into CPU tensors without mutating the reset reference.
Training on different runtimes may produce small numerical differences; compare inference
using one shared artifact when isolating runtime effects.

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
Checkpoints do not contain raw dataset rows. Duplicate headers, repeated/empty feature
names, and rows with missing/extra fields fail with explicit messages. Empty numeric
cells are imputed; malformed numeric text is rejected. Feature-name whitespace is trimmed.
The split is seeded but not stratified: insufficient class coverage in training fails,
and missing evaluation classes are reported explicitly. The report includes class counts
for both splits; inspect these before treating a score as representative.

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
convergence. An epoch-based trainer and automatic tuning are outside this lifecycle.


## Quality beyond accuracy

Evaluation adds `classification-quality-v1` metrics outside timed batches:

- Balanced accuracy: mean recall across classes present in evaluation.
- Macro F1: unweighted average across all known labels, with zero for undefined F1.
- Weighted F1: average weighted by held-out support.
- Per-class support, predicted count, precision, recall, and F1.
- Confusion matrix with true classes in rows and predicted classes in columns.
- Majority baseline accuracy: held-out accuracy from always predicting the training
  split's most frequent class (ties use the first label), plus accuracy above that baseline.

Undefined precision/recall are null. Missing held-out classes are listed explicitly;
balanced accuracy can look perfect while untested classes remain. Macro F1 includes
those known but absent labels with zero F1. These conventions are recorded and tested;
they are not estimates of statistical confidence. The comparison command includes
balanced-accuracy and F1 changes; old reports without those fields show unavailable.
CSV export preserves all metrics; per-class lists and confusion matrices are JSON cells.

## Reproducible classification suites

```sh
# Each case writes a unique checkpoint; rerunning never reuses an old training artifact.
uv run --locked --extra cpu runtime-bench suite experiments/classification-train.toml \
  --node node-a --condition idle
# Requires results/classifier.pt from the quick start above:
uv run --locked --extra cpu runtime-bench suite experiments/classification-infer.toml \
  --node node-a --condition idle
```

Edit runtime/device for the target node while retaining workload settings. Change the
inference manifest's checkpoint path to compare another artifact. The suite runner does
not infer training/inference dependencies; input checkpoints are explicit. The sklearn
`classify` full-fit baseline remains a separate workload requiring `pipeline`.
