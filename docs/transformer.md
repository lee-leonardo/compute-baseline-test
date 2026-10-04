# Transformer: from-scratch sequence training and inference

`transformer train` fits a small bidirectional encoder classifier and saves its weights.
`transformer infer` measures forward predictions from that checkpoint on the held-out
split. This complements tabular classification and pretrained embedding fine-tuning with
attention and positional learning from scratch. It is not autoregressive text generation,
LLM serving, or a benchmark of peak theoretical TFLOPS.

PyTorch CPU, CUDA and MPS are implemented. No pretrained weights or `pipeline` extra are
needed. MLX is not implemented for this lifecycle; unsupported requests fail explicitly.
The legacy `news` and `stateful` examples remain unchanged and use different protocols.

## Offline quick start

```sh
uv run --locked --extra cpu runtime-bench transformer train \
  --device cpu --width 64 --length 32 --limit 1000 \
  --batch 16 --steps 100 --repeats 3 --checkpoint results/transformer.pt
uv run --locked --extra cpu runtime-bench transformer infer \
  --device cpu --batch 16 --steps 100 --repeats 3 \
  --checkpoint results/transformer.pt
```

The synthetic endpoint-order task generates unique sequences of integers 1–32. Its
binary label states whether the first symbol is greater than the last. Labels alternate
during generation to balance the full dataset; a seeded 80/20 row split supplies disjoint
training and held-out sequences. Interior symbols are distractors. This is a deliberately
simple positional-learning task, not evidence of natural-language understanding.

The generator is independent of batch size, steps and repeats. Its tensor fingerprint,
version, labels and split counts are recorded. Length changes the generated problem;
compare equal lengths when claiming speedup. Synthetic limits are 10–100,000 sequences,
with length at least four. The full selected dataset is prepared on CPU before timing.

## Real labeled text

Use the existing AG News acquisition command or supply the same CSV format:

```sh
uv run --locked --extra cpu runtime-bench fetch news
uv run --locked --extra cpu runtime-bench transformer train \
  --data data/ag-news.csv --limit 2000 --length 64 --width 128 \
  --device cpu --checkpoint results/news-transformer.pt
uv run --locked --extra cpu runtime-bench transformer infer \
  --data data/ag-news.csv --device cpu --checkpoint results/news-transformer.pt
```

CSV rows must be `label,title,description`, without a header; labels are 1–4. The first
`--limit` rows are selected and split with the saved seed. Word hashing deterministically
maps lowercased word tokens to 1–8191, reserving zero for padding. There is no learned
vocabulary to fit on held-out data. Hash collisions are possible; this lightweight tokenizer
is a benchmark choice, not a claim of state-of-the-art NLP preprocessing.

Empty token sequences and duplicate tokenized/truncated inputs fail before splitting.
Deduplicate or select a different dataset if validation reports collisions; changing length
changes what is considered a duplicate. Training must contain every known class. Missing
held-out classes remain explicit in quality metrics. Mark toy CSVs with `--synthetic-data`.

## Architecture and runtime

Two Transformer encoder blocks each use four attention heads and a feed-forward width
of four times `--width`. Width must be divisible by four. The model uses learned positional
embeddings, independently initialized block matrices, zero dropout, masked mean pooling,
and a linear classification head. Attention excludes padding keys, and pooling excludes
padding positions. The model is bidirectional, not causal: classification can inspect the
whole input sequence.

Training uses cross-entropy and the shared AdamW adapter (learning rate 0.001, weight decay
0.01). Every trial starts from the same initialization and optimizer, including a reset
after discarded warmup. Zero dropout avoids an additional source of trial variation.
Inference performs forward prediction without gradient or optimizer updates. Quality scoring
runs separately over the full held-out split after the final trial.

Use `--device mps` on Apple or `--extra cuda --device cuda` on NVIDIA (the extra belongs to
`uv run`, the device to `runtime-bench`). FP32 is the common baseline; the CUDA adapter also
supports FP16/BF16 with its existing checks. Explicit unavailable devices never fall back.

## Artifacts, quality and timing

Training saves the final measured trial, not the best-scoring one, to a new `.pt` artifact.
Existing paths are rejected. Inference requires a saved artifact; a missing checkpoint
prints a training instruction rather than silently creating random weights. Artifacts load
with `weights_only=True` and contain CPU weights, architecture/data settings, and fingerprints;
they do not contain raw input data or resumable optimizer state.

Inference restores width, length, limit, seed and synthetic labeling from the checkpoint.
For text it requires the original file fingerprint. For synthetic data it regenerates and
checks the tensor fingerprint. Batch and timing budgets may change. Copy one checkpoint
between machines to isolate inference runtime differences. Length sweeps belong to training;
a saved checkpoint's positional capacity and evaluation inputs are fixed.

Reports use `transformer-v1`. Micro-style timings separate synchronized transfer and compute,
with loop throughput and median/p95 compute latency. `padded_tokens_per_second` equals
sequence throughput times configured length; it counts padded input positions, not useful
nonpadding text tokens or autoregressively generated tokens. Preprocessing is in setup,
not per-batch timing. There is no extra `--profile diagnostic` mode because transfer/compute
boundaries are already measured. Whole-job time includes setup, resets, warmup and held-out
evaluation. Checkpoint serialization and report writing are excluded from that wall time;
resource sampling also sees checkpoint serialization.

Accuracy, cross-entropy, balanced accuracy, F1, per-class results, confusion counts and the
training-majority baseline use the same definitions as [classification](classification.md).
Inspect quality alongside throughput; fixed-step speed does not demonstrate convergence.

## Suites and comparisons

```sh
uv run --locked --extra cpu runtime-bench suite experiments/transformer-train.toml \
  --node node-a --condition idle
# Requires the quick-start results/transformer.pt artifact:
uv run --locked --extra cpu runtime-bench suite experiments/transformer-infer.toml \
  --node node-a --condition idle
```

The training suite sweeps batch and length with a fresh checkpoint per case. Inference
sweeps batch against one explicit saved artifact. Repeat an unchanged configuration under
user-managed background load with a different condition label. Existing compare/export
commands consume these reports; different tasks/protocols, data or weight fingerprints are
not pooled as equivalent jobs. RNN and autoencoder comparisons remain future optional
baselines rather than implicit substitutes for the requested Transformer implementation.
