# Embeddings: paired-text training and checkpoint inference

`embedding train` fine-tunes a pretrained encoder with a contrastive objective.
`embedding infer` reuses the saved encoder and tokenizer to benchmark held-out text
encoding. This is distinct from news classification fine-tuning and from the legacy
`embeddings` task, which only runs an existing pretrained encoder.

The first implementation supports PyTorch CPU, CUDA and MPS. Install `pipeline` in
addition to your hardware extra. MLX embedding training is not implemented. No model
server is involved; all measured model loading is local with remote code disabled.

## Prepare model and data

```sh
uv sync --locked --extra cpu --extra pipeline
uv run --locked --extra cpu --extra pipeline runtime-bench fetch minilm
uv run --locked --extra cpu --extra pipeline runtime-bench fetch news
uv run --locked --extra cpu --extra pipeline runtime-bench embedding pairs \
  --data data/ag-news.csv --output data/news-pairs.csv --limit 2000
```

Pair preparation is a separate, unmeasured command. It uses each news title as an anchor
and its description as a positive example, ignores news labels, skips empty/identical or
repeated normalized texts, and records source/output SHA-256 values. This is weak
supervision, not a curated semantic-similarity benchmark. Review the data before making
claims about embedding quality. The news mirror is unpinned; copy the same acquired
files across nodes. Existing pair outputs are never overwritten.

Alternatively supply your own UTF-8 CSV with exactly these columns:

```csv
anchor,positive
How do I make coffee?,Instructions for brewing coffee.
How do I plant tomatoes?,A guide to growing tomato plants.
```

That illustrates the format, not a runnable dataset: at least 10 pairs are required.
Provide many varied pairs for meaningful learning. Use `--synthetic-data` with toy data.
Pairs must be nonempty and unique across both columns after NFKC, whitespace trimming,
and case folding for duplicate detection. The encoder sees NFKC/trimmed text. This
conservative validation prevents exact-text leakage across the seeded 80/20 row split
and avoids obvious duplicate negatives. Semantic near duplicates and tokenizer/truncation
collisions can still create false negatives; they require dataset review.

## Train and reuse an encoder

```sh
uv run --locked --extra cpu --extra pipeline runtime-bench embedding train \
  --data data/news-pairs.csv --model minilm --checkpoint results/embedding-model \
  --device cpu --batch 8 --length 64 --steps 50 --repeats 3

uv run --locked --extra cpu --extra pipeline runtime-bench embedding infer \
  --data data/news-pairs.csv --checkpoint results/embedding-model \
  --device cpu --batch 8 --length 64 --steps 50 --repeats 3
```

Use `--device mps` for Apple MPS. For CUDA, replace `--extra cpu` with `--extra cuda`
on all commands and select `--device cuda`. FP32 is the common baseline; CUDA FP16/BF16
follow the adapter's existing device checks. Keep precision separate in comparisons.
MiniLM is the default base encoder; DistilBERT/BERT aliases or local Transformers encoder
directories are also accepted. Fetch the chosen alias before training.

A checkpoint is a directory containing safetensors model weights, configuration, tokenizer
assets, and `benchmark.json`. Training saves the final reset trial after evaluation; it
does not select the best trial or save optimizer state for resuming. Existing destinations
are rejected. Interrupted saves may leave an incomplete directory; inference rejects
missing metadata or changed assets. Choose a new training destination to recover.

Inference requires the same paired file fingerprint and restores the training seed,
row limit and synthetic-data label. Batch size and sequence length remain explicit
benchmark settings. Copy one complete checkpoint directory between nodes for matched
inference; retraining independently produces a different experiment. Neither inference
nor a missing-file error silently downloads weights or retrains a model.

## Objective and trial lifecycle

For a batch of B unique positive pairs, encode both sides with masked mean pooling and
FP32 L2 normalization. The B×B cosine-similarity matrix divided by temperature 0.05 supplies
logits. The aligned diagonal is positive; other batch members act as negatives. Training
minimizes the average anchor-to-positive and positive-to-anchor cross-entropies using
AdamW (learning rate 2e-5, weight decay 0.01) and full-model updates.

Training requires 2 ≤ batch ≤ selected training pairs. Batch size changes both hardware
work and the negative set, so batch sweeps are capacity/behavior studies, not equal-work
speedup comparisons. Every trial reloads original weights and reconstructs optimizer and
RNG state after warmup. Pretrained dropout is retained for training; evaluation disables it.

Inference's timed loop encodes held-out anchors only and never updates weights. Training
throughput counts pairs/sec (two encoded texts per pair); inference counts anchor texts/sec.
The report's `sample_unit` makes the distinction explicit. Do not compare these as the same
operation. Trials include CPU tokenization, device transfer, model execution and synchronization.
Diagnostic mode separately records those three phases with an added transfer synchronization.

The `embedding-v1` protocol is separate from legacy `operational-v1`. Setup/loading, resets,
warmup, timed trials and retrieval evaluation are in workload wall time; Python startup,
optional-library imports, acquisition, checkpoint serialization and report writes are excluded.
Whole-job resource sampling also covers checkpoint writes. Fixed-step throughput does not
establish time to convergence or a target quality.

## Held-out quality

After the final trial, encode both sides of the full held-out split and rank each anchor's
paired positive among every held-out candidate. Report recall@1, mean reciprocal rank,
positive cosine similarity, normalized-vector norm, dimensions, and candidate count.
Exact ties receive the worst rank within the tie, avoiding inflated scores on collapsed
or indistinguishable embeddings. Retrieval is anchor-to-positive; batch negatives are not
used as a substitute for the held-out candidate set.

Retrieval evaluation currently constructs the candidate similarity matrix on CPU and uses
quadratic memory in held-out pair count. The default limit of 2,000 pairs gives about 400
held-out candidates. Candidate count affects difficulty: compare the same data, split,
length, weights and precision. Tiny offline fixtures establish correctness, not NLP quality.

## Suites and debugging

```sh
uv run --locked --extra cpu --extra pipeline runtime-bench suite \
  experiments/embedding-train.toml --node node-a --condition idle
uv run --locked --extra cpu --extra pipeline runtime-bench suite \
  experiments/embedding-infer.toml --node node-a --condition idle
```

The training manifest uses the same destination as the walkthrough, so choose one approach
or update the path before another training run. Training and inference are explicit separate
stages; the suite does not auto-create dependencies. Inspect `dataset`, `quality`, trial
`final_loss`, and checkpoint fingerprints in JSON. Comparison/export support this protocol,
including retrieval-metric changes. Use `embedding train --help` and `embedding infer --help`
for focused flags. Visualization and automatic scheduling remain outside the harness.
