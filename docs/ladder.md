# A workload ladder for testing hardware differences

Use the same representative workloads to observe differences between Apple Silicon,
NVIDIA RTX, and CPU configurations such as AMD or Intel. Measure elapsed time,
throughput, memory use and whether each workload completes. Start with the same data,
model, batch size and steps, then increase work to find each system's limits. Let each
computer use its actual available resources.

This is a hardware-testing aid. Results apply to the tested hardware/runtime/settings,
rather than proving that one manufacturer is best for every task. AMD GPU execution is a
future extension; the current CPU path supports AMD CPUs, while CUDA is NVIDIA-specific.
See [hardware coverage](hardware.md).

| Task | Work | What it shows |
| --- | --- | --- |
| `classify` | sklearn histogram gradient boosting | Ordinary CPU/data ML |
| `embeddings` | Unicode cleanup, tokenization, pretrained MiniLM, mean pooling | A CPU → GPU NLP pipeline |
| `infer` | Pretrained DistilBERT or BERT fill-mask requests | Local model latency, throughput and memory |
| `finetune` | Full-model pretrained Transformer training on AG News labels | Backward/optimizer cost and capacity limits |

MLX remains an optional Apple variant of the coffee/smoke MLPs. The original small
Transformer and streaming task remain useful learning examples; they are not the main
operational ladder. See the [MLX walkthrough](mlx-example.md).

## Install and acquire inputs

Use Python 3.12 and uv. On Windows, Ubuntu 24.04 under WSL2 is a straightforward Linux
environment; the host NVIDIA driver must already expose CUDA there. The harness does not
set WSL memory limits, install drivers, or change host configuration. Use a sensible host
resource allocation instead of forcing all computers to have the same RAM.

```sh
# NVIDIA Linux/Windows, including WSL
uv sync --locked --extra cuda --extra pipeline
# Apple/CPU instead
uv sync --locked --extra cpu --extra pipeline
```

Downloads are separate from runtime measurements. Fetch only the models you intend to use:

```sh
uv run --locked --extra cpu --extra pipeline runtime-bench-fetch news
uv run --locked --extra cpu --extra pipeline runtime-bench-fetch minilm
uv run --locked --extra cpu --extra pipeline runtime-bench-fetch distilbert
# Larger encoder, for the next capability step
uv run --locked --extra cpu --extra pipeline runtime-bench-fetch bert
```

Use `--extra cuda` instead of `--extra cpu` on the CUDA machine for every `uv run`.
All three public model aliases point to committed immutable revisions in `model_specs.py`.
Files stay in the ignored `data/models/` cache. Runs use cached files only and never execute
remote model code. `--model /absolute/path/to/local/model` supports a saved Transformers
model/tokenizer directory; local assets are fingerprinted too.

The `news` fetch command downloads a [CSV mirror](https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv)
to `data/ag-news.csv` and records its URL and file hash; it never overwrites existing data.
Unlike model revisions, this mirror URL is not pinned: copy the exact acquired file
between computers and compare its SHA-256.

Alternatively, supply a local [AG News](https://huggingface.co/datasets/fancyzhx/ag_news)
CSV with `label,title,description`, no header, labels 1–4. Use the same file on every host.
`--limit` defaults to the first 2,000 rows; the harness uses a seeded 80/20 split. Use `--synthetic-data` if supplying toy data. These
small runs establish runtime behavior, not published NLP quality. Model/data licenses
and cards remain with their upstream sources.

## Climb the ladder

```sh
# 1. No download/input required. This CPU baseline is explicitly synthetic.
uv run --locked --extra cpu --extra pipeline runtime-bench classify --device cpu

# Or classify the coffee CSV with reviewed categorical target and numeric features
uv run --locked --extra cpu --extra pipeline runtime-bench classify --device cpu \
  --data data/coffee.csv --target YOUR_LABEL --features NUMERIC_A,NUMERIC_B

# 2. Embeddings, including CPU cleanup/tokenization on each measured batch
uv run --locked --extra cuda --extra pipeline runtime-bench embeddings \
  --device cuda --data data/ag-news.csv --batch 32 --length 128

# 3. Actual pretrained fill-mask inference, without a freshly initialized task head
uv run --locked --extra cuda --extra pipeline runtime-bench infer \
  --device cuda --model distilbert --data data/ag-news.csv --batch 32 --length 128

# 4. Full-model fine-tuning; labels initialize a fresh four-class head
uv run --locked --extra cuda --extra pipeline runtime-bench finetune \
  --device cuda --model distilbert --data data/ag-news.csv --batch 16 --steps 100
```

On Apple, use `--extra cpu` and `--device mps`; CPU uses `--device cpu`. These pretrained
steps use PyTorch, not MLX. Start with FP32; CUDA also permits FP16/BF16 as separate runs.
The ladder defaults to one trial and one warmup batch. `classify` performs a full fit
without a discarded warmup fit; `--steps` sets boosting iterations there. CPU threads
use the available physical core count unless you pass `--threads`. MLX's own threads
remain runtime managed.

Increase one parameter at a time: batch 16 → 32 → 64 → 128, length 128 → 256 → 512,
then model DistilBERT → BERT. Use `--limit` for more distinct text examples and `--steps`
for a longer job. Repeated batches satisfy a fixed step budget; the report records both
selected data rows and the number of executed samples. Coffee classification uses the
full supplied file; `--limit` controls synthetic CPU data and text selection only.

Do not reduce the workload automatically after failure. A caught OOM writes a failed
text/JSON report and exits nonzero. That configuration is part of the capability boundary.
A process killed by the OS cannot reliably write a report; note that outcome manually.

## Read hardware-test results

Each successful run writes a small text summary and JSON sidecar. Focus on:

- Total job wall time, including imports of optional pipeline libraries, loading,
  preprocessing, warmup, resets, execution and evaluation within the harness.
- Timed pipeline throughput and median/p95 batch latency. Embeddings/inference/training
  batch timing includes tokenization, array transfer, model execution and synchronization.
- Sampled peak process RSS and CPU use; NVIDIA device utilization and sampled device memory
  when NVML is available; CUDA allocator peak and Apple runtime allocation fields separately.
- Classification/fine-tuning evaluation accuracy and loss, plus the configuration and
  completion/failure outcome. Embedding norm is a sanity check, not embedding quality.

`job_samples_per_second` includes setup/evaluation overhead; trial `samples_per_second`
describes the measured work loop. Model acquisition, uv installation, top-level Python
startup/imports before the harness and report writing are excluded. Pretrained weights are
reloaded after training warmup to restore the starting model/head/optimizer/RNG without
keeping a second model on the GPU. Loading/reset costs remain part of total job wall time.

Sampling runs at 250 ms and can miss short memory spikes. Process CPU percentages use
100% per fully utilized core and can exceed 100%. NVIDIA counters are device-wide and
include other services; missing counters remain null with a reason. Apple GPU utilization
has no portable counter here. Apple allocations are not separate VRAM or total unified
memory. Host available RAM and swap changes are system-wide context, not proof of which
process paged. Resources cover setup through evaluation, not just the timed loop.

For identical work, compare reports directly:

```sh
uv run --locked --extra cpu runtime-bench-compare results/baseline.json results/candidate.json
```

Operational comparisons permit different CPU thread budgets: using each machine's cores
is part of the result. Dataset/model fingerprints, seed and work budgets must still match.
For increased workloads or caught failures, show outcomes without making a speedup claim:

```sh
uv run --locked --extra cpu runtime-bench-compare --capacity \
  results/baseline.json results/larger-or-failed.json
```

For an optional contention test, repeat a workload with your usual services running and
record what changed. Keep these runs separate from idle-system runs. Note responsiveness
and waiting in your own observations; the harness records measurements and completion
outcomes rather than assigning subjective hardware ratings.

Model references: [MiniLM embedding/pooling example](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
[DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased),
[BERT](https://huggingface.co/google-bert/bert-base-uncased),
[sklearn estimator](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).
