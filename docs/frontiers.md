# Crossover, boundary refinement and representative pipelines

These workflows use short, bounded requests. Sustained stress, automatic complexity
ladders and concurrent services are outside this tranche. Configure resource budgets
for the node; example thresholds are policy choices, not measured hardware limits.

## Matched crossover observations

Start with `experiments/crossover.toml`, a batch-only synthetic MLP sweep. Set the
candidate device explicitly to one available on the node (`cuda`, `mps`, or an MLX
profile with runtime `mlx` and device `gpu`). Use the same precision, CPU thread
budget, seed, repeats, warmup and workload settings on both profiles. Select uv extras
for the actual runtime; the example below uses CUDA. CPU/MPS use `--extra cpu`, with
`--extra mlx` added for MLX.

```sh
uv run --locked --extra cuda runtime-bench saturate plan experiments/crossover.toml
uv run --locked --extra cuda runtime-bench saturate run experiments/crossover.toml \
  --node node-a --condition idle
uv run --locked --extra cuda runtime-bench crossover results/saturation-<id>/index.json \
  --baseline cpu --candidate accelerator --axis batch --min-speedup 1.10 --max-cv 0.10 \
  --output results/crossover.json
```

`crossover` reads reports without new compute. Each profile's ladder must increase
only the selected axis (`batch`, `length` or `width`). It uses the existing report
compatibility checks for measurement protocol, data/preprocessing/weights, parameter
count and source revision, and requires matching planned work and node conditions.
Rows flag dirty source checkouts, which require manual verification of identical code.
Cross-runtime ratios describe the complete execution configuration, including kernel
and framework differences; they do not isolate hardware alone.

The primary decision metric is measured loop/request throughput. Job wall time,
latency ratios, trial spread and quality changes are also retained. At least three
trials per report and an acceptable coefficient of variation are required. A candidate
is beneficial only if its **slowest trial** exceeds the baseline's **fastest trial**
by the requested margin. The inverse identifies a baseline preference. Otherwise the
result is ambiguous. These observed ranges are not statistical confidence intervals.
Gated and untested stages do not become inferred speedups.

Only adjacent baseline-preferred → candidate-beneficial observations produce a
transition interval. Nonmonotone observations remain visible; the tool does not invent
a unique global crossover. For ambiguity, collect a new bounded campaign with more
trials or closer stage values. Profiles execute sequentially, so control external load
and consider a repeated campaign in reversed profile order before strong conclusions.

## Refining a policy boundary

```sh
uv run --locked --extra cpu runtime-bench boundary results/saturation-<id>/index.json \
  --profile cpu --axis batch --resolution 4 --max-probes 6 --confirmations 2
```

Use an increasing single-axis ladder with a measured pass followed by a threshold
stop. A mixed batch/width ladder such as the generic saturation example must first be
split into separate sweeps. The command verifies endpoint report evidence, repeats the
passing endpoint, then bisects the interval. Every observation requires at least two
fresh-process confirmations. The old heavier stopped endpoint is not rerun.

The passing endpoint must still match its original report's measurement identity,
source revision and reported hardware/environment. Keep assets and conditions unchanged;
like ordinary comparisons, a dirty checkout requires manual source verification.
A mismatch blocks refinement. Probes inherit all profile gates and live limits. Errors,
OOM, live guards and whole-host/device contention gates block refinement rather than
becoming a numerical upper bound. Contradictory confirmation outcomes stop as ambiguous.
Only wall time, compute/request p95, process RSS and CUDA allocator threshold stops
support refinement. Transformer/news/stateful width endpoints and resolution must be
multiples of four.

The probe budget counts distinct midpoint values; endpoint confirmations are additional.
At most `(max_probes + 1) * confirmations` child runs execute, each with the original
timeout. Results retain the initial interval, all probe indexes and the final interval.
This is a **conditional bracket assuming locally monotone behavior**, not proof of an
absolute maximum. Exhausting the probe budget leaves an explicitly unresolved interval.
Root `index.json` is written atomically and preserves progress on interruption.

## Representative text pipeline

```sh
uv run --locked --extra cpu --extra pipeline runtime-bench fetch news
uv run --locked --extra cpu --extra pipeline runtime-bench fetch minilm
uv run --locked --extra cpu --extra pipeline runtime-bench pipeline \
  --device cpu --model minilm --data data/ag-news.csv --mode infer \
  --limit 500 --batch 16 --length 64 --steps 5 --warmup 1 --repeats 3
uv run --locked --extra cpu --extra pipeline runtime-bench saturate run experiments/pipeline.toml \
  --node node-a --condition idle
```

The flow is NFKC normalization/tokenization → selected-device encoder → normalized
mean embedding → CPU copy → standardized logistic classifier. Outputs actually feed
subsequent stages. The classifier and scaler fit only on training embeddings; measured
requests come from the held-out split. Full held-out classification quality is evaluated
outside the request timing. Each trial reloads seeded encoder weights and refits the
classifier. Warmup consists of stateless inference requests. No downloads occur inside
measurement. Synthetic/local toy fixtures must use `--synthetic-data`.

`pipeline-v1` times phase-synchronized requests and reports tokenization, transfer,
embedding, CPU return and classification time separately, plus total request latency
and throughput. These phase measurements include instrumentation and are not equivalent
to an unsynchronized production pipeline. `wall_seconds` additionally includes loading,
training the classifier, warmup and full evaluation; `setup_seconds` accumulates trial
preparation including warmup. Do not sum peaks from different memory domains.

The encoder uses PyTorch CPU/CUDA/MPS; the classifier stays on CPU, explicitly labeled
`torch+sklearn`. FP32 is supported throughout; existing adapter rules govern CUDA mixed
precision. MLX is not supported by this pipeline. Existing `embeddings --profile
diagnostic` runs provide isolated encoder context, but their timing/work scope differs:
compare them as component diagnostics, not as equal-work pipeline speedups.

`steps` bounds the number of measured requests; each contains `batch` held-out examples,
wrapping deterministically if needed. A batch sweep therefore changes the work per run;
only equal batch/step points across profiles are compared. Use `batch_p95_ms` for pipeline
gates. Add an explicit accelerator profile to the pipeline manifest to collect matched
composite crossover observations using the same analysis command.

Crossover exits 0 for a valid analysis (including ambiguous/untested outcomes), 2 for
invalid input. Boundary exits 0 for a resolved or budget-limited bracket, 1 when blocked
or ambiguous, 2 for invalid input, and 130 on Ctrl-C. Read the recorded status rather
than treating exit 0 as proof that a crossover or exact limit was found.
