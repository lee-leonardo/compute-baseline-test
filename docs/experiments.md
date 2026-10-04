# Reproducible experiments

Use a fixed work budget first: dataset, model, initialization, batch size, sequence
length, steps, precision, and seed. Repeat trials to expose variability. A shorter
runtime is meaningful only when the compared work and learning outcome remain comparable.

## Manifest workflow

`experiments/smoke.toml` runs offline. `experiments/nlp.toml` requires `pipeline` and
previously fetched news, MiniLM, and DistilBERT assets:

```sh
uv sync --locked --extra cpu --extra pipeline
uv run --locked --extra cpu --extra pipeline runtime-bench-fetch news
uv run --locked --extra cpu --extra pipeline runtime-bench-fetch minilm
uv run --locked --extra cpu --extra pipeline runtime-bench-fetch distilbert
uv run --locked --extra cpu --extra pipeline runtime-bench-suite experiments/nlp.toml \
  --node node-a --condition idle --dry-run
uv run --locked --extra cpu --extra pipeline runtime-bench-suite experiments/nlp.toml \
  --node node-a --condition idle
```

Run from the repository root. Input paths resolve relative to the working directory,
not the manifest. Keep selected uv extras on all invocations. Installation and acquisition
are outside measured execution. There is no Ollama/server dependency.

A manifest has `version = 1`, a `[defaults]` table and named `[[cases]]` tables.
Case options override defaults; option names use CLI spelling with underscores, e.g.
`model_cache`. `task` is required. `[cases.sweep]` supports positive integer `batch`
and `length` lists and expands their Cartesian product, up to 256 cases total.
Node, condition, and output belong to the suite command and cannot be set in a case.

The runner validates syntax and CLI argument types before execution. Hardware,
model compatibility, and runtime failures are recorded when the case executes.
Each case runs sequentially in a fresh subprocess. A failed case does not stop the
remaining cases; the suite exits nonzero if any fail. It does not retry or resize
failed jobs. Numeric case folders avoid using arbitrary labels as filesystem paths.
An incremental `index.json` records exit codes, arguments, and report paths, including
process failures that could not write a report. A partial index is not a completed suite.

## Standard versus diagnostic measurement

Standard mode retains the existing protocols (the new embedding lifecycle uses `embedding-v1`): synchronized batch pipeline timing for
pretrained NLP, and transfer/compute timing for micro workloads. Setup and full-job wall
time are retained separately. Micro and operational measurements are not interchangeable.

Set `profile = "diagnostic"` in an NLP case, or pass `--profile diagnostic` to a direct
`embeddings`, `infer`, or `finetune` command. Trial `phases` records total seconds for:

- CPU text cleanup, tokenization, and fill-mask preparation.
- Token and training-label device placement followed by synchronization.
- Forward execution and, when training, backward and optimizer work followed by synchronization.

These are wall-clock phase durations, not GPU kernel-only timings. Transfer on unified
memory is not a PCIe-bandwidth measurement. Extra synchronization and timer calls can
change behavior; compare diagnostic runs only with diagnostic runs. Other tasks reject
this mode. Acquisition, setup, warmup, reset, evaluation and telemetry overhead are not
attributed to these three timed-loop phases. Do not infer missing phase times by subtraction.

## Interpreting comparisons

Use explicit devices for cross-node experiments. Use the same source revision and
assets; the comparison rejects mismatched work/protocols and warns about dirty source.
CPU thread budgets may differ for operational tasks; this is included in the comparison.
Node labels and operating-condition labels do not define workload identity.

Repeat an identical manifest under a separately managed background workload and change
`--condition`. Compare the resulting reports to quantify benchmark slowdown. Keep a record
of the background workload configuration outside the public source. A label is not proof
that the background load was stable, and benchmark telemetry does not establish how
responsive the other services were.

For capacity sweeps, inspect each batch/length configuration independently. Failure is
an outcome; use `--capacity` instead of calculating speedup between unequal jobs.
More samples/sec does not prove faster convergence. Time-to-quality, service latency probes, and automatic job scheduling are not implemented.
Persisted checkpoints are available for classification MLPs and the paired-text embedding lifecycle.

## Export contract

Schema 1 JSON remains readable; new records add `report_revision = 2`, labels, and an
experiment ID. The ID hashes effective workload configuration, dataset/model provenance,
parameter count and protocol; execution location, thread count, labels, and file paths
are excluded. Actual dataset metadata may conservatively split groups when runtime/model
implementations differ. Failures have no verified experiment ID.

`runtime-bench-export` accepts report files, not suite indexes. The CSV preserves flattened
JSON fields using dotted column names, one row per trial, and one row for a failed report.
Lists remain JSON cells; unavailable values are blank, not zero. Run-level fields repeat
on each trial row: do not sum those fields across trials. Old and new reports can be
exported together. Preserve the original JSON for full fidelity and provenance.

Comparisons report median-based speedup and throughput gain, throughput min/max and
sample coefficient of variation when at least two trials exist. These describe trial
spread, not statistical confidence. No composite hardware score or theoretical TFLOPS
estimate is produced.
