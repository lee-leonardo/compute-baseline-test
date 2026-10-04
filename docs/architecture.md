# Architecture

The purpose is empirical hardware comparison across Apple Silicon, NVIDIA RTX, and
CPU configurations such as AMD/Intel, with room for other verified adapters. See
[hardware coverage](hardware.md); AMD GPU/ROCm support has not been implemented.

`cli.py` owns command help, configuration validation, dispatch and report output.
`micro.py` owns repeated learning trials and evaluation; `classification.py` owns
the MLP checkpoint and preprocessing contract. See the [CLI guide](cli.md) and
[classification lifecycle](classification.md).
`torch_runtime.py` and `mlx_runtime.py` own device execution, optimization, synchronization,
and runtime memory metrics. `hardware.py` describes the relevant host/device hardware. `workloads.py` builds deterministic learning data and PyTorch modules.
`model_specs.select_model` chooses task defaults and rejects the embedding-only MiniLM
checkpoint for fill-mask inference. The pretrained loader also rejects missing inference
weights so local checkpoints cannot silently introduce random prediction heads.
See [model and runtime combinations](models-and-runtimes.md).

`operational.py` owns the representative pipeline work loop; `hf_runtime.py` executes
pretrained models, and `model_specs.py` handles separate acquisition of pinned model assets.
`profiling.Profiler` owns shared sampling and aggregation; vendor collectors implement
the telemetry protocol independently of workload execution. See [profiling](profiling.md).
Workloads return a model, training tensors, evaluation tensors, and metadata.

In the micro workloads, preprocessing happens on CPU before timing. The representative
pretrained pipelines include CPU cleanup/tokenization in batch timing. Coffee fits imputation and scaling only
on the training split. Coffee and news use a seeded 80/20 row split of the supplied file;
news does not consume a separate official test file. Models initialize identically per seed;
each trial resets weights, optimizer, and streaming state after warmup. Dropout is disabled
in the small learning models; pretrained fine-tuning retains the model's training dropout.
State carries across timed chunks and is cleared at trial and evaluation boundaries.

Runtime adapters support PyTorch CPU/CUDA/MPS and optional Apple MLX CPU/Metal GPU. ARM names the CPU architecture;
it does not identify a GPU programming interface. Apple GPU work uses MPS or the explicitly selected MLX runtime.
Other ARM GPU systems require a separately supported backend. NVIDIA Tensor Cores and
Apple GPU matrix hardware are not treated as equivalent or inferred from processor architecture.

FP32 with CUDA TF32 disabled is the common baseline. CUDA autocast FP16 uses loss scaling;
BF16 checks device support. NVIDIA utilization is sampled when available; unsupported utilization counters remain
unknown. Matrix-unit utilization and energy consumption are not inferred.
Compilation and distributed execution are outside the initial scope.

The harness is a synchronous CLI with nonzero failure exit codes suitable for another
system to invoke after clone and `uv sync --locked`. It writes successful results and caught failure outcomes, retaining configuration and
resource context for capability boundaries.
Synthetic workloads make offline smoke checks possible; real datasets are explicit local inputs.

MLX supports classification/coffee/smoke MLPs in FP32. Classification adapters expose
prediction, scoring, and canonical CPU model export so the harness never reads MLX
weights directly. `classification_metrics.py` summarizes a shared confusion matrix. Canonical CPU weights are copied into MLX and
fingerprinted before placement in either runtime. Shared timing loops force MLX lazy
evaluation of loss, weights and optimizer state. AdamW settings, including bias correction,
match PyTorch. Native parity tests are opt-in. See [ADR 0002](adr/0002-mlx-runtime.md)
and the [MLX walkthrough](mlx-example.md).

The primary workflow is the [hardware workload ladder](ladder.md). Reports keep micro and
operational timing protocols separate. Actual host resources are retained rather than
normalized; compare equal-work iteration time first, then increase work to find limits.
See [ADR 0003](adr/0003-operational-workload-ladder.md).

`experiments.py` expands versioned manifests and supervises isolated CLI processes;
it never executes model operations itself. `reporting.py` annotates completed/failed
reports and exports per-trial tables. Node and condition labels describe execution
context independently of workload identity. See [experiment workflow](experiments.md)
and [ADR 0006](adr/0006-experiment-profiling.md). New objectives and checkpoint support
are tracked in the [workload roadmap](roadmap.md).

`embedding.py` owns paired-text data validation, encoder trial resets, retrieval evaluation,
and checkpoint directories; `embedding_runtime.py` owns normalized pooling and contrastive
updates on the selected device. This uses the separate `embedding-v1` protocol. The
legacy plural `embeddings` workload remains unchanged. See [ADR 0009](adr/0009-embedding-lifecycle.md).

`transformer.py` owns the from-scratch sequence model, data contract and checkpoint format.
It reuses `micro.py` trials, `TorchRuntime`, and classification quality aggregation, with
`transformer-v1` report semantics. The legacy news/stateful models remain unchanged.
See [ADR 0010](adr/0010-transformer-lifecycle.md).
