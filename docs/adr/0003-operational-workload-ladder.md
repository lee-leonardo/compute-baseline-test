# ADR 0003: representative work and capability boundaries

Status: Accepted; supersedes ADR 0001's emphasis on fixed hardware-normalized experiments.

## Context

The useful question is how each computer changes practical local iteration, capacity and
resource-management friction. A synthetic model-throughput multiple alone cannot answer it.
We need mundane representative tasks and a clear way to retain capacity failures.

## Decision

Add an optional `pipeline` extra with sklearn and Hugging Face Transformers. The main
ladder is CPU classification, NLP preprocessing/embeddings, pretrained fill-mask inference,
and supervised full-model fine-tuning. Use public pinned MiniLM, DistilBERT and BERT models.
Fetch models explicitly before measured work, disable remote code and use safetensors.
Keep original learning workloads and MLX variants available.

Reuse compact text/JSON reports. Add a 250-ms psutil sampler and optional NVIDIA NVML
counters through the CUDA extra. Preserve distinct process RSS, allocator memory and
whole-device memory/utilization semantics; never substitute guesses for missing counters.
No telemetry service, container, distributed executor or profiling stack is introduced.

Operational timing includes real batch preprocessing and transfer. Whole-job timing
includes model loading/warmup/reset/evaluation; it excludes acquisition and startup outside
the harness. Reset pretrained models after warmup by reloading rather than retaining a
second GPU model. Use a seeded newly initialized classification head for fine-tuning.

Default to each machine's available physical CPU cores, one trial and one warmup batch.
Do not constrain RAM or alter WSL settings. Preserve the common seed, data/model provenance
and starting work budget, then let users increase workload on machines that have headroom.
Record caught failures, including OOM, with a nonzero exit code and configuration/resources.
Allow side-by-side capacity output without calculating a misleading speedup across unequal work.

## Consequences

Results describe the whole operational machine/runtime, including available CPU resources
and contention. Timed loops and whole-job waiting are separate. Sampling can miss brief
spikes; OS kills cannot always be recorded. Native CUDA/NVML needs independent host validation.
Measurements apply to the tested hardware/runtime/settings; subjective observations
remain separate from the report. See ADR 0004 for the current hardware-comparison framing.
Pretrained weights/model cache add a download step; optional dependencies remain optional.

## Checkpoint head validation

Fill-mask inference rejects the MiniLM embedding alias and local checkpoints with
missing model weights. Transformers can otherwise initialize absent task heads,
which would violate the pretrained-inference workload contract. Fine-tuning still
permits a fresh four-class classification head. Model selection and acquisition
remain independent of the execution runtime; see [model/runtime combinations](../models-and-runtimes.md).
