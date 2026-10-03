# Hardware coverage and comparison

This repo helps test how hardware configurations differ when running the same ML work.
The useful outputs are elapsed time, throughput, memory use, utilization where available,
and completion or capacity failure. Workloads span CPU classification, an NLP pipeline,
pretrained inference and fine-tuning. Small model/MLX examples provide additional variants.

## Current coverage

| Hardware | Runtime in this repo | Coverage |
| --- | --- | --- |
| Apple Silicon CPU | CPU | sklearn and PyTorch workloads; optional MLX tabular examples |
| Apple Silicon GPU | PyTorch MPS | Representative pretrained pipelines and learning workloads |
| Apple Silicon GPU | MLX / Metal | Coffee/smoke MLPs in FP32; not the pretrained workload ladder |
| NVIDIA RTX GPU | PyTorch CUDA | GPU workloads with the CUDA extra and a compatible host driver |
| AMD or Intel CPU | CPU | CPU workloads, subject to package/platform availability |
| AMD GPU | Future ROCm adapter | Not configured, implemented or validated |
| Other GPU/accelerator | Future adapter | Not implemented or validated |

“Covered” describes an implemented execution path, not verification of every chip,
operating system or driver. Native MPS/MLX checks have run on an Apple host; native
CUDA/NVIDIA telemetry still requires verification on a CUDA host. CPU tests do not
substitute for native verification on other configurations.

Use `--device cpu` for an AMD CPU. It does not exercise an AMD GPU. The committed CUDA
wheels and NVIDIA telemetry are NVIDIA-specific. Installing a ROCm wheel by hand does
not make this repo an AMD GPU test: device identification and memory reporting currently
assume NVIDIA for the CUDA path. An AMD GPU extension needs explicit packaging, device
identification, synchronization, allocation reporting and native workload verification.

## Interpret differences

Start with the same code, dataset, model revision, seed, batch, sequence length, precision
and work budget. Record the CPU thread budget and available host resources too. The
representative pipeline defaults use each machine's physical cores; differences in those
resources are part of the whole-system result. The optional micro protocol can hold the
thread budget fixed for a narrower comparison.

Compare Apple MPS and NVIDIA CUDA to observe differences in the hardware/runtime stacks.
Compare MPS and MLX on the same Apple GPU to examine runtime effects. Neither comparison
isolates hardware alone: framework kernels, drivers, CPU preprocessing, memory layout and
precision contribute. A manufacturer, ARM architecture, or “Tensor Core” label does not
predict performance for every workload.

After equal-work runs, increase batch size, sequence length, model size or duration on
each configuration. Retain failed/OOM reports as observations about workload limits. Use
`runtime-bench-compare --capacity` for unequal workloads or failures; those outputs do not
claim a speedup for doing different amounts of work.

Process RSS, CUDA allocator memory, Apple unified-memory allocations and whole-device
memory are distinct counters. Missing utilization data stays unknown. Keep idle and
contended runs separate, and record any manual responsiveness observations alongside the
reports. Conclusions should name the workload and tested settings instead of assigning a
universal hardware ranking.

Follow the [workload ladder](ladder.md) for commands and [measurement details](benchmarking.md)
for timing and memory definitions.
