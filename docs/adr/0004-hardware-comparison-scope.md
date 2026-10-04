# ADR 0004: frame results as hardware differences

Status: Accepted; clarifies the purpose of ADRs 0001–0003. No runtime behavior changes.

## Context

The repo is a lightweight aid for testing differences between Apple Silicon, NVIDIA RTX
and other configurations such as AMD. Representative workloads make runtime, memory and
capacity differences observable without assigning a universal hardware ranking.

## Decision

Use hardware-comparison language in the README, workload guide and CLI help. Describe
completion, elapsed time, throughput, resource use and workload limits as the outcomes.
Keep representative workloads, the shared harness/runtime/report boundaries and the
existing timing protocols.

State support separately from future extensions: Apple MPS/MLX and NVIDIA CUDA are
implemented paths; AMD/Intel CPUs use the CPU path. AMD GPU/ROCm and other accelerators
require explicit implementation and native verification. No AMD GPU support is implied
by generic PyTorch portability or by the CUDA device API.

Document hardware/runtime effects and distinct memory counters. Preserve missing data as
unknown and avoid hardware-performance claims beyond measured workloads/configurations.
Keep machine-specific context out of repository documentation.

## Consequences

Readers can identify what they can test today and what needs an adapter. The project
remains a small hardware-testing aid with representative tasks and compact reports.
Existing commands and report protocols remain unchanged; this decision adds no backend.
