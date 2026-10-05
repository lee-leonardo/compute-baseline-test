# ADR 0011: Ordered saturation with per-profile checkpoint gates

## Decision

Add a separate `saturate plan/run/status` CLI and versioned TOML plan. Keep the existing
suite's independent-case semantics. Saturation profiles pin runtime/device/precision;
ordered stages express workload escalation using the existing CLI options. The parent
harness launches each stage in a fresh process and evaluates explicit report thresholds
before authorizing the next stage. It does not change adapters or workload timing.

Fail closed for missing required telemetry, invalid reports, nonzero exits and OOM.
Require a parent wall-time budget and minimum host-memory guard; allow a child RSS
budget. Sample guards every 100 ms, terminate and reap the child on breach/interruption.
Keep host memory, process RSS, device memory and allocator peaks distinct. Preserve
raw reports, full commands, gate evidence, last pass and first stop in an atomic index.

Explicit prior-index reuse carries only stops forward for an identical plan and
node/condition/working directory. An explicit standalone baseline import must match
the first stage's resolved configuration and node/condition, and can only stop work.
Successes are remeasured. No optimistic resume is provided: historical artifacts cannot
silently qualify new hardware, changed datasets or changed software for heavier work.

## Consequences

The result is a policy boundary or a censored/error observation, not an absolute
hardware maximum. An exhausted ladder has not found the limit. Missing telemetry
blocks escalation without implying saturation. Independent profiles can continue.
Synthetic workloads remain labeled in their existing reports. Timing synchronization,
seeding, trial resets, preprocessing and fingerprints stay owned by existing layers.

Live guards are best effort, sample only the direct child RSS and whole-host available
RAM, and cannot prevent instantaneous OOM. Device gates are post-stage only. External
load and supervision overhead affect measurements. Explicit node labels are not a
hardware identity guarantee. Concurrency, automatic boundary refinement, throughput
plateau inference and multi-stage pipeline composition are future workload designs.
