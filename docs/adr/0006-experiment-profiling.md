# ADR 0006: Experiment manifests and diagnostic profiling

## Decision

Use a small versioned TOML manifest to expand batch/length sweeps into sequential,
isolated CLI processes. Keep runtime execution in existing adapters and workload
semantics in workload modules. Retain legacy schema 1 fields and add report revision 2
metadata. Export original per-trial measurements without a composite hardware score.

Standard timing remains unchanged. Opt-in diagnostic NLP timing introduces explicit
boundaries around tokenization, transfer, and execution, including label transfer.
Comparison rejects different profiling modes, with absent mode interpreted as legacy
standard. Diagnostic instrumentation may affect measured performance.

## Consequences

Suites preserve case failures and logs and return a nonzero exit status. Missing hardware
never becomes an implicit CPU run. Run labels describe user-managed context; the runner
does not orchestrate background services. IDs group work independently of node location,
but do not replace provenance and quality checks. Full reports remain authoritative;
CSV is a visualization-ready interchange format. New learning objectives and checkpoint
lifecycles are deliberately separate workload tranches.
