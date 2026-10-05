# ADR 0012: Bounded crossover, boundary refinement and text pipelines

## Decision

Focus the next tranche on short matched-work sweeps, one-axis policy-boundary refinement
and an actual tokenize/embed/classify dataflow. Remove sustained stress from the example
ladder. Keep runtime/device execution in existing adapters, workload composition in a
new pipeline module, and crossover/refinement orchestration in the harness.

Crossover analysis reuses report compatibility validation and requires at least three
trials plus configurable spread and benefit thresholds. Observed worst/best trial ranges
support conservative preferences; they are not confidence intervals. Only adjacent
opposite preferences bracket a transition. No extrapolated or unique crossover is claimed.

Boundary refinement requires source report evidence and rechecks the passing endpoint's
provenance and behavior. Midpoints use fresh-process confirmations and inherited guards,
with finite probe/time budgets. Unknown failures and contention guards do not define a
capacity boundary. The result is conditional on locally monotone scaling and unchanged
conditions. Preserve every child artifact and atomic progress index.

The new `pipeline-v1` workload uses a frozen pretrained encoder on the selected device,
then a CPU StandardScaler/logistic classifier fit only on training embeddings. Each
trial reloads the seeded model and refits the classifier. Timed held-out requests include
normalization/tokenization, transfers, embedding and classifier inference. Explicit phase
synchronization is part of this protocol. Preparation, warmup and full quality evaluation
remain outside request timing and inside job wall time. Seeds, data/model fingerprints
and separate memory metrics are retained. No implicit GPU fallback or online acquisition.

## Consequences

Component and composite observations now support practical placement decisions without
long-duration stress. Quality differences remain visible; faster does not imply equally
useful predictions. Sequential profile execution can be confounded by changing external
load. Instrumented phase timing is not an unsynchronized production latency claim.
Cross-framework results describe software plus hardware. Width/length capacity, more
pipeline families and complex model ladders can build on these observations later.
Native accelerator and public-model verification remain separate from offline CPU tests.
