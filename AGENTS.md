# Agent contribution rules

Keep the harness → runtime adapter → workload → report boundary explicit.
Use uv and the committed Python/package pins. Do not commit credentials, raw datasets,
local env files, or generated results. Gitignore must not derive rules from environment values.
Update ADRs when changing runtime, timing semantics, or workload design.

Never fabricate hardware measurements or silently replace a requested GPU with CPU.
Keep synthetic results labeled. Preserve accelerator synchronization, seed handling,
train-only preprocessing, trial resets, and data fingerprints. Distinguish memory metrics.
Keep machine-specific private context out of repository documentation.

Run pytest, ruff check, and ruff format --check via the selected uv extra after changes.
Native GPU verification must be reported separately from CPU tests.
