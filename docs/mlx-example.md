# MLX example: the same MLP on Apple hardware

MLX provides NumPy-style arrays, automatic differentiation, and neural network layers.
This Apple runtime variant helps test hardware/runtime differences alongside NVIDIA RTX
CUDA and CPU runs. The example runs the same coffee or smoke classifier through the harness, using
Apple CPU or Metal GPU execution. It starts with the exact PyTorch weights and data,
rather than relying on two frameworks' random seeds producing the same numbers.

## Run it

Use native ARM Python on Apple Silicon macOS, with a release supported by the locked MLX
wheel. [MLX installation requirements](https://ml-explore.github.io/mlx/build/html/install.html)
list macOS 14 or newer; the selected wheel can impose a newer deployment target.
uv resolves the compatible binary and will fail on unsupported systems.

```sh
uv sync --locked --extra cpu --extra mlx
uv run --locked --extra cpu --extra mlx runtime-bench smoke --runtime mlx --device gpu

# Shared PyTorch baseline on the same Apple GPU
uv run --locked --extra cpu --extra mlx runtime-bench smoke --runtime torch --device mps

# Explicit MLX CPU example
uv run --locked --extra cpu --extra mlx runtime-bench smoke --runtime mlx --device cpu

# CUDA analog on a NVIDIA host, with matching task/configuration/revision
uv run --locked --extra cuda runtime-bench smoke --runtime torch --device cuda
```

The default is FP32 training. `--mode infer` measures execution of the randomly initialized
model, not inference from a trained checkpoint. MLX currently supports **coffee and smoke**;
news and stateful remain PyTorch workloads. Use the same target/features for coffee runs:

```sh
uv run --locked --extra cpu --extra mlx runtime-bench coffee --runtime mlx --device gpu \
  --data data/coffee.csv --target YOUR_LABEL --features NUMERIC_A,NUMERIC_B
```

For meaningful GPU scaling, sweep width and batch size in matching runs, for example
`--width 1024 --batch 512 --steps 100 --repeats 5`. A tiny smoke MLP is an installation
check and can primarily expose dispatch overhead. A wider smoke model remains a synthetic
classifier; it does not become real coffee data.

## Follow the code

`mlx_workloads.py` defines a short `TabularMLP` by copying the canonical Linear layers
and inserting exact GELU. `mlx_runtime.py` owns differentiation, optimizer state, and
evaluation. `cli.py` times both runtime adapters with the same loops.

The essential MLX training pattern is:

```python
loss_and_grad = nn.value_and_grad(model, loss_fn)
loss, grads = loss_and_grad(model, x, y)
optimizer.update(model, grads)
mx.eval(loss, model.parameters(), optimizer.state)
mx.synchronize()
```

MLX operations build lazy graphs. Evaluating the loss alone would not guarantee the
weight update finished. The harness materializes weights, optimizer state and the loss,
and synchronizes within compute timing. See [MLX optimizers](https://ml-explore.github.io/mlx/build/html/python/optimizers.html)
and [synchronization](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.synchronize.html).

MLX AdamW defaults to disabled bias correction; this example explicitly enables it to
match PyTorch, with learning rate 0.001, betas 0.9/0.999, epsilon 1e-8 and weight decay 0.01.
See [MLX AdamW](https://ml-explore.github.io/mlx/build/html/python/optimizers/_autosummary/mlx.optimizers.AdamW.html).
No `mx.compile` or `torch.compile` is enabled in this example.

## Read the results

The text summary names `runtime=mlx`, `device=mlx-gpu`, MLX's version, and unified memory.
JSON contains the same workload configuration, initial-weight fingerprint, data fingerprint,
quality and trials as the PyTorch result. Compare with:

```sh
uv run --locked --extra cpu --extra mlx runtime-bench-compare \
  results/baseline.json results/candidate.json
```

A runtime comparison on the same Apple GPU helps examine software differences. A CUDA
versus MLX comparison across hardware measures the complete hardware/runtime stack.
AMD CPUs can participate through the CPU path; AMD GPU execution is not configured here.
See [hardware coverage](hardware.md). Read median/p95 latency, loop throughput, and quality together. Do not attribute the
entire speed difference to matrix acceleration hardware.

`transfer_total_seconds` measures host batch → runtime-ready arrays in both adapters.
For MLX this includes NumPy conversion, int32 label conversion and array materialization
in shared memory, rather than a discrete-GPU transfer. MLX peak/active/cache allocator
bytes are separate metrics; do not add them to process RSS or equate them to CUDA VRAM.
`--threads` controls CPU preparation only for MLX; its execution threads are runtime managed.
PyTorch initialization/import costs remain part of this example's host process.

## Verify on Apple hardware

```sh
uv sync --locked --extra cpu --extra mlx --group dev
uv run --locked --extra cpu --extra mlx pytest --run-mlx
uv run --locked --extra cpu --extra mlx ruff check .
uv run --locked --extra cpu --extra mlx ruff format --check .
```

The native checks compare outputs and multiple AdamW updates for both tabular models,
then check training/inference trial resets and cross-runtime report comparison.
Without `--run-mlx`, CPU tests still run and native GPU tests are skipped deliberately.
