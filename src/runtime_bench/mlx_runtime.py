"""Optional Apple MLX runtime with explicit lazy evaluation boundaries."""

import importlib.metadata
import platform

import numpy as np
import torch

from .hardware import identify


def validate(args):
    if args.task not in ("coffee", "smoke"):
        raise ValueError(
            "MLX example supports coffee and smoke; use --runtime torch for news/stateful"
        )
    if args.precision != "fp32":
        raise ValueError("The MLX example supports fp32 only for matched PyTorch comparisons")
    if args.device not in ("auto", "gpu", "cpu"):
        raise ValueError("MLX requires --device gpu, cpu or auto; CUDA/MPS are PyTorch backends")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("This MLX variant requires native Apple Silicon macOS")


class MLXRuntime:
    name = "mlx"

    def __init__(self, args, reference):
        validate(args)
        try:
            import mlx.core as mx
            import mlx.nn as nn
            import mlx.optimizers as optim
            from .mlx_workloads import TabularMLP, loss_fn
        except ImportError as exc:
            raise ValueError(
                "Install the optional MLX runtime with uv sync --extra cpu --extra mlx"
            ) from exc
        self.mx, self.nn, self.optim = mx, nn, optim
        self.model_class, self.loss_fn = TabularMLP, loss_fn
        self.args, self.reference = args, reference
        # Auto means Apple GPU; it never falls back to CPU for MLX.
        self.device_name = "mlx-cpu" if args.device == "cpu" else "mlx-gpu"
        if self.device_name == "mlx-gpu" and not mx.metal.is_available():
            raise ValueError("MLX Metal GPU requested but unavailable; no CPU fallback")
        mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
        mx.random.seed(args.seed)
        self.parameters = sum(p.numel() for p in reference.parameters())
        self.reset()

    def reset(self):
        # Rebuild from the unchanged CPU reference, not from previously trained weights.
        self.model = self.model_class(self.reference)
        self.model.train(self.args.mode == "train")
        self.optimizer = self.optim.AdamW(
            learning_rate=0.001,
            betas=[0.9, 0.999],
            eps=1e-8,
            weight_decay=0.01,
            bias_correction=True,
        )
        self.loss_and_grad = self.nn.value_and_grad(self.model, self.loss_fn)
        self.mx.eval(self.model.parameters())
        self.synchronize()

    def prepare(self, x, y):
        # Array materialization on unified memory is not a discrete-GPU PCIe copy.
        x = self.mx.array(x.numpy())
        y = self.mx.array(y.numpy().astype(np.int32))
        self.mx.eval(x, y)
        return x, y

    def step(self, x, y):
        if self.args.mode == "train":
            loss, grads = self.loss_and_grad(self.model, x, y)
            self.optimizer.update(self.model, grads)
            # Evaluating only the scalar loss would omit optimizer update execution.
            self.mx.eval(loss, self.model.parameters(), self.optimizer.state)
        else:
            loss = self.loss_fn(self.model, x, y)
            self.mx.eval(loss)
        return self.mx.stop_gradient(loss)

    def synchronize(self):
        self.mx.synchronize()

    def losses(self, values):
        losses = self.mx.stack(values)
        self.mx.eval(losses)
        return losses.tolist()

    def reset_peak_memory(self):
        self.mx.reset_peak_memory()

    def memory(self):
        return {
            "cuda_peak_allocated_bytes": None,
            "mps_current_allocated_bytes": None,
            "mlx_peak_allocated_bytes": self.mx.get_peak_memory(),
            "mlx_active_allocated_bytes": self.mx.get_active_memory(),
            "mlx_cache_bytes": self.mx.get_cache_memory(),
        }

    def begin_evaluation(self):
        self.model.eval()

    def score(self, x, y):
        logits = self.model(x)
        loss = self.nn.losses.cross_entropy(logits, y, reduction="mean")
        correct = self.mx.sum(self.mx.argmax(logits, axis=-1) == y)
        self.mx.eval(loss, correct)
        return correct.item(), y.size, loss.item()

    def hardware(self):
        # Reuse host discovery; MPS need not be available for MLX GPU execution.
        hardware = identify(torch.device("cpu"))
        if self.device_name == "mlx-gpu":
            info = self.mx.device_info(self.mx.gpu)
            hardware.update(
                gpu=info.get("device_name", "Apple GPU"), gpu_vendor="Apple", unified_memory=True
            )
        hardware["mlx"] = importlib.metadata.version("mlx")
        hardware["mlx_threads"] = "runtime managed; --threads controls CPU preparation only"
        return hardware
