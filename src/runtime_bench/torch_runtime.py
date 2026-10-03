"""PyTorch adapter: the harness owns timing, the adapter owns execution."""

import copy

import torch
from torch import nn

from .hardware import identify, select_device, synchronize


class TorchRuntime:
    name = "torch"

    def __init__(self, args, model):
        if args.device == "gpu":
            raise ValueError("PyTorch requires --device cuda or mps; gpu is the MLX device name")
        self.args = args
        self.device = select_device(args.device)
        self.device_name = str(self.device)
        if args.precision != "fp32" and self.device.type != "cuda":
            raise ValueError(
                "Mixed precision currently requires CUDA; fp32 is the cross-backend baseline"
            )
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("Selected CUDA device does not support bf16")
        torch.set_float32_matmul_precision("highest")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.model = model.to(self.device)
        self.initial = copy.deepcopy(model.state_dict())
        self.criterion = nn.CrossEntropyLoss()
        self.dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
        self.parameters = sum(p.numel() for p in model.parameters())

    def reset(self):
        self.model.load_state_dict(self.initial)
        if self.args.task == "stateful":
            self.model.reset()
        self.model.train(self.args.mode == "train")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.args.precision == "fp16")

    def prepare(self, x, y):
        return x.to(self.device), y.to(self.device)

    def forward(self, x, y):
        with torch.autocast(
            device_type=self.device.type, dtype=self.dtype, enabled=self.args.precision != "fp32"
        ):
            logits = self.model(x)
            loss = self.criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        return logits, loss

    def step(self, x, y):
        with torch.set_grad_enabled(self.args.mode == "train"):
            _, loss = self.forward(x, y)
            if self.args.mode == "train":
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
        return loss.detach()

    def synchronize(self):
        synchronize(self.device)

    def losses(self, values):
        return torch.stack(values).cpu().tolist()

    def reset_peak_memory(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def memory(self):
        return {
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device)
            if self.device.type == "cuda"
            else None,
            "mps_current_allocated_bytes": torch.mps.current_allocated_memory()
            if self.device.type == "mps"
            else None,
        }

    def begin_evaluation(self):
        self.model.eval()
        if self.args.task == "stateful":
            self.model.reset()

    def score(self, x, y):
        with torch.inference_mode():
            logits, loss = self.forward(x, y)
            return (logits.argmax(-1) == y).sum().item(), y.numel(), loss.item()

    def hardware(self):
        return identify(self.device)
