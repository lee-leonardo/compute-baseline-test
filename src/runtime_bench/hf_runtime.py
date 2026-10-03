"""Pretrained Transformers execution, kept separate from pipeline timing."""

import torch

from .hardware import identify, select_device, synchronize


class HFRuntime:
    def __init__(self, args, model):
        self.args = args
        if args.device == "gpu":
            raise ValueError("Use --device cuda, mps or cpu for pretrained PyTorch workloads")
        self.device = select_device(args.device)
        if args.precision != "fp32" and self.device.type != "cuda":
            raise ValueError("Mixed precision requires CUDA in this initial ladder")
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("Selected CUDA device does not support bf16")
        self.model = model.to(self.device)
        self.model.train(args.task == "finetune")
        self.dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
        torch.set_float32_matmul_precision("highest")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.optimizer = (
            torch.optim.AdamW(model.parameters(), lr=2e-5) if args.task == "finetune" else None
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")

    def prepare(self, tokens):
        return {k: v.to(self.device) for k, v in tokens.items()}

    def step(self, tokens, labels=None):
        training = labels is not None
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.dtype,
                enabled=self.args.precision != "fp32",
            ):
                if training:
                    output = self.model(**tokens, labels=labels.to(self.device))
                    result = output.loss
                elif self.args.task == "embeddings":
                    output = self.model(**tokens).last_hidden_state
                    mask = tokens["attention_mask"].unsqueeze(-1)
                    result = (output * mask).sum(1) / mask.sum(1).clamp_min(1)
                    result = torch.nn.functional.normalize(result.float(), p=2, dim=1)
                else:
                    result = self.model(**tokens).logits
            if training:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(result).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
        return result.detach()

    def synchronize(self):
        synchronize(self.device)

    def reset_peak(self):
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

    def hardware(self):
        return identify(self.device)
