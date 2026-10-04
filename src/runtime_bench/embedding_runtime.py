"""Device execution for normalized embeddings and symmetric in-batch contrastive loss."""

import torch

from .hf_runtime import HFRuntime


class EmbeddingRuntime(HFRuntime):
    """Reuse HF device/memory boundaries while keeping the embedding objective explicit."""

    temperature = 0.05

    def __init__(self, args, model):
        super().__init__(args, model)
        self.training = args.mode == "train"
        self.model.train(self.training)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5) if self.training else None

    def encode(self, tokens):
        """Masked mean pooling plus FP32 L2 normalization; preserves gradients."""
        with torch.autocast(
            device_type=self.device.type, dtype=self.dtype, enabled=self.args.precision != "fp32"
        ):
            hidden = self.model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (hidden.float() * mask).sum(1) / mask.sum(1).clamp_min(1)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)

    def step(self, left, right=None):
        """Train on paired positives or encode a batch with no weight updates."""
        with torch.set_grad_enabled(right is not None):
            if right is None:
                return self.encode(left).detach()
            a, b = self.encode(left), self.encode(right)
            logits = (a @ b.T) / self.temperature
            labels = torch.arange(len(a), device=self.device)
            loss = (
                torch.nn.functional.cross_entropy(logits, labels)
                + torch.nn.functional.cross_entropy(logits.T, labels)
            ) / 2
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        return loss.detach()
