"""A small, readable MLX counterpart of the tabular PyTorch MLPs.

Imported only after the runtime checks the host and optional dependency.
Weights use the same (output, input) layout in both frameworks.
"""

import mlx.core as mx
import mlx.nn as nn
from torch import nn as torch_nn


class TabularMLP(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.layers = []
        for layer in reference:
            if isinstance(layer, torch_nn.Linear):
                linear = nn.Linear(layer.in_features, layer.out_features)
                linear.weight = mx.array(layer.weight.detach().cpu().numpy().copy())
                linear.bias = mx.array(layer.bias.detach().cpu().numpy().copy())
                self.layers.append(linear)
            elif isinstance(layer, torch_nn.GELU):
                self.layers.append(nn.GELU(approx="none"))
            else:
                raise ValueError(f"Unsupported MLX layer: {type(layer).__name__}")

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def loss_fn(model, x, y):
    return nn.losses.cross_entropy(model(x), y, reduction="mean")
