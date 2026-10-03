"""Data preprocessing stays on CPU and outside steady-state timing."""

import csv
import hashlib
import math
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_indices(n, seed):
    if n < 10:
        raise ValueError("At least 10 rows required")
    order = np.random.default_rng(seed).permutation(n)
    cut = int(n * 0.8)
    return order[:cut], order[cut:]


def coffee(args, preprocessing=None):
    """Build a tabular MLP, fitting preprocessing on train only unless supplied.

    Checkpoint inference supplies stored statistics to avoid refitting on input.
    The seeded split and categorical label ordering are shared across runtimes.
    """
    if not args.data or not args.target or not args.features:
        raise ValueError(
            "coffee requires --data, --target and --features (explicit, leakage-safe columns)"
        )
    with open(args.data, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    features = args.features.split(",")
    if args.target in features:
        raise ValueError("Target must not appear in features")
    if not rows or any(c not in rows[0] for c in [args.target, *features]):
        raise ValueError("CSV is empty or requested columns are missing")
    if any(not r[args.target].strip() for r in rows):
        raise ValueError("Missing target labels; clean the CSV before benchmarking")
    labels = sorted({r[args.target] for r in rows})
    if not 2 <= len(labels) <= min(100, len(rows) // 2):
        raise ValueError("Use a categorical classification target, not continuous price/score")
    train, test = split_indices(len(rows), args.seed)
    if set(rows[i][args.target] for i in train) != set(labels):
        raise ValueError("Training split misses a class; use more data or another seed")

    # Fit imputation and scaling only to the training split.
    def number(value):
        try:
            result = float(value)
            return result if math.isfinite(result) else np.nan
        except ValueError:
            if value.strip():
                raise ValueError(
                    "Coffee features must be numeric; encode categorical columns first"
                )
            return np.nan

    x = np.array([[number(r[c]) for c in features] for r in rows], dtype=np.float32)
    if np.isnan(x[train]).all(axis=0).any():
        raise ValueError("A feature has no finite training values")
    median = (
        np.nanmedian(x[train], axis=0)
        if preprocessing is None
        else np.asarray(preprocessing["median"], dtype=np.float32)
    )
    x = np.where(np.isnan(x), median, x)
    if preprocessing is None:
        mean, std = x[train].mean(0), x[train].std(0)
    else:
        mean = np.asarray(preprocessing["mean"], dtype=np.float32)
        std = np.asarray(preprocessing["std"], dtype=np.float32)
    x = (x - mean) / np.maximum(std, 1e-6)
    y = np.array([labels.index(r[args.target]) for r in rows], dtype=np.int64)
    model = nn.Sequential(
        nn.Linear(len(features), args.width),
        nn.GELU(),
        nn.Linear(args.width, args.width),
        nn.GELU(),
        nn.Linear(args.width, len(labels)),
    )
    return (
        model,
        (torch.from_numpy(x[train]), torch.from_numpy(y[train])),
        (torch.from_numpy(x[test]), torch.from_numpy(y[test])),
        {
            "data_sha256": digest(args.data),
            "features": features,
            "preprocessing": {
                "median": median.tolist(),
                "mean": mean.tolist(),
                "std": std.tolist(),
            },
            "labels": labels,
            "train_rows": len(train),
            "test_rows": len(test),
            "synthetic": False,
        },
    )


class TextTransformer(nn.Module):
    def __init__(self, width, length, classes=4):
        super().__init__()
        self.embedding = nn.Embedding(8192, width, padding_idx=0)
        self.position = nn.Parameter(torch.randn(1, length * 2, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, 4, width * 4, dropout=0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Linear(width, classes)

    def forward(self, x):
        h = self.encoder(self.embedding(x) + self.position[:, : x.shape[1]])
        mask = (x != 0).unsqueeze(-1)
        return self.head((h * mask).sum(1) / mask.sum(1).clamp_min(1))


class MemoryTransformer(TextTransformer):
    """Carry detached embedded input memory; causal attention over previous/current chunks."""

    def __init__(self, width, length):
        super().__init__(width, length, classes=32)
        self.memory_length = length
        self.memory = None

    def reset(self):
        self.memory = None

    def forward(self, x):
        current = self.embedding(x)
        h = current if self.memory is None else torch.cat((self.memory, current), dim=1)
        self.memory = h[:, -self.memory_length :].detach()
        length = h.shape[1]
        mask = torch.triu(torch.full((length, length), float("-inf"), device=x.device), 1)
        h = self.encoder(h + self.position[:, :length], mask=mask)
        return self.head(h[:, -x.shape[1] :])


def tokenize(text, length):
    tokens = re.findall(r"\w+", text.lower())[:length]
    ids = [
        1 + int.from_bytes(hashlib.blake2b(t.encode(), digest_size=4).digest(), "little") % 8191
        for t in tokens
    ]
    return ids + [0] * (length - len(ids))


def news(args):
    if not args.data:
        raise ValueError("news requires --data (AG News CSV: label,title,description; no header)")
    with open(args.data, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    x = torch.tensor([tokenize(" ".join(r[1:]), args.length) for r in rows])
    y = torch.tensor([int(r[0]) - 1 for r in rows])
    if not ((y >= 0) & (y < 4)).all():
        raise ValueError("AG News labels must be 1..4")
    train, test = split_indices(len(rows), args.seed)
    return (
        TextTransformer(args.width, args.length),
        (x[train], y[train]),
        (x[test], y[test]),
        {
            "data_sha256": digest(args.data),
            "synthetic": False,
            "train_rows": len(train),
            "test_rows": len(test),
            "tokenizer": "blake2b-word-v1-8192",
        },
    )


def synthetic(args):
    generator = torch.Generator().manual_seed(args.seed)
    if args.task == "stateful":
        # Each independent stream predicts the same position in the preceding chunk.
        x = torch.randint(
            1, 32, (args.steps + args.warmup + 2, args.batch, args.length), generator=generator
        )
        y = torch.roll(x, 1, dims=0)
        return (
            MemoryTransformer(args.width, args.length),
            (x, y),
            (x, y),
            {
                "synthetic": True,
                "generator": "delayed-copy-v1",
                "quality_split": "same-stream diagnostic",
            },
        )
    x = torch.randn(1024, 16, generator=generator)
    y = (x[:, 0] > 0).long()
    return (
        nn.Sequential(nn.Linear(16, args.width), nn.GELU(), nn.Linear(args.width, 2)),
        (x[:800], y[:800]),
        (x[800:], y[800:]),
        {"synthetic": True, "generator": "linear-class-v1"},
    )


def build(args):
    model, train, test, metadata = {
        "coffee": coffee,
        "news": news,
        "stateful": synthetic,
        "smoke": synthetic,
    }[args.task](args)
    # Exact shared CPU initialization, including shape/name, before device placement.
    fingerprint = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        fingerprint.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype}".encode())
        fingerprint.update(tensor.numpy().tobytes())
    metadata["initial_weights_sha256"] = fingerprint.hexdigest()
    return model, train, test, metadata
