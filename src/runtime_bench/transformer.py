"""From-scratch sequence classification with portable Transformer checkpoints.

The synthetic task compares first and last symbols; CSV mode classifies AG News.
Both use disjoint seeded row splits and a padding-aware, bidirectional encoder.
This is a learning benchmark, not autoregressive language-model serving.
"""

import csv
import hashlib
import pickle
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn

from .operational import fingerprint
from .workloads import digest, split_indices, tokenize


class SequenceTransformer(nn.Module):
    """Two encoder blocks, four attention heads, learned positions, masked mean pooling."""

    def __init__(self, width, length, vocabulary, classes):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, width, padding_idx=0)
        self.position = nn.Parameter(torch.randn(1, length, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, 4, width * 4, dropout=0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        # Initialize blocks independently rather than retaining cloned identical weights.
        for block in self.encoder.layers:
            for parameter in block.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.head = nn.Linear(width, classes)

    def forward(self, x):
        valid = x != 0
        hidden = self.embedding(x) + self.position[:, : x.shape[1]]
        hidden = self.encoder(hidden, src_key_padding_mask=~valid)
        mask = valid.unsqueeze(-1)
        return self.head((hidden * mask).sum(1) / mask.sum(1).clamp_min(1))


def validate(args):
    """Check lifecycle/backend constraints; restored architecture is checked in prepare."""
    if args.runtime != "torch":
        raise ValueError("Transformer lifecycle requires --runtime torch (CPU/CUDA/MPS)")
    if args.model is not None:
        raise ValueError("Transformer trains from scratch; use --checkpoint, not --model")
    if args.mode == "infer" and args.checkpoint is None:
        raise ValueError("transformer infer requires --checkpoint from transformer train")
    if args.mode == "train":
        if args.checkpoint is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            args.checkpoint = args.output / f"transformer-{stamp}.pt"
        if args.checkpoint.exists():
            raise ValueError("Transformer checkpoint already exists; choose a new output path")
        dimensions(args)


def dimensions(args):
    """Require valid attention width and enough examples/positions for this task."""
    if args.width < 4 or args.width % 4:
        raise ValueError("Transformer --width must be positive and divisible by four")
    if args.length < 4 or args.limit < 10:
        raise ValueError("Transformer requires --length >= 4 and --limit >= 10")


def synthetic_sequences(args):
    """Generate unique sequences with balanced, position-dependent binary labels."""
    if args.limit > 100000:
        raise ValueError("Synthetic sequence fixture supports at most 100000 rows")
    generator = torch.Generator().manual_seed(args.seed)
    rows, seen = [], set()
    for index in range(args.limit):
        while True:
            row = torch.randint(1, 33, (args.length,), generator=generator)
            if row[0] == row[-1]:
                continue
            low, high = sorted((int(row[0]), int(row[-1])))
            row[0], row[-1] = (high, low) if index % 2 else (low, high)
            key = tuple(row.tolist())
            if key not in seen:
                seen.add(key)
                rows.append(row)
                break
    x = torch.stack(rows)
    y = (x[:, 0] > x[:, -1]).long()
    h = hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest()
    return (
        x,
        y,
        {
            "synthetic": True,
            "generator": "endpoint-order-v1",
            "data_sha256": h,
            "labels": ["first_less_than_last", "first_greater_than_last"],
            "vocabulary": 33,
        },
    )


def news_sequences(args):
    """Tokenize a bounded labeled CSV; reject empty and repeated normalized input rows."""
    xs, ys, seen = [], [], set()
    with args.data.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if len(xs) == args.limit:
                break
            if len(row) != 3 or row[0] not in ("1", "2", "3", "4"):
                raise ValueError(
                    "Transformer CSV requires label,title,description without header; labels 1..4"
                )
            tokens = tokenize(" ".join(row[1:]), args.length)
            if not any(tokens):
                raise ValueError("Transformer input contains no word tokens")
            key = tuple(tokens)
            if key in seen:
                raise ValueError("Repeated tokenized text; deduplicate CSV before splitting")
            seen.add(key)
            xs.append(tokens)
            ys.append(int(row[0]) - 1)
    if len(xs) < 10:
        raise ValueError("At least 10 sequence rows required")
    return (
        torch.tensor(xs),
        torch.tensor(ys),
        {
            "synthetic": args.synthetic_data,
            "data_sha256": digest(args.data),
            "labels": ["1", "2", "3", "4"],
            "vocabulary": 8192,
            "tokenizer": "blake2b-word-v1-8192",
            "padding_id": 0,
        },
    )


def load_checkpoint(path):
    """Load CPU tensors and primitive metadata; missing artifacts never trigger training."""
    if not path.is_file():
        raise ValueError("Transformer checkpoint missing; run transformer train before inference")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format") != "transformer-v1":
            raise ValueError("Unsupported Transformer checkpoint")
        if not {"config", "dataset", "weights_sha256", "state_dict"} <= payload.keys():
            raise ValueError("Incomplete Transformer checkpoint")
        if (
            not isinstance(payload["config"], dict)
            or not {"width", "length", "limit", "seed", "synthetic_data", "csv"}
            <= payload["config"].keys()
        ):
            raise ValueError("Incomplete Transformer checkpoint configuration")
        if not isinstance(payload["dataset"], dict) or "data_sha256" not in payload["dataset"]:
            raise ValueError("Incomplete Transformer dataset metadata")
        if not isinstance(payload["state_dict"], dict):
            raise ValueError("Invalid Transformer weights")
        return payload
    except (pickle.UnpicklingError, EOFError, TypeError) as exc:
        raise ValueError(f"Invalid Transformer checkpoint: {exc}") from exc


def prepare(args):
    """Restore architecture, data and split before constructing the canonical CPU model."""
    saved = load_checkpoint(args.checkpoint) if args.mode == "infer" else None
    if saved:
        if saved["config"]["csv"]:
            if args.data is None or digest(args.data) != saved["dataset"]["data_sha256"]:
                raise ValueError("Transformer inference requires the original CSV fingerprint")
        elif args.data is not None:
            raise ValueError("Synthetic Transformer checkpoint cannot be used with CSV input")
        for key in ("width", "length", "limit", "seed", "synthetic_data"):
            setattr(args, key, saved["config"][key])
    dimensions(args)
    x, y, metadata = news_sequences(args) if args.data else synthetic_sequences(args)
    a, b = split_indices(len(x), args.seed)
    classes = len(metadata["labels"])
    train_counts = np.bincount(y[a].numpy(), minlength=classes)
    if not (train_counts > 0).all():
        raise ValueError("Training split misses a class; supply more data or another seed")
    metadata.update(
        train_rows=len(a),
        test_rows=len(b),
        selected_rows=len(x),
        class_distribution={
            "labels": metadata["labels"],
            "train": train_counts.tolist(),
            "test": np.bincount(y[b].numpy(), minlength=classes).tolist(),
        },
        train_majority_class=int(train_counts.argmax()),
        architecture="bidirectional-encoder-2layer-4head-maskedmean-v1",
        dropout=0.0,
        padding_mask=True,
    )
    torch.manual_seed(args.seed)
    model = SequenceTransformer(args.width, args.length, metadata["vocabulary"], classes)
    if saved:
        model.load_state_dict(saved["state_dict"], strict=True)
        if fingerprint(model) != saved["weights_sha256"]:
            raise ValueError("Transformer checkpoint weight fingerprint mismatch")
        if metadata["data_sha256"] != saved["dataset"]["data_sha256"]:
            raise ValueError("Transformer regenerated data fingerprint mismatch")
        metadata["checkpoint_weights_sha256"] = saved["weights_sha256"]
    metadata["initial_weights_sha256"] = fingerprint(model)
    train, test = (x[a], y[a]), (x[b], y[b])
    return model, test if saved else train, test, metadata


def save_checkpoint(args, model, metadata):
    """Save final-trial CPU weights after evaluation, without overwriting artifacts."""
    payload = {
        "format": "transformer-v1",
        "dataset": metadata,
        "config": {
            key: getattr(args, key)
            for key in ("width", "length", "limit", "seed", "synthetic_data")
        },
        "state_dict": model.state_dict(),
        "weights_sha256": fingerprint(model),
    }
    payload["config"]["csv"] = args.data is not None
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with args.checkpoint.open("xb") as handle:
        torch.save(payload, handle)
    return {
        "path": str(args.checkpoint),
        "weights_sha256": payload["weights_sha256"],
        "sha256": digest(args.checkpoint),
        "saved_trial": args.repeats,
    }
