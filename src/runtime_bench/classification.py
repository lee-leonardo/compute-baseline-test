"""Matched MLP training and held-out inference with portable checkpoint artifacts.

Checkpoints contain CPU state tensors and primitive metadata, never serialized
model objects or raw datasets. Inference reconstructs the workload and reuses the
stored preprocessing, split seed, architecture, and trained weights.
"""

import pickle
from datetime import datetime, timezone

import numpy as np
import torch

from .operational import fingerprint
from .workloads import coffee, digest, synthetic


def validate(args):
    """Reject incompatible backends and incomplete lifecycle requests before work."""
    if args.model is not None:
        raise ValueError("Classification uses its own MLP; use --checkpoint, not --model")
    if args.mode == "infer" and args.checkpoint is None:
        raise ValueError("classification infer requires --checkpoint from classification train")
    if args.mode == "train":
        if args.data is None and (args.target or args.features):
            raise ValueError("--target and --features require --data")
        if args.data is not None and (not args.target or not args.features):
            raise ValueError("CSV training requires --target and --features")
        if args.checkpoint is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            args.checkpoint = args.output / f"classification-{stamp}.pt"
        if args.checkpoint.exists():
            raise ValueError("Checkpoint already exists; choose a new --checkpoint output path")


def load_checkpoint(path):
    """Load only tensors/primitives on CPU and validate the artifact envelope."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format") != "classification-v1":
            raise ValueError("Unsupported classification checkpoint format")
        if not all(key in payload for key in ("config", "dataset", "state_dict", "weights_sha256")):
            raise ValueError("Incomplete classification checkpoint")
        config = payload["config"]
        required = {"width", "seed", "target", "features", "synthetic_data", "csv"}
        if not isinstance(config, dict) or not required <= config.keys():
            raise ValueError("Incomplete classification checkpoint configuration")
        if not isinstance(payload["dataset"], dict) or not isinstance(payload["state_dict"], dict):
            raise ValueError("Invalid classification checkpoint metadata or weights")
        return payload
    except (pickle.UnpicklingError, EOFError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid classification checkpoint: {exc}") from exc


def prepare(args):
    """Build training data or restore a trained model and its exact held-out split.

    CSV inference deliberately requires the same data fingerprint as training.
    This benchmarks a reproducible labeled evaluation workload, not arbitrary
    unlabeled prediction input. Synthetic inputs are regenerated from the seed.
    """
    payload = load_checkpoint(args.checkpoint) if args.mode == "infer" else None
    if payload:
        config = payload["config"]
        if config["csv"]:
            if args.data is None:
                raise ValueError("This checkpoint requires --data with the original CSV")
            if digest(args.data) != payload["dataset"]["data_sha256"]:
                raise ValueError("CSV fingerprint differs from the checkpoint training dataset")
        elif args.data is not None:
            raise ValueError("A synthetic checkpoint cannot be used with CSV input")
        for key in ("width", "seed", "target", "features", "synthetic_data"):
            setattr(args, key, config[key])
    torch.manual_seed(args.seed)
    if args.data is not None:
        preprocessing = payload["dataset"]["preprocessing"] if payload else None
        model, train, test, metadata = coffee(args, preprocessing)
        metadata["synthetic"] = args.synthetic_data
    else:
        model, train, test, metadata = synthetic(args)
        metadata["labels"] = ["0", "1"]
        metadata.update(train_rows=len(train[0]), test_rows=len(test[0]))
    classes = len(metadata["labels"])
    train_counts = np.bincount(train[1].numpy(), minlength=classes)
    test_counts = np.bincount(test[1].numpy(), minlength=classes)
    metadata["class_distribution"] = {
        "labels": metadata["labels"],
        "train": train_counts.tolist(),
        "test": test_counts.tolist(),
    }
    metadata["train_majority_class"] = int(train_counts.argmax())
    if payload:
        model.load_state_dict(payload["state_dict"], strict=True)
        if fingerprint(model) != payload["weights_sha256"]:
            raise ValueError("Checkpoint weight fingerprint mismatch")
        metadata["checkpoint_weights_sha256"] = payload["weights_sha256"]
        train = test  # Timed inference requests and quality scoring use held-out examples.
    metadata["initial_weights_sha256"] = fingerprint(model)
    return model, train, test, metadata


def save_checkpoint(args, model, metadata):
    """Persist the final reset trial after evaluation, outside measured job time.

    Exclusive creation prevents accidental overwrites. Each trial starts from
    the same weights; this saves the final trial, not a best-of-repeats selection.
    """
    weights = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    payload = {
        "format": "classification-v1",
        "config": {
            key: getattr(args, key)
            for key in ("width", "seed", "target", "features", "synthetic_data")
        },
        "dataset": metadata,
        "state_dict": weights,
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
