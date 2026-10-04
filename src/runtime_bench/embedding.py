"""Paired-text embedding lifecycle: data contract, resets, evaluation and artifacts.

Training uses unique paired positives and in-batch negatives. Inference times held-out
anchors; retrieval evaluation encodes both sides against the full held-out candidate set.
Acquisition and checkpoint writes remain outside measured execution.
"""

import csv
import hashlib
import json
import math
import statistics
import time
import unicodedata

import numpy as np
import torch

from .embedding_runtime import EmbeddingRuntime
from .model_specs import resolve
from .operational import fingerprint, report
from .workloads import digest, split_indices


def normalize(text):
    """Use the same text normalization in duplicate checks and tokenization."""
    return unicodedata.normalize("NFKC", text).strip()


def assets_hash(path):
    """Hash encoder/tokenizer assets independent of their directory location."""
    h = hashlib.sha256()
    for file in sorted(path.rglob("*")):
        if file.is_file() and file.name != "benchmark.json":
            h.update(str(file.relative_to(path)).encode())
            with file.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(chunk)
    return h.hexdigest()


def validate(args):
    """Reject ambiguous lifecycle requests before loading a model."""
    if args.runtime != "torch":
        raise ValueError("Embedding training/checkpoints require --runtime torch (CPU/CUDA/MPS)")
    if args.data is None or args.checkpoint is None:
        raise ValueError(
            "embedding train/infer requires --data paired.csv and --checkpoint DIRECTORY"
        )
    if args.mode == "train" and args.checkpoint.exists():
        raise ValueError("Embedding checkpoint already exists; choose a new output directory")
    if args.mode == "infer" and args.model is not None:
        raise ValueError("Embedding inference uses --checkpoint, not --model")


def read_pairs(args):
    """Load unique nonempty pairs; disallow shared normalized text across rows.

    This conservative policy prevents exact-text split leakage and duplicate negatives.
    Semantic near duplicates still require dataset review by the caller.
    """
    pairs, seen = [], set()
    with args.data.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["anchor", "positive"]:
            raise ValueError("Paired CSV requires exactly the header anchor,positive")
        for row in reader:
            if len(pairs) >= args.limit:
                break
            if None in row or any(v is None for v in row.values()):
                raise ValueError("Paired CSV rows require exactly two fields")
            pair = tuple(normalize(row[key]) for key in ("anchor", "positive"))
            keys = {text.casefold() for text in pair}
            if not all(pair) or len(keys) != 2 or seen.intersection(keys):
                raise ValueError(
                    "Pairs must be nonempty and unique across both columns after normalization"
                )
            seen.update(keys)
            pairs.append(pair)
    train, test = split_indices(len(pairs), args.seed)
    return pairs, train, test


def retrieval_quality(left, right):
    """Rank each aligned positive among all held-out candidates; ties rank pessimistically."""
    scores = left @ right.T
    correct = scores.diag()
    ranks = (scores >= correct[:, None]).sum(1).float()
    return {
        "retrieval_recall_at_1": (ranks == 1).float().mean().item(),
        "retrieval_mrr": (1 / ranks).mean().item(),
        "mean_positive_cosine": correct.mean().item(),
        "mean_embedding_norm": left.norm(dim=1).mean().item(),
        "embedding_dimensions": left.shape[1],
        "retrieval_candidates": len(right),
        "evaluation_pairs": len(left),
        "tie_policy": "pessimistic",
    }


def run(args):
    """Run independently reset encoder trials and save the final training trial."""
    from transformers import AutoModel, AutoTokenizer
    import transformers

    started = time.perf_counter()
    if args.mode == "infer":
        metadata_path = args.checkpoint / "benchmark.json"
        if not metadata_path.is_file():
            raise ValueError("Embedding checkpoint missing/incomplete; run embedding train first")
        saved = json.loads(metadata_path.read_text())
        if not isinstance(saved, dict) or saved.get("format") != "embedding-v1":
            raise ValueError("Unsupported embedding checkpoint")
        if not {"dataset", "config", "weights_sha256", "assets_sha256"} <= saved.keys():
            raise ValueError("Incomplete embedding checkpoint metadata")
        if (
            not isinstance(saved["dataset"], dict)
            or "data_sha256" not in saved["dataset"]
            or not isinstance(saved["config"], dict)
            or not {"seed", "limit", "synthetic_data"} <= saved["config"].keys()
        ):
            raise ValueError("Incomplete embedding checkpoint configuration")
        if digest(args.data) != saved["dataset"]["data_sha256"]:
            raise ValueError("Paired data fingerprint differs from the embedding checkpoint")
        if assets_hash(args.checkpoint) != saved["assets_sha256"]:
            raise ValueError("Embedding checkpoint assets fingerprint mismatch")
        for key in ("seed", "limit", "synthetic_data"):
            setattr(args, key, saved["config"][key])
        path = args.checkpoint
        provenance = {"model_id": "trained-embedding", "model_revision": saved["weights_sha256"]}
    else:
        path, provenance = resolve(args.model or "minilm", args.model_cache)
    pairs, train, test = read_pairs(args)
    if args.mode == "train" and not 2 <= args.batch <= len(train):
        raise ValueError(
            "Contrastive training requires 2 <= --batch <= training pairs (no repeated negatives)"
        )
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    metadata = {
        "synthetic": args.synthetic_data,
        "data_sha256": digest(args.data),
        "model_assets_sha256": assets_hash(path),
        "train_rows": len(train),
        "test_rows": len(test),
        "selected_rows": len(pairs),
        "timed_split": "train" if args.mode == "train" else "test",
        "sample_unit": "pair" if args.mode == "train" else "anchor text",
        "objective": "symmetric-in-batch-contrastive-v1",
        "temperature": 0.05,
        "optimizer": {"name": "AdamW", "learning_rate": 2e-5, "weight_decay": 0.01},
        "pooling": "masked-mean-l2-v1",
        "preprocessing": "NFKC-strip-v1",
        **provenance,
    }

    def load_runtime():
        torch.manual_seed(args.seed)
        model = AutoModel.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, use_safetensors=True
        )
        if args.length > model.config.max_position_embeddings:
            raise ValueError("--length exceeds this encoder's position capacity")
        metadata["initial_weights_sha256"] = fingerprint(model)
        if args.mode == "infer" and metadata["initial_weights_sha256"] != saved["weights_sha256"]:
            raise ValueError("Embedding checkpoint weight fingerprint mismatch")
        return EmbeddingRuntime(args, model)

    def tokens(indices, side):
        return tokenizer(
            [pairs[int(i)][side] for i in indices],
            padding="max_length",
            truncation=True,
            max_length=args.length,
            return_tensors="pt",
        )

    runtime = load_runtime()
    setup = time.perf_counter() - started
    parameters = sum(p.numel() for p in runtime.model.parameters())
    timed = train if args.mode == "train" else test
    trials = []
    for repeat in range(args.repeats):
        if repeat:
            runtime = None
            runtime = load_runtime()
        for step in range(args.warmup):
            indices = timed[(np.arange(args.batch) + step * args.batch) % len(timed)]
            runtime.step(
                runtime.prepare(tokens(indices, 0)),
                runtime.prepare(tokens(indices, 1)) if args.mode == "train" else None,
            )
        runtime.synchronize()
        runtime = None
        runtime = load_runtime()
        runtime.synchronize()
        runtime.reset_peak()
        durations, losses = [], []
        phases = {"tokenize_seconds": 0.0, "transfer_seconds": 0.0, "execute_seconds": 0.0}
        start_loop = time.perf_counter()
        for step in range(args.steps):
            indices = timed[(np.arange(args.batch) + step * args.batch) % len(timed)]
            runtime.synchronize()
            start = time.perf_counter()
            left = tokens(indices, 0)
            right = tokens(indices, 1) if args.mode == "train" else None
            tokenized = time.perf_counter()
            left = runtime.prepare(left)
            right = runtime.prepare(right) if right is not None else None
            if args.profile == "diagnostic":
                runtime.synchronize()
            transferred = time.perf_counter()
            value = runtime.step(left, right)
            runtime.synchronize()
            finished = time.perf_counter()
            durations.append(finished - start)
            phases["tokenize_seconds"] += tokenized - start
            phases["transfer_seconds"] += transferred - tokenized
            phases["execute_seconds"] += finished - transferred
            if args.mode == "train":
                losses.append(value.item())
        elapsed = time.perf_counter() - start_loop
        if not torch.isfinite(value).all().item() or not all(math.isfinite(v) for v in losses):
            raise ValueError("Non-finite embedding output or loss")
        trials.append(
            {
                "batch_median_ms": statistics.median(durations) * 1000,
                "batch_p95_ms": sorted(durations)[math.ceil(len(durations) * 0.95) - 1] * 1000,
                "samples_per_second": args.batch * args.steps / elapsed,
                "wall_seconds": elapsed,
                "final_loss": losses[-1] if losses else None,
                "phases": phases if args.profile == "diagnostic" else None,
                **runtime.memory(),
            }
        )
    runtime.model.eval()
    encoded = [[], []]
    with torch.inference_mode():
        for side in (0, 1):
            for offset in range(0, len(test), args.batch):
                result = runtime.step(
                    runtime.prepare(tokens(test[offset : offset + args.batch], side))
                )
                encoded[side].append(result.cpu())
    left, right = (torch.cat(values) for values in encoded)
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("Non-finite embedding evaluation")
    quality = retrieval_quality(left, right)
    hardware = runtime.hardware()
    hardware.update(selected_device=str(runtime.device), transformers=transformers.__version__)
    result = report(args, started, setup, trials, quality, hardware, metadata, parameters)
    result["protocol"] = "embedding-v1"
    result["job_samples_per_second"] = (
        args.batch * args.steps * args.repeats / result["wall_seconds"]
    )
    if args.mode == "train":
        args.checkpoint.mkdir(parents=True, exist_ok=False)
        runtime.model.save_pretrained(args.checkpoint, safe_serialization=True)
        tokenizer.save_pretrained(args.checkpoint)
        saved = {
            "format": "embedding-v1",
            "dataset": metadata,
            "config": {k: getattr(args, k) for k in ("seed", "limit", "synthetic_data")},
            "weights_sha256": fingerprint(runtime.model),
            "assets_sha256": assets_hash(args.checkpoint),
        }
        (args.checkpoint / "benchmark.json").write_text(json.dumps(saved, indent=2) + "\n")
        result["checkpoint"] = {
            "path": str(args.checkpoint),
            "weights_sha256": saved["weights_sha256"],
            "saved_trial": args.repeats,
        }
    return result


def create_news_pairs(source, output, limit):
    """Prepare weakly supervised title/description positives before measured runs.

    Empty, identical, or repeated normalized texts are skipped. Labels are not used.
    Never overwrite an existing output; preserve source and output fingerprints.
    """
    if limit < 10:
        raise ValueError("At least 10 pairs are required")
    if output.exists():
        raise ValueError("Pair output already exists; choose another path")
    pairs, seen = [], set()
    with source.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if len(row) != 3 or row[0] not in ("1", "2", "3", "4"):
                raise ValueError("Expected headerless AG News label,title,description rows")
            pair = tuple(normalize(text) for text in row[1:])
            keys = {text.casefold() for text in pair}
            if not all(pair) or len(keys) != 2 or seen.intersection(keys):
                continue
            pairs.append(pair)
            seen.update(keys)
            if len(pairs) == limit:
                break
    if len(pairs) < 10:
        raise ValueError("Fewer than 10 distinct usable title/description pairs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["anchor", "positive"])
        writer.writerows(pairs)
    output.with_suffix(".source.json").write_text(
        json.dumps(
            {
                "preparation": "ag-news-title-description-v1",
                "source_sha256": digest(source),
                "sha256": digest(output),
                "pairs": len(pairs),
                "supervision": "weak title/description pairing",
            },
            indent=2,
        )
        + "\n"
    )
    return output
