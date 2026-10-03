"""Representative CPU → NLP → pretrained inference → fine-tuning workloads."""

import csv
import hashlib
import math
import statistics
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .hardware import command, identify
from .hf_runtime import HFRuntime
from .model_specs import resolve, select_model
from .workloads import coffee, digest, split_indices


TASKS = ("classify", "embeddings", "infer", "finetune")


def fingerprint(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(f"{name}:{tuple(value.shape)}:{value.dtype}".encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def report(args, started, setup, trials, quality, hardware, dataset, parameters=0):
    return {
        "schema_version": 1,
        "protocol": "operational-v1",
        "status": "completed",
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": command("git", "rev-parse", "HEAD"),
        "git_dirty": bool(command("git", "status", "--porcelain")),
        "runtime": "sklearn" if args.task == "classify" else "torch",
        "device": "cpu" if args.task == "classify" else str(hardware.pop("selected_device")),
        "hardware": hardware,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "dataset": dataset,
        "parameters": parameters,
        "setup_seconds": setup,
        "wall_seconds": time.perf_counter() - started,
        "quality": quality,
        "trials": trials,
    }


def classify(args):
    from sklearn.datasets import make_classification
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, log_loss
    import sklearn
    from threadpoolctl import threadpool_limits

    if args.device not in ("auto", "cpu") or args.precision != "fp32":
        raise ValueError("classify is a CPU sklearn task; use --device cpu --precision fp32")
    started = time.perf_counter()
    if args.data:
        _, train, test, metadata = coffee(args)
        train, test = tuple(t.numpy() for t in train), tuple(t.numpy() for t in test)
        metadata["synthetic"] = args.synthetic_data
    else:
        x, y = make_classification(
            n_samples=args.limit, n_features=32, n_informative=16, random_state=args.seed
        )
        a, b = split_indices(len(x), args.seed)
        train, test = (x[a], y[a]), (x[b], y[b])
        metadata = {
            "synthetic": True,
            "generator": "sklearn-classification-v1",
            "train_rows": len(a),
            "test_rows": len(b),
        }
    setup = time.perf_counter() - started
    trials = []
    with threadpool_limits(limits=args.threads):
        for _ in range(args.repeats):
            # A full fit is the representative work unit; no discarded warmup fit.
            model = HistGradientBoostingClassifier(
                max_iter=args.steps, random_state=args.seed, early_stopping=False
            )
            start = time.perf_counter()
            model.fit(*train)
            predicted = model.predict(test[0])
            probabilities = model.predict_proba(test[0])
            elapsed = time.perf_counter() - start
            trials.append(
                {
                    "batch_median_ms": elapsed * 1000,
                    "batch_p95_ms": elapsed * 1000,
                    "samples_per_second": len(train[0]) / elapsed,
                    "wall_seconds": elapsed,
                    "estimator_iterations": model.n_iter_,
                    "cuda_peak_allocated_bytes": None,
                    "mps_current_allocated_bytes": None,
                }
            )
    quality = {
        "accuracy": accuracy_score(test[1], predicted),
        "cross_entropy": log_loss(test[1], probabilities, labels=model.classes_),
    }
    hardware = identify(torch.device("cpu"))
    hardware["sklearn"] = sklearn.__version__
    return report(args, started, setup, trials, quality, hardware, metadata)


def read_texts(args):
    if not args.data:
        raise ValueError(
            "Text ladder tasks require --data AG News CSV (label,title,description; no header)"
        )
    texts, labels = [], []
    with args.data.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(texts) >= args.limit:
                break
            if len(row) < 3 or row[0] not in ("1", "2", "3", "4"):
                raise ValueError(
                    "AG News CSV requires label 1..4, title and description without a header"
                )
            texts.append(" ".join(row[1:]))
            labels.append(int(row[0]) - 1)
    train, test = split_indices(len(texts), args.seed)
    return texts, torch.tensor(labels), train, test


def pretrained(args):
    model_name = select_model(args.task, args.model)
    from transformers import (
        AutoModel,
        AutoModelForMaskedLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
    import transformers

    started = time.perf_counter()
    texts, labels, train, test = read_texts(args)
    path, provenance = resolve(model_name, args.model_cache)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model_class = {
        "embeddings": AutoModel,
        "infer": AutoModelForMaskedLM,
        "finetune": AutoModelForSequenceClassification,
    }[args.task]
    # Fingerprint local model assets as well as pinned public revisions.
    assets = hashlib.sha256()
    for file in sorted(path.rglob("*")):
        if file.is_file() and file.suffix in (".json", ".txt", ".safetensors"):
            assets.update(str(file.relative_to(path)).encode())
            with file.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    assets.update(chunk)
    metadata = {
        "synthetic": args.synthetic_data,
        "data_sha256": digest(args.data),
        "train_rows": len(train),
        "test_rows": len(test),
        "selected_rows": len(texts),
        "preprocessing": "unicode-NFKC-strip-v1",
        "model_assets_sha256": assets.hexdigest(),
        **provenance,
    }

    def load_runtime():
        torch.manual_seed(args.seed)
        kwargs = {"num_labels": 4} if args.task == "finetune" else {}
        model, loading = model_class.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            output_loading_info=True,
            **kwargs,
        )
        if args.task == "infer" and loading["missing_keys"]:
            raise ValueError(
                "infer requires a complete pretrained fill-mask checkpoint; "
                "loading would randomly initialize: " + ", ".join(loading["missing_keys"])
            )
        if args.length > model.config.max_position_embeddings:
            raise ValueError("--length exceeds this model's position capacity")
        # Canonical CPU fingerprint also captures the seeded fresh classification head.
        metadata["initial_weights_sha256"] = fingerprint(model)
        metadata["attention_implementation"] = model.config._attn_implementation
        return HFRuntime(args, model)

    def tokens_for(indices):
        clean = [unicodedata.normalize("NFKC", texts[int(i)]).strip() for i in indices]
        tokens = tokenizer(
            clean,
            padding="max_length",
            truncation=True,
            max_length=args.length,
            return_tensors="pt",
        )
        if args.task == "infer":
            # A realistic fill-mask inference request: mask the first ordinary text token.
            if tokenizer.mask_token_id is None:
                raise ValueError("The inference model requires a tokenizer with a mask token")
            for lane in range(len(indices)):
                special = tokenizer.get_special_tokens_mask(
                    tokens["input_ids"][lane].tolist(), already_has_special_tokens=True
                )
                positions = [i for i, flag in enumerate(special) if not flag]
                if positions:
                    tokens["input_ids"][lane, positions[0]] = tokenizer.mask_token_id
        return tokens

    runtime = load_runtime()
    setup = time.perf_counter() - started
    parameters = sum(p.numel() for p in runtime.model.parameters())
    trials = []
    for repeat in range(args.repeats):
        if repeat:
            runtime = None  # Release old GPU parameters/optimizer before reloading.
            runtime = load_runtime()
        for i in range(args.warmup):
            indices = train[(np.arange(args.batch) + i * args.batch) % len(train)]
            runtime.step(
                runtime.prepare(tokens_for(indices)),
                labels[indices] if args.task == "finetune" else None,
            )
        runtime.synchronize()
        runtime = None
        runtime = load_runtime()  # Reset pretrained weights, head, optimizer and dropout RNG.
        runtime.synchronize()
        runtime.reset_peak()
        durations, values = [], []
        phases = {"tokenize_seconds": 0.0, "transfer_seconds": 0.0, "execute_seconds": 0.0}
        start_loop = time.perf_counter()
        for i in range(args.steps):
            indices = train[(np.arange(args.batch) + i * args.batch) % len(train)]
            runtime.synchronize()
            value = None  # Do not retain a previous full-vocabulary output across requests.
            start = time.perf_counter()
            if args.profile == "diagnostic":
                tokens = tokens_for(indices)
                tokenized = time.perf_counter()
                tokens = runtime.prepare(tokens)
                batch_labels = (
                    labels[indices].to(runtime.device) if args.task == "finetune" else None
                )
                runtime.synchronize()
                transferred = time.perf_counter()
                value = runtime.step(tokens, batch_labels)
                runtime.synchronize()
                phases["tokenize_seconds"] += tokenized - start
                phases["transfer_seconds"] += transferred - tokenized
                phases["execute_seconds"] += time.perf_counter() - transferred
            else:
                tokens = runtime.prepare(tokens_for(indices))
                value = runtime.step(tokens, labels[indices] if args.task == "finetune" else None)
                runtime.synchronize()
            durations.append(time.perf_counter() - start)
            if args.task == "finetune":
                values.append(value.item())
        elapsed = time.perf_counter() - start_loop
        if not torch.isfinite(value).all().item() or not all(math.isfinite(v) for v in values):
            raise ValueError("Non-finite pipeline outputs; result is invalid")
        trials.append(
            {
                "batch_median_ms": statistics.median(durations) * 1000,
                "batch_p95_ms": sorted(durations)[math.ceil(len(durations) * 0.95) - 1] * 1000,
                "samples_per_second": args.batch * args.steps / elapsed,
                "wall_seconds": elapsed,
                "final_loss": values[-1] if values else None,
                "phases": phases if args.profile == "diagnostic" else None,
                **runtime.memory(),
            }
        )
    quality = {"accuracy": None, "cross_entropy": None}
    if args.task == "finetune":
        runtime.model.eval()
        correct, count, loss_sum = 0, 0, 0.0
        with torch.inference_mode():
            for offset in range(0, len(test), args.batch):
                indices = test[offset : offset + args.batch]
                tokens = runtime.prepare(tokens_for(indices))
                output = runtime.model(**tokens, labels=labels[indices].to(runtime.device))
                correct += (output.logits.argmax(-1).cpu() == labels[indices]).sum().item()
                count += len(indices)
                loss_sum += output.loss.item() * len(indices)
        if not math.isfinite(loss_sum):
            raise ValueError("Non-finite evaluation loss")
        quality = {"accuracy": correct / count, "cross_entropy": loss_sum / count}
    elif args.task == "embeddings":
        quality["mean_embedding_norm"] = value.float().norm(dim=-1).mean().item()
        quality["embedding_dimensions"] = value.shape[-1]
    else:
        quality["output_kind"] = "pretrained fill-mask logits; no classification quality score"
    hardware = runtime.hardware()
    hardware["transformers"] = transformers.__version__
    hardware["selected_device"] = str(runtime.device)
    return report(args, started, setup, trials, quality, hardware, metadata, parameters)


def run(args):
    if args.runtime != "torch":
        raise ValueError(
            "The workload ladder uses sklearn/PyTorch; MLX remains available for coffee/smoke"
        )
    if args.task == "classify":
        return classify(args)
    return pretrained(args)
