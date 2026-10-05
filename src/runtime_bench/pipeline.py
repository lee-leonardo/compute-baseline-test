"""Bounded held-out text requests through tokenization, embeddings and classification."""

import copy
import hashlib
import math
import statistics
import time
import unicodedata

import numpy as np
import torch

from .hf_runtime import HFRuntime
from .model_specs import resolve, select_model
from .operational import fingerprint, read_texts, report
from .workloads import digest


def run(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    from transformers import AutoModel, AutoTokenizer
    import sklearn
    import transformers

    started = time.perf_counter()
    texts, labels, train, test = read_texts(args)
    if len(set(labels[train].tolist())) < 2 or not set(labels[test].tolist()) <= set(
        labels[train].tolist()
    ):
        raise ValueError(
            "Pipeline training split must contain at least two classes and all test classes"
        )
    path, provenance = resolve(select_model("embeddings", args.model), args.model_cache)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    assets = hashlib.sha256()
    for file in sorted(path.rglob("*")):
        if file.is_file() and file.suffix in {".json", ".txt", ".safetensors"}:
            assets.update(str(file.relative_to(path)).encode())
            with file.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    assets.update(chunk)
    metadata = {
        "synthetic": args.synthetic_data,
        "data_sha256": digest(args.data),
        "model_assets_sha256": assets.hexdigest(),
        "train_rows": len(train),
        "test_rows": len(test),
        "selected_rows": len(texts),
        "timed_split": "test",
        "preprocessing": "NFKC-strip; normalized-mean-embedding; train-only-StandardScaler-v1",
        "classifier": "LogisticRegression-C1-lbfgs-max_iter200",
        **provenance,
    }
    adapter_args = copy.copy(args)
    adapter_args.task = "embeddings"
    trials = []
    setup_seconds = 0.0
    with threadpool_limits(limits=args.threads):
        for repeat in range(args.repeats):
            setup_start = started if repeat == 0 else time.perf_counter()
            torch.manual_seed(args.seed)
            model = AutoModel.from_pretrained(
                path, local_files_only=True, trust_remote_code=False, use_safetensors=True
            )
            if args.length > model.config.max_position_embeddings:
                raise ValueError("--length exceeds this model's position capacity")
            metadata["initial_weights_sha256"] = fingerprint(model)
            metadata["attention_implementation"] = model.config._attn_implementation
            runtime = HFRuntime(adapter_args, model)
            parameters = sum(p.numel() for p in model.parameters())

            def embed(indices):
                start = time.perf_counter()
                clean = [unicodedata.normalize("NFKC", texts[int(i)]).strip() for i in indices]
                tokens = tokenizer(
                    clean,
                    padding="max_length",
                    truncation=True,
                    max_length=args.length,
                    return_tensors="pt",
                )
                tokenized = time.perf_counter()
                tokens = runtime.prepare(tokens)
                runtime.synchronize()
                transferred = time.perf_counter()
                vectors = runtime.step(tokens)
                runtime.synchronize()
                encoded = time.perf_counter()
                vectors = vectors.float().cpu().numpy()
                copied = time.perf_counter()
                if not np.isfinite(vectors).all():
                    raise ValueError("Non-finite pipeline embeddings")
                return vectors, {
                    "tokenize_seconds": tokenized - start,
                    "transfer_seconds": transferred - tokenized,
                    "embed_seconds": encoded - transferred,
                    "return_to_cpu_seconds": copied - encoded,
                }

            x_train = np.concatenate(
                [embed(train[i : i + args.batch])[0] for i in range(0, len(train), args.batch)]
            )
            classifier = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=200, random_state=args.seed)
            )
            classifier.fit(x_train, labels[train].numpy())
            metadata["classifier_training_rows"] = len(train)

            # Requests are stateless inference. Each trial reloads the encoder and refits the head.
            def request(indices):
                runtime.synchronize()
                start = time.perf_counter()
                vectors, phases = embed(indices)
                classification_start = time.perf_counter()
                probabilities = classifier.predict_proba(vectors)
                phases["classify_seconds"] = time.perf_counter() - classification_start
                if not np.isfinite(probabilities).all():
                    raise ValueError("Non-finite pipeline probabilities")
                return probabilities, phases, time.perf_counter() - start

            for i in range(args.warmup):
                request(test[(np.arange(args.batch) + i * args.batch) % len(test)])
            runtime.synchronize()
            runtime.reset_peak()
            setup_seconds += time.perf_counter() - setup_start
            durations, totals = [], {}
            start = time.perf_counter()
            for i in range(args.steps):
                _, phases, elapsed = request(
                    test[(np.arange(args.batch) + i * args.batch) % len(test)]
                )
                durations.append(elapsed)
                for key, value in phases.items():
                    totals[key] = totals.get(key, 0) + value
            elapsed = time.perf_counter() - start
            trials.append(
                {
                    "batch_median_ms": statistics.median(durations) * 1000,
                    "batch_p95_ms": sorted(durations)[math.ceil(len(durations) * 0.95) - 1] * 1000,
                    "samples_per_second": args.batch * args.steps / elapsed,
                    "wall_seconds": elapsed,
                    "phases": totals,
                    **runtime.memory(),
                }
            )
            predictions = np.concatenate(
                [request(test[i : i + args.batch])[0] for i in range(0, len(test), args.batch)]
            )
            classes = classifier.classes_
            quality = {
                "accuracy": accuracy_score(labels[test], classes[predictions.argmax(1)]),
                "cross_entropy": log_loss(labels[test], predictions, labels=classes),
            }
            hardware = runtime.hardware()
            hardware.update(
                selected_device=str(runtime.device),
                transformers=transformers.__version__,
                sklearn=sklearn.__version__,
            )
            # Drop each trial's accelerator state before constructing the next trial.
            runtime = model = classifier = x_train = None
    result = report(args, started, setup_seconds, trials, quality, hardware, metadata, parameters)
    result["protocol"] = "pipeline-v1"
    result["runtime"] = "torch+sklearn"
    result["timing_scope"] = (
        "held-out request: normalize/tokenize -> transfer -> embed -> CPU copy -> classify; phase-synchronized"
    )
    return result
