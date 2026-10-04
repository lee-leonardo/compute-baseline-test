"""Resettable learning-workload trials, synchronized timing, and held-out scoring.

The CLI resolves configuration; this module owns the trial lifecycle. Runtime
adapters own device operations. Workloads construct data and canonical CPU weights.
"""

import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch

from .hardware import command
from .torch_runtime import TorchRuntime
from .workloads import build


def run_micro(args):
    """Run warmup/reset/trials, score held-out data, and optionally save trained weights."""
    started = time.perf_counter()
    if args.runtime == "mlx":
        from .mlx_runtime import MLXRuntime, validate

        validate(args)
        adapter = MLXRuntime
    else:
        adapter = TorchRuntime
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.task in ("classification", "transformer"):
        if args.task == "transformer":
            from .transformer import prepare
        else:
            from .classification import prepare

        model, train, test, metadata = prepare(args)
    else:
        model, train, test, metadata = build(args)
    runtime = adapter(args, model)
    stateful = args.task == "stateful"
    prediction_only = args.task in ("classification", "transformer") and args.mode == "infer"

    def batch(data, step):
        x, y = data
        if stateful:
            index = step % len(x)
            return x[index], y[index]
        ids = (torch.arange(args.batch) + step * args.batch) % len(x)
        return x[ids], y[ids]

    runtime.synchronize()
    setup_seconds = time.perf_counter() - started
    trials = []
    for _ in range(args.repeats):
        runtime.reset()
        for i in range(args.warmup):
            x, y = batch(train, i)
            x, y = runtime.prepare(x, y)
            if prediction_only:
                runtime.predict(x)
            else:
                runtime.step(x, y)
        runtime.synchronize()
        # Warmup compiles/initializes kernels, but must not change starting weights/state.
        runtime.reset()
        runtime.reset_peak_memory()
        latencies, transfers, losses, rss = [], [], [], []
        trial_start = time.perf_counter()
        for i in range(args.steps):
            x, y = batch(train, i)
            runtime.synchronize()
            start = time.perf_counter()
            x, y = runtime.prepare(x, y)
            runtime.synchronize()
            transfers.append(time.perf_counter() - start)
            start = time.perf_counter()
            if prediction_only:
                predictions = runtime.predict(x)
            else:
                losses.append(runtime.step(x, y))
            runtime.synchronize()
            latencies.append(time.perf_counter() - start)
            rss.append(psutil.Process().memory_info().rss)
        total = time.perf_counter() - trial_start
        if prediction_only:
            # Scoring is outside timed inference; the full held-out split is scored below.
            loss_values = [runtime.prediction_loss(predictions, y)]
        else:
            loss_values = runtime.losses(losses)
        final_loss = loss_values[-1]
        if not all(math.isfinite(loss) for loss in loss_values):
            raise ValueError("Non-finite loss; result is invalid")
        trials.append(
            {
                "compute_median_ms": statistics.median(latencies) * 1000,
                "compute_p95_ms": sorted(latencies)[max(0, int(len(latencies) * 0.95 + 0.999) - 1)]
                * 1000,
                "compute_samples_per_second": args.batch * args.steps / sum(latencies),
                "loop_samples_per_second": args.batch * args.steps / total,
                "transfer_total_seconds": sum(transfers),
                "loop_seconds": total,
                "final_loss": final_loss,
                "process_rss_sampled_peak_bytes": max(rss),
                **runtime.memory(),
            }
        )
    runtime.begin_evaluation()
    correct, count, loss_sum = 0, 0, 0.0
    confusion = (
        np.zeros((len(metadata["labels"]), len(metadata["labels"])), dtype=np.int64)
        if args.task in ("classification", "transformer")
        else None
    )
    iterations = len(test[0]) if stateful else (len(test[0]) + args.batch - 1) // args.batch
    for i in range(iterations):
        if stateful:
            x, y = batch(test, i)
        else:
            x, y = (t[i * args.batch : (i + 1) * args.batch] for t in test)
        if confusion is not None:
            prepared_x, prepared_y = runtime.prepare(x, y)
            logits = runtime.predict(prepared_x)
            predicted = runtime.predicted_labels(logits)
            batch_loss = runtime.prediction_loss(logits, prepared_y)
            actual = y.numpy()
            np.add.at(confusion, (actual, predicted), 1)
            batch_correct, batch_count = int((actual == predicted).sum()), len(actual)
        else:
            batch_correct, batch_count, batch_loss = runtime.score(*runtime.prepare(x, y))
        if stateful and i == 0:
            continue  # No preceding chunk exists at the start of a stream.
        correct += batch_correct
        count += batch_count
        loss_sum += batch_loss * batch_count
    if not count or not math.isfinite(loss_sum):
        raise ValueError("Invalid evaluation outputs; result is invalid")
    wall_seconds = time.perf_counter() - started
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    result = {
        "schema_version": 1,
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": command("git", "rev-parse", "HEAD"),
        "git_dirty": bool(command("git", "status", "--porcelain")),
        "runtime": runtime.name,
        "device": runtime.device_name,
        "hardware": runtime.hardware(),
        "config": config,
        "dataset": metadata,
        "parameters": runtime.parameters,
        "setup_seconds": setup_seconds,
        "wall_seconds": wall_seconds,
        "quality": {"accuracy": correct / count, "cross_entropy": loss_sum / count},
        "trials": trials,
    }

    if args.task == "transformer":
        for trial in trials:
            trial["padded_tokens_per_second"] = trial["loop_samples_per_second"] * args.length
        result["job_samples_per_second"] = args.batch * args.steps * args.repeats / wall_seconds
    if args.task in ("classification", "transformer"):
        result["protocol"] = f"{args.task}-v1"
        result["dataset"]["timed_split"] = "train" if args.mode == "train" else "test"
        from .classification_metrics import summarize

        result["quality"].update(summarize(confusion, metadata["labels"]))
        majority = metadata["train_majority_class"]
        result["quality"]["majority_baseline_accuracy"] = float(confusion.sum(1)[majority] / count)
        result["quality"]["accuracy_above_majority_baseline"] = (
            result["quality"]["accuracy"] - result["quality"]["majority_baseline_accuracy"]
        )
        if args.mode == "train":
            if args.task == "transformer":
                from .transformer import save_checkpoint
            else:
                from .classification import save_checkpoint

            result["checkpoint"] = save_checkpoint(args, runtime.export_model(), metadata)
    return result
