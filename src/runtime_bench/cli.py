import argparse
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch

from .hardware import command, identify, select_device
from .torch_runtime import TorchRuntime
from .workloads import build
from .profiling import Profiler
from .operational import TASKS


def parser():
    p = argparse.ArgumentParser(
        description="Test hardware differences across Apple, NVIDIA RTX and CPU configurations using ML workloads"
    )
    p.add_argument("task", choices=["coffee", "news", "stateful", "smoke", *TASKS])
    p.add_argument(
        "--runtime",
        choices=["torch", "mlx"],
        default="torch",
        help="Execution adapter (default: torch); mlx supports coffee/smoke only",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps", "gpu"],
        default=os.getenv("BENCH_DEVICE", "auto"),
    )
    p.add_argument("--output", type=Path, default=Path(os.getenv("BENCH_OUTPUT", "results")))
    p.add_argument("--node", default="unspecified", help="Public-safe node label")
    p.add_argument("--condition", default="unspecified", help="User-managed operating condition")
    p.add_argument("--profile", choices=["standard", "diagnostic"], default="standard")
    p.add_argument("--data", type=Path)
    p.add_argument("--target")
    p.add_argument(
        "--features", help="Comma-separated numeric predictors; exclude target derivatives"
    )
    p.add_argument("--mode", choices=["train", "infer"], default=None)
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--threads",
        type=int,
        default=None,
        help="CPU threads; defaults to available physical cores",
    )
    p.add_argument(
        "--limit", type=int, default=2000, help="Maximum text rows or synthetic sklearn rows"
    )
    p.add_argument(
        "--synthetic-data", action="store_true", help="Label supplied toy/synthetic data explicitly"
    )
    p.add_argument(
        "--model",
        help="Local assets or minilm/distilbert/bert; defaults: embeddings=minilm, "
        "infer/finetune=distilbert. MiniLM does not support fill-mask inference.",
    )
    p.add_argument("--model-cache", type=Path, default=Path("data/models"))
    return p


def run_micro(args):
    started = time.perf_counter()
    if args.runtime == "mlx":
        from .mlx_runtime import MLXRuntime, validate

        validate(args)
        adapter = MLXRuntime
    else:
        adapter = TorchRuntime
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model, train, test, metadata = build(args)
    runtime = adapter(args, model)
    stateful = args.task == "stateful"

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
            runtime.step(*runtime.prepare(x, y))
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
            losses.append(runtime.step(x, y))
            runtime.synchronize()
            latencies.append(time.perf_counter() - start)
            rss.append(psutil.Process().memory_info().rss)
        total = time.perf_counter() - trial_start
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
    iterations = len(test[0]) if stateful else (len(test[0]) + args.batch - 1) // args.batch
    for i in range(iterations):
        if stateful:
            x, y = batch(test, i)
        else:
            x, y = (t[i * args.batch : (i + 1) * args.batch] for t in test)
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
    return {
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


def run(args):
    if args.profile == "diagnostic" and args.task not in ("embeddings", "infer", "finetune"):
        raise ValueError("Diagnostic phase profiling supports embeddings, infer and finetune")
    args.mode = args.mode or ("infer" if args.task in ("infer", "embeddings") else "train")
    args.repeats = args.repeats if args.repeats is not None else (1 if args.task in TASKS else 3)
    args.warmup = args.warmup if args.warmup is not None else (1 if args.task in TASKS else 5)
    args.threads = (
        args.threads if args.threads is not None else (psutil.cpu_count(logical=False) or 1)
    )
    for name in ("batch", "width", "length", "steps", "repeats", "threads", "limit"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name} must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be nonnegative")
    if args.width % 4:
        raise ValueError("--width must be divisible by 4")
    if args.task in TASKS and args.mode != (
        "infer" if args.task in ("infer", "embeddings") else "train"
    ):
        raise ValueError("The selected ladder task has a fixed mode; omit --mode")
    if args.runtime == "torch" and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("Disable PYTORCH_ENABLE_MPS_FALLBACK to avoid mixing CPU and GPU results")
    # Classification never samples an unrelated GPU selected by auto.
    if args.task == "classify" and args.device == "auto":
        args.device = "cpu"
    operation_start = time.perf_counter()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    sampler = Profiler(args)
    try:
        with sampler:
            if args.task in TASKS:
                from .operational import run as run_operational

                result = run_operational(args)
            else:
                result = run_micro(args)
    except (ValueError, RuntimeError, OSError, ImportError, MemoryError) as exc:
        exc.resources = sampler.summary()
        exc.wall_seconds = time.perf_counter() - operation_start
        raise
    if args.task in TASKS:
        result["wall_seconds"] = time.perf_counter() - operation_start
        units = (
            result["dataset"]["train_rows"] if args.task == "classify" else args.batch * args.steps
        ) * args.repeats
        result["job_samples_per_second"] = units / result["wall_seconds"]
    result["resources"] = sampler.summary()
    result.setdefault("status", "completed")
    result.setdefault("protocol", "micro-v2")
    from .reporting import annotate

    return annotate(result)


def save_failure(args, exc):
    args.output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = args.output / f"{stamp}-{args.task}-failed"
    reason = (
        "out_of_memory"
        if isinstance(exc, (torch.OutOfMemoryError, MemoryError))
        or "out of memory" in str(exc).lower()
        else "error"
    )
    # Describe actual available hardware without changing the requested execution backend.
    try:
        device = (
            select_device(args.device)
            if args.runtime == "torch" and args.device != "gpu"
            else torch.device("cpu")
        )
        hardware = identify(device)
    except (RuntimeError, ValueError):
        hardware = identify(torch.device("cpu"))
        hardware.update(gpu="unknown (requested backend unavailable)", gpu_vendor="unknown")
    record = {
        "schema_version": 1,
        "status": "failed",
        "hardware": hardware,
        "wall_seconds": getattr(exc, "wall_seconds", None),
        "reason": reason,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "resources": getattr(exc, "resources", {}),
        "git_commit": command("git", "rev-parse", "HEAD"),
    }
    from .reporting import annotate

    annotate(record)
    stem.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    stem.with_suffix(".txt").write_text(
        f"status=failed task={args.task} reason={reason}\n{exc}\ncpu={hardware['cpu']} arch={hardware['architecture']} gpu={hardware['gpu']}\nConfiguration and sampled resources: {stem.with_suffix('.json').name}\n"
    )
    return stem.with_suffix(".txt")


def save(result, output):
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = output / f"{stamp}-{result['config']['task']}-{result['device']}"
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    h, c = result["hardware"], result["config"]
    lines = [
        "runtime-bench schema=1",
        f"utc={result['utc']}",
        f"status={result.get('status', 'completed')} protocol={result.get('protocol', 'micro-v1')} runtime={result.get('runtime', 'torch')} task={c['task']} mode={c['mode']} device={result['device']} precision={c['precision']}",
        f"cpu={h['cpu']} vendor={h['cpu_vendor']} arch={h['architecture']}",
        f"gpu={h['gpu']} vendor={h['gpu_vendor']} unified_memory={h['unified_memory']}",
        f"ram_bytes={h['ram_bytes']} gpu_memory_bytes={h['gpu_memory_bytes']}",
        f"python={h['python']} torch={h['torch']} cuda={h['cuda_runtime']} mlx={h.get('mlx', 'not used')}",
        f"batch={c['batch']} width={c['width']} length={c['length']} steps={c['steps']} seed={c['seed']}",
        f"parameters={result['parameters']} synthetic={result['dataset']['synthetic']}",
        f"setup_seconds={result['setup_seconds']:.4f} wall_seconds={result['wall_seconds']:.4f}",
        f"job_samples_per_second={result.get('job_samples_per_second', 'not measured')}",
        f"model={result['dataset'].get('model_id', 'local estimator')} revision={result['dataset'].get('model_revision', 'not applicable')} transformers={h.get('transformers', 'not used')} sklearn={h.get('sklearn', 'not used')}",
        "quality=" + json.dumps(result["quality"]),
        "resources=" + json.dumps(result.get("resources", {})),
    ]
    for i, trial in enumerate(result["trials"], 1):
        lines.append(f"trial={i} " + " ".join(f"{k}={v}" for k, v in trial.items()))
    lines.append(
        "Full configuration, data fingerprint and software provenance: "
        + stem.with_suffix(".json").name
    )
    stem.with_suffix(".txt").write_text("\n".join(lines) + "\n")
    return stem.with_suffix(".txt")


def main():
    p = parser()
    args = p.parse_args()
    try:
        result = run(args)
        print(save(result, args.output))
    except (ValueError, RuntimeError, OSError, ImportError, MemoryError) as exc:
        path = save_failure(args, exc)
        p.exit(2, f"workload failed: {exc}\nFailure report: {path}\n")


if __name__ == "__main__":
    main()
