"""Command parsing, configuration validation, workload dispatch, and report output.

Use ``main`` for the public CLI and ``parser``/``run`` for flat manifest cases.
Device execution lives in runtime adapters; trial timing lives in ``micro`` and
``operational``. Expected failures produce a report and exit status 2.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch

from .hardware import command, identify, select_device
from .micro import run_micro
from .profiling import Profiler
from .operational import TASKS


def parser():
    """Build the backward-compatible flat parser used by manifest cases."""
    p = argparse.ArgumentParser(
        description="Run one workload. Prefer classification train/infer for checkpoint workflows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "task", choices=["classification", "coffee", "news", "stateful", "smoke", *TASKS]
    )
    p.add_argument(
        "--runtime",
        choices=["torch", "mlx"],
        default="torch",
        help="Execution adapter (default: torch); mlx supports classification/coffee/smoke",
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
    p.add_argument(
        "--checkpoint", type=Path, help="Classification: output for train, input for infer"
    )
    p.add_argument("--model-cache", type=Path, default=Path("data/models"))
    descriptions = {
        "task": "Workload to execute (classify is the separate sklearn baseline)",
        "device": "Execution device; auto selects an available backend",
        "output": "Directory for text and JSON reports",
        "profile": "Diagnostic NLP phase timing adds synchronization overhead",
        "data": "Input CSV; required for CSV and text workloads",
        "target": "Categorical target column in a tabular CSV",
        "mode": "Learning examples: train or infer; NLP tasks have fixed modes",
        "precision": "FP32 baseline; FP16/BF16 require compatible CUDA",
        "batch": "Examples per measured batch",
        "width": "Learning model hidden width; Transformers require a multiple of four",
        "length": "Padded sequence length for text/sequence workloads",
        "steps": "Batches per trial; classify uses boosting iterations",
        "warmup": "Discarded warmup batches (micro: 5, operational: 1)",
        "repeats": "Independent reset trials (micro: 3, operational: 1)",
        "seed": "Initialization and split seed",
        "model_cache": "Local pretrained model cache populated by fetch",
    }
    for action in p._actions:
        if action.help is None:
            action.help = descriptions.get(action.dest)
    return p


def run(args):
    """Resolve defaults, validate a request, sample resources, and dispatch one job."""
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
    if args.task in ("news", "stateful") and args.width % 4:
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
    if args.task == "classification":
        from .classification import validate

        validate(args)
    elif args.checkpoint is not None:
        raise ValueError("--checkpoint is supported only by classification")
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
    """Write a failed outcome without changing the requested execution backend."""
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
    """Write machine-readable JSON and a human-readable summary; return the text path."""
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
    if result.get("checkpoint"):
        lines.append("checkpoint=" + json.dumps(result["checkpoint"]))
    for i, trial in enumerate(result["trials"], 1):
        lines.append(f"trial={i} " + " ".join(f"{k}={v}" for k, v in trial.items()))
    lines.append(
        "Full configuration, data fingerprint and software provenance: "
        + stem.with_suffix(".json").name
    )
    stem.with_suffix(".txt").write_text("\n".join(lines) + "\n")
    return stem.with_suffix(".txt")


def classification_parser():
    """Expose only classification options, with separate train/infer help pages.

    Both subcommands resolve to the same flat Namespace as manifest cases. The
    inference page omits architecture and preprocessing knobs because the saved
    checkpoint owns those values.
    """
    flat = parser()
    defaults = vars(flat.parse_args(["classification"]))
    p = argparse.ArgumentParser(
        prog="runtime-bench classification",
        description="Train a tabular MLP or benchmark its saved weights on held-out data.",
    )
    modes = p.add_subparsers(dest="mode", required=True)
    shared = {
        "runtime",
        "device",
        "precision",
        "batch",
        "steps",
        "warmup",
        "repeats",
        "threads",
        "output",
        "node",
        "condition",
        "data",
        "checkpoint",
    }
    for mode in ("train", "infer"):
        sub = modes.add_parser(
            mode,
            help="Fit and save an MLP" if mode == "train" else "Load and benchmark an MLP",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        sub.set_defaults(**(defaults | {"mode": mode}))
        groups = {
            "data": sub.add_argument_group("data and checkpoint"),
            "execution": sub.add_argument_group("execution"),
            "budget": sub.add_argument_group("work budget"),
            "reports": sub.add_argument_group("reports and labels"),
        }
        selected = shared | (
            {"target", "features", "width", "seed", "synthetic_data"} if mode == "train" else set()
        )
        for action in flat._actions:
            if action.dest not in selected:
                continue
            kwargs = {"default": action.default, "help": action.help}
            if isinstance(action, argparse._StoreTrueAction):
                kwargs["action"] = "store_true"
            else:
                kwargs.update(type=action.type, choices=action.choices)
            focused_help = {
                "data": "Original labeled CSV; omit for the synthetic fixture",
                "runtime": "Execution adapter: torch or native Apple MLX (FP32)",
                "device": "Torch: cpu/cuda/mps; MLX: cpu/gpu; auto is runtime-specific",
                "steps": "Measured batches per reset trial",
                "warmup": "Discarded warmup batches before each measured trial",
                "repeats": "Independent trials, each reset to the same starting weights",
            }
            if action.dest in focused_help:
                kwargs["help"] = focused_help[action.dest]
            if action.dest in ("warmup", "repeats"):
                kwargs["default"] = 5 if action.dest == "warmup" else 3
            if action.dest == "device":
                kwargs["choices"] = ["auto", "cpu", "cuda", "mps", "gpu"]
            if action.dest == "checkpoint":
                kwargs["required"] = mode == "infer"
                kwargs["help"] = (
                    "New checkpoint path (default: timestamped file in output)"
                    if mode == "train"
                    else "Checkpoint written by classification train"
                )
            group = (
                "data"
                if action.dest in ("data", "target", "features", "checkpoint", "synthetic_data")
                else "reports"
                if action.dest in ("output", "node", "condition")
                else "execution"
                if action.dest in ("runtime", "device", "precision", "threads")
                else "budget"
            )
            groups[group].add_argument(*action.option_strings, **kwargs)
    return p


def command_parser():
    """Build the short command index; individual commands own their detailed help."""
    p = argparse.ArgumentParser(
        prog="runtime-bench",
        description="Reproducible ML workload profiling on local CPU and GPU runtimes.",
        epilog="Use runtime-bench COMMAND --help. Existing flat workload commands remain supported.",
    )
    commands = p.add_subparsers(dest="command")
    for name, help_text in {
        "classification": "Train a classifier or infer from a saved checkpoint",
        "run": "Run an existing workload with advanced options",
        "suite": "Run a TOML experiment manifest",
        "fetch": "Download dataset or pretrained model assets",
        "compare": "Compare two report files",
        "export": "Export report files to per-trial CSV",
    }.items():
        commands.add_parser(name, help=help_text, add_help=False)
    return p


def main():
    """Dispatch a public command; preserve old entry points and failure exit codes.

    Argument errors exit 2. Expected runtime errors also exit 2 after writing a
    failure report. Suite commands retain their own exit 1 for failed cases.
    """
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        command_parser().print_help()
        return
    if argv[0] in ("suite", "fetch", "compare", "export"):
        from . import compare, experiments, model_specs, reporting

        entry = {
            "suite": experiments.main,
            "fetch": model_specs.main,
            "compare": compare.main,
            "export": reporting.main,
        }[argv[0]]
        original = sys.argv
        try:
            sys.argv = [f"runtime-bench {argv[0]}", *argv[1:]]
            return entry()
        finally:
            sys.argv = original
    if argv[0] == "classification":
        # Flat form remains available for manifests: classification --mode train ...
        if len(argv) > 1 and argv[1] not in ("train", "infer", "-h", "--help"):
            p, arguments = parser(), argv
        else:
            p, arguments = classification_parser(), argv[1:]
    else:
        p, arguments = parser(), argv[1:] if argv[0] == "run" else argv
    args = p.parse_args(arguments)
    try:
        result = run(args)
        print(save(result, args.output))
        if result.get("checkpoint"):
            print(f"Checkpoint: {result['checkpoint']['path']}")
    except (ValueError, RuntimeError, OSError, ImportError, MemoryError) as exc:
        path = save_failure(args, exc)
        p.exit(2, f"workload failed: {exc}\nFailure report: {path}\n")


if __name__ == "__main__":
    main()
