"""Compare compatible reports without silently pooling different experiments."""

import argparse
import json
import statistics
from pathlib import Path


MATCH_KEYS = (
    "profile",
    "task",
    "mode",
    "precision",
    "batch",
    "width",
    "length",
    "steps",
    "warmup",
    "repeats",
    "seed",
    "threads",
    "target",
    "features",
    "limit",
    "model",
)


def compare(baseline, candidate):
    for report in (baseline, candidate):
        if (
            report.get("schema_version") != 1
            or not report.get("trials")
            or report.get("status", "completed") != "completed"
        ):
            raise ValueError("Expected a schema 1 report with completed trials")
    if baseline.get("protocol", "micro-v1") != candidate.get("protocol", "micro-v1"):
        raise ValueError("Incompatible measurement protocols")
    operational = baseline.get("protocol") in ("operational-v1", "embedding-v1")
    keys = [
        key for key in MATCH_KEYS if not operational or key not in ("threads", "width", "model")
    ]
    differences = [
        key
        for key in keys
        if baseline["config"].get(key, "standard" if key == "profile" else None)
        != candidate["config"].get(key, "standard" if key == "profile" else None)
    ]
    if baseline["dataset"] != candidate["dataset"]:
        differences.append("dataset fingerprint / preprocessing")
    if baseline["parameters"] != candidate["parameters"]:
        differences.append("parameter count")
    if baseline["git_commit"] != candidate["git_commit"]:
        differences.append("git revision")
    if differences:
        raise ValueError("Incompatible reports: " + ", ".join(differences))

    def median(report, key):
        return statistics.median(t[key] for t in report["trials"])

    operational = baseline.get("protocol") in ("operational-v1", "embedding-v1")
    latency = "batch_median_ms" if operational else "compute_median_ms"
    throughput = "samples_per_second" if operational else "loop_samples_per_second"
    result = {
        "batch_latency_speedup" if operational else "compute_latency_speedup": median(
            baseline, latency
        )
        / median(candidate, latency),
        "loop_throughput_gain": median(candidate, throughput) / median(baseline, throughput),
        "job_wall_time_speedup": baseline["wall_seconds"] / candidate["wall_seconds"],
    }
    # Dispersion remains separate from speedup, based on independent reset trials.
    for label, report in (("baseline", baseline), ("candidate", candidate)):
        rates = [t[throughput] for t in report["trials"]]
        result[f"{label}_throughput_min"] = min(rates)
        result[f"{label}_throughput_max"] = max(rates)
        result[f"{label}_throughput_cv"] = (
            statistics.stdev(rates) / statistics.mean(rates) if len(rates) > 1 else None
        )
    for key in (
        "accuracy",
        "cross_entropy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "retrieval_recall_at_1",
        "retrieval_mrr",
    ):
        a, b = baseline["quality"].get(key), candidate["quality"].get(key)
        result[f"{key}_change"] = b - a if a is not None and b is not None else None
    return result


def capacity_line(report):
    c, r = report["config"], report.get("resources", {})
    trials = report.get("trials", [])
    gpu_peak = max((t.get("cuda_peak_allocated_bytes") or 0 for t in trials), default=0) or None
    rate_key = (
        "samples_per_second"
        if report.get("protocol") in ("operational-v1", "embedding-v1")
        else "loop_samples_per_second"
    )
    throughput = statistics.median(t[rate_key] for t in trials) if trials else None
    return (
        f"status={report.get('status', 'completed')} task={c['task']} model={report.get('dataset', {}).get('model_id', c.get('model'))} "
        f"device={report.get('device', c.get('device'))} batch={c['batch']} length={c['length']} "
        f"wall_seconds={report.get('wall_seconds')} samples_per_second={throughput} "
        f"ram_sampled_peak_bytes={r.get('process_rss_sampled_peak_bytes')} "
        f"cuda_peak_allocated_bytes={gpu_peak} reason={report.get('reason')}"
    )


def main():
    """Parse report paths, print compatible ratios, and exit 2 on invalid comparisons."""
    p = argparse.ArgumentParser(description="Candidate gains relative to a matching baseline")
    p.add_argument("baseline", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument(
        "--capacity",
        action="store_true",
        help="Show outcomes/configurations side by side, including failures; no speedup claim",
    )
    args = p.parse_args()
    try:
        baseline, candidate = (
            json.loads(path.read_text()) for path in (args.baseline, args.candidate)
        )
        if args.capacity:
            print("baseline: " + capacity_line(baseline))
            print("candidate: " + capacity_line(candidate))
            return
        result = compare(baseline, candidate)
        for label, report in (("baseline", baseline), ("candidate", candidate)):
            hardware = report["hardware"]
            print(
                f"{label}: {hardware['cpu']} / {hardware['gpu']} / {report.get('runtime', 'torch')}:{report['device']}"
            )
        if baseline.get("runtime", "torch") != candidate.get("runtime", "torch"):
            print("Cross-framework comparison: gains include runtime and kernel differences.")
        if baseline["config"].get("threads") != candidate["config"].get("threads"):
            print("CPU thread budgets differ; operational ratios include this resource difference.")
        for key, value in result.items():
            print(f"{key}={value:.6f}" if value is not None else f"{key}=unavailable")
        if baseline["config"].get("condition") != candidate["config"].get("condition"):
            print("Operating conditions differ; ratios include contention differences.")
        print(
            "Speedup/gain ratios >1 favor candidate. Review trial spread, software differences and quality before conclusions."
        )
        if baseline["git_dirty"] or candidate["git_dirty"]:
            print("Warning: at least one checkout was dirty; manually verify identical source.")
    except (ValueError, OSError, KeyError, ZeroDivisionError) as exc:
        p.exit(2, f"comparison failed: {exc}\n")


if __name__ == "__main__":
    main()
