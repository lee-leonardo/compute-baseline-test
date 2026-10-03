"""Portable experiment identities and lossless, per-trial table export."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


EXECUTION_KEYS = {
    "runtime",
    "device",
    "threads",
    "output",
    "data",
    "model_cache",
    "model",
    "node",
    "condition",
}


def annotate(report):
    config = report["config"]
    identity = {
        "protocol": report.get("protocol"),
        "config": {k: v for k, v in config.items() if k not in EXECUTION_KEYS},
        "dataset": report.get("dataset"),
        "parameters": report.get("parameters"),
    }
    # Failed jobs have no verified dataset/model identity.
    report["experiment_id"] = (
        hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if report.get("status") == "completed"
        else None
    )
    report["report_revision"] = 2
    report["labels"] = {key: config.get(key) for key in ("node", "condition")}
    return report


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        else:
            result[name] = json.dumps(item) if isinstance(item, list) else item
    return result


def export(paths, output):
    rows = []
    for path in paths:
        report = json.loads(path.read_text())
        if report.get("schema_version") != 1 or "config" not in report:
            raise ValueError(f"Not a benchmark report: {path}")
        base = flatten({k: v for k, v in report.items() if k != "trials"})
        for index, trial in enumerate(report.get("trials") or [None], 1):
            rows.append(
                {
                    "source": str(path),
                    "trial_index": index if trial is not None else None,
                    **base,
                    **flatten({"trial": trial or {}}),
                }
            )
    if not rows:
        raise ValueError("No reports supplied")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    p = argparse.ArgumentParser(description="Export benchmark JSON reports to one row per trial")
    p.add_argument("reports", nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    try:
        print(f"Exported {export(args.reports, args.output)} rows to {args.output}")
    except (OSError, ValueError, TypeError) as exc:
        p.exit(2, f"export failed: {exc}\n")
