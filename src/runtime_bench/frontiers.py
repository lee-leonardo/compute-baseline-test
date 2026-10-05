"""Matched crossover observations and bounded one-axis policy refinement."""

import argparse
import copy
import hashlib
import json
import math
import statistics
import tempfile
from pathlib import Path

from .cli import parser, resolve_defaults
from .compare import compare
from . import saturation


AXES = ("batch", "length", "width")
REFINABLE_METRICS = {
    "wall_seconds",
    "compute_p95_ms",
    "batch_p95_ms",
    "process_rss_bytes",
    "cuda_allocated_bytes",
}


def threshold_only(reasons):
    return bool(reasons) and all(
        r.get("reason") == "threshold" and r.get("metric") in REFINABLE_METRICS for r in reasons
    )


def read_index(path):
    index = json.loads(path.read_text())
    if not isinstance(index, dict) or index.get("protocol") != "saturation-v1":
        raise ValueError("Expected saturation-v1 index")
    return index


def config(case):
    args = parser().parse_args(case["argv"])
    resolve_defaults(args)
    return {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in {"output", "node", "condition"}
    }


def ladder(index, name, axis):
    profiles = [p for p in index["plan"]["profiles"] if p["name"] == name]
    if len(profiles) != 1:
        raise ValueError(f"Unknown profile: {name}")
    profile = profiles[0]
    configs = [config(c) for c in profile["cases"]]
    values = [c[axis] for c in configs]
    fixed = [{k: v for k, v in c.items() if k != axis} for c in configs]
    if values != sorted(set(values)) or any(c != fixed[0] for c in fixed):
        raise ValueError(f"Stages must increase only {axis}, holding other options fixed")
    outcomes = next((p["stages"] for p in index["profiles"] if p["name"] == name), [])
    return profile, configs, {s["name"]: s for s in outcomes}


def load_report(outcome):
    paths = outcome.get("reports", [])
    if len(paths) != 1:
        raise ValueError("Expected one original workload report")
    return json.loads(Path(paths[0]).read_text())


def verify_case(report, case):
    expected = config(case)
    actual = report.get("config", {})
    for key, value in expected.items():
        if key == "checkpoint" and expected["mode"] == "train":
            continue
        if actual.get(key) != value:
            raise ValueError(f"Report differs from planned stage: {key}")


def crossover(index, baseline, candidate, axis, benefit=1.1, max_cv=0.1):
    if (
        baseline == candidate
        or not math.isfinite(benefit)
        or benefit <= 1
        or not math.isfinite(max_cv)
        or max_cv < 0
    ):
        raise ValueError("Choose different profiles, benefit > 1 and max_cv >= 0")
    a, ac, ao = ladder(index, baseline, axis)
    b, bc, bo = ladder(index, candidate, axis)
    if len(ac) != len(bc) or any(
        {k: v for k, v in x.items() if k not in {"runtime", "device"}}
        != {k: v for k, v in y.items() if k not in {"runtime", "device"}}
        for x, y in zip(ac, bc)
    ):
        raise ValueError("Profiles must use matched work, precision, thread budget and axis values")
    rows = []
    for ca, cb, options in zip(a["cases"], b["cases"], ac):
        row = {axis: options[axis], "classification": "untested"}
        sa, sb = ao.get(ca["name"], {}), bo.get(cb["name"], {})
        row["baseline_status"], row["candidate_status"] = sa.get("status"), sb.get("status")
        if sa.get("status") == sb.get("status") == "passed":
            ra, rb = load_report(sa), load_report(sb)
            verify_case(ra, ca)
            verify_case(rb, cb)
            if any(r.get("config", {}).get(axis) != options[axis] for r in (ra, rb)):
                raise ValueError("Report axis value differs from planned stage")
            if any(
                r.get("config", {}).get("condition") != index["condition"]
                or r.get("config", {}).get("node") != index["node"]
                for r in (ra, rb)
            ):
                raise ValueError("Report node/condition differs from campaign")
            ratios = compare(
                ra, rb
            )  # Protocol, seed, work, weights/data, revision and quality checks.
            key = (
                "samples_per_second"
                if ra["protocol"] in ("operational-v1", "embedding-v1", "pipeline-v1")
                else "loop_samples_per_second"
            )
            rates = [[t[key] for t in r["trials"]] for r in (ra, rb)]
            if any(not saturation.positive(v) for samples in rates for v in samples):
                raise ValueError("Crossover requires finite positive throughput")
            row.update(
                classification="ambiguous",
                ratios=ratios,
                execution_profiles=[f"{r.get('runtime')}:{r.get('device')}" for r in (ra, rb)],
                source_reports=[*sa["reports"], *sb["reports"]],
                dirty_source=any(r.get("git_dirty", True) for r in (ra, rb)),
            )
            spread_ok = all(
                len(samples) >= 3 and statistics.stdev(samples) / statistics.mean(samples) <= max_cv
                for samples in rates
            )
            lower = min(rates[1]) / max(rates[0])
            upper = max(rates[1]) / min(rates[0])
            row.update(observed_gain_range=[lower, upper], spread_acceptable=spread_ok)
            if spread_ok and lower >= benefit:
                row["classification"] = "candidate_beneficial"
            elif spread_ok and upper <= 1 / benefit:
                row["classification"] = "baseline_preferred"
        elif "stopped" in (sa.get("status"), sb.get("status")):
            row["classification"] = "gated"
        rows.append(row)
    intervals = [
        [left[axis], right[axis]]
        for left, right in zip(rows, rows[1:])
        if left["classification"] == "baseline_preferred"
        and right["classification"] == "candidate_beneficial"
    ]
    return {
        "protocol": "crossover-v1",
        "axis": axis,
        "baseline": baseline,
        "candidate": candidate,
        "min_speedup": benefit,
        "max_cv": max_cv,
        "rows": rows,
        "observed_transition_intervals": intervals,
        "interpretation": "Observed trial ranges, not confidence intervals. No monotonicity or untested crossover inferred; cross-runtime gains include software differences.",
    }


def refine(index, name, axis, output, resolution=1, max_probes=6, confirmations=2):
    if resolution < 1 or max_probes < 1 or confirmations < 2:
        raise ValueError("Require resolution >= 1, max_probes >= 1 and confirmations >= 2")
    profile, configs, outcomes = ladder(index, name, axis)
    if index.get("working_directory") != str(Path.cwd().resolve()):
        raise ValueError("Refinement requires the original working directory")
    lower = upper = None
    lower_case = None
    for case, options in zip(profile["cases"], configs):
        outcome = outcomes.get(case["name"], {})
        if outcome.get("status") == "passed":
            evidence = load_report(outcome)
            verify_case(evidence, case)
            if evidence.get("config", {}).get(axis) != options[axis] or saturation.evaluate(
                evidence, profile["gates"]
            ):
                raise ValueError("Passing endpoint report does not support the checkpoint")
            lower, lower_case = options[axis], case
        elif outcome.get("status") == "stopped":
            evidence = load_report(outcome)
            verify_case(evidence, case)
            reasons = saturation.evaluate(evidence, profile["gates"])
            if evidence.get("config", {}).get(axis) != options[axis]:
                raise ValueError("Stopped endpoint axis differs from the plan")
            if not threshold_only(reasons):
                raise ValueError(
                    "Refine only measured threshold stops, not OOM, errors or live guards"
                )
            upper = options[axis]
            break
        else:
            raise ValueError("Missing or skipped endpoint evidence")
    if lower is None or upper is None:
        raise ValueError("Need a passing stage followed by a measured threshold stop")
    lower_report = load_report(outcomes[lower_case["name"]])
    if evidence.get("git_commit") != lower_report.get("git_commit"):
        raise ValueError("Boundary endpoints must use the same source revision")
    stride = (
        4 if axis == "width" and configs[0]["task"] in {"transformer", "news", "stateful"} else 1
    )
    if resolution % stride or lower % stride or upper % stride:
        raise ValueError(f"Width endpoints and resolution must be multiples of {stride}")
    output.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="boundary-", dir=output)).resolve()
    state = {
        "protocol": "boundary-v1",
        "profile": name,
        "axis": axis,
        "status": "running",
        "original_interval": [lower, upper],
        "interval": [lower, upper],
        "resolution": resolution,
        "max_probes": max_probes,
        "confirmations": confirmations,
        "observations": [],
        "source_plan": index["plan"],
        "source_outcomes": outcomes,
        "node": index["node"],
        "condition": index["condition"],
        "interpretation": "Conditional one-axis bracket assuming locally monotone behavior and unchanged conditions; not an absolute hardware maximum.",
    }
    saturation.write_index(root, state)

    def measure(value):
        states = []
        for repeat in range(confirmations):
            case = copy.deepcopy(lower_case)
            case["name"] = f"{axis}-{value}-confirmation-{repeat + 1}"
            case["argv"] += [f"--{axis}", str(value)]
            plan = {
                "version": 1,
                "limits": index["plan"]["limits"],
                "gates": profile["gates"],
                "profiles": [{**profile, "cases": [case]}],
            }
            plan["id"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
            observation = {"value": value, "confirmation": repeat + 1, "status": "running"}
            state["observations"].append(observation)
            saturation.write_index(root, state)
            folder, result = saturation.execute(plan, root, index["node"], index["condition"])
            stage = result["profiles"][0]["stages"][0]
            reasons = stage.get("decisions", [])
            status = (
                "pass"
                if stage["status"] == "passed"
                else "threshold"
                if threshold_only(reasons)
                else "blocked"
            )
            if value == state["original_interval"][0] and status == "pass":
                fresh = load_report(stage)
                try:
                    compare(lower_report, fresh)
                    if lower_report.get("hardware") != fresh.get("hardware"):
                        raise ValueError("Hardware/environment differs from source endpoint")
                except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
                    status = "blocked"
                    reasons = [{"reason": "endpoint_provenance_changed", "detail": str(exc)}]
            observation.update(status=status, index=str(folder / "index.json"), decisions=reasons)
            saturation.write_index(root, state)
            states.append(status)
            if status == "blocked":
                return "blocked"
        return states[0] if len(set(states)) == 1 else "ambiguous"

    try:
        # Recheck the safe endpoint; never rerun a known heavier stopped endpoint.
        if measure(lower) != "pass":
            state["status"] = "baseline_not_reproduced"
        else:
            for _ in range(max_probes):
                if upper - lower <= resolution:
                    break
                midpoint = ((lower + upper) // (2 * stride)) * stride
                observed = measure(midpoint)
                if observed not in {"pass", "threshold"}:
                    state["status"] = observed
                    break
                if observed == "pass":
                    lower = midpoint
                else:
                    upper = midpoint
                state["interval"] = [lower, upper]
                saturation.write_index(root, state)
            if state["status"] == "running":
                state["status"] = (
                    "resolution_reached"
                    if upper - lower <= resolution
                    else "probe_budget_exhausted"
                )
    except BaseException:
        state["status"] = "interrupted"
        saturation.write_index(root, state)
        raise
    saturation.write_index(root, state)
    return root, state


def crossover_main():
    p = argparse.ArgumentParser(
        description="Analyze matched crossover observations without new compute"
    )
    p.add_argument("index", type=Path)
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--axis", choices=AXES, default="batch")
    p.add_argument("--min-speedup", type=float, default=1.1)
    p.add_argument("--max-cv", type=float, default=0.1)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    try:
        result = crossover(
            read_index(args.index),
            args.baseline,
            args.candidate,
            args.axis,
            args.min_speedup,
            args.max_cv,
        )
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        print(text)
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        p.exit(2, f"crossover failed: {exc}\n")


def boundary_main():
    p = argparse.ArgumentParser(
        description="Refine one-axis policy boundaries with bounded confirmation probes"
    )
    p.add_argument("index", type=Path)
    p.add_argument("--profile", required=True)
    p.add_argument("--axis", choices=AXES, default="batch")
    p.add_argument("--resolution", type=int, default=1)
    p.add_argument("--max-probes", type=int, default=6)
    p.add_argument("--confirmations", type=int, default=2)
    p.add_argument("--output", type=Path, default=Path("results"))
    args = p.parse_args()
    try:
        root, state = refine(
            read_index(args.index),
            args.profile,
            args.axis,
            args.output,
            args.resolution,
            args.max_probes,
            args.confirmations,
        )
        print(root / "index.json")
        if state["status"] not in {"resolution_reached", "probe_budget_exhausted"}:
            p.exit(1, f"Refinement stopped: {state['status']}\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.exit(2, f"boundary failed: {exc}\n")
    except KeyboardInterrupt:
        p.exit(130, "Boundary interrupted; child and index handled by saturation guards.\n")
