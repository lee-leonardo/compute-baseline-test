"""Ordered, isolated workload escalation with auditable per-profile stop gates."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

import psutil

from .cli import parser, resolve_defaults
from .experiments import expand_manifest


# Each memory domain remains independent. No inferred VRAM or utilization proxy.
METRICS = {
    "wall_seconds": ("wall_seconds", "max"),
    "batch_p95_ms": ("trials.batch_p95_ms", "max"),
    "compute_p95_ms": ("trials.compute_p95_ms", "max"),
    "process_rss_bytes": ("resources.process_rss_sampled_peak_bytes", "max"),
    "host_available_bytes": ("resources.host_available_ram_min_bytes", "min"),
    "host_swap_growth_bytes": ("resources.host_swap_used_change_bytes", "max"),
    "cuda_allocated_bytes": ("trials.cuda_peak_allocated_bytes", "max"),
    "gpu_device_used_bytes": ("resources.gpu_device_used_sampled_peak_bytes", "max"),
}
LIVE_KEYS = {"timeout_seconds", "min_host_available_bytes", "max_process_rss_bytes"}


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def validate_gates(gates):
    if not isinstance(gates, dict) or set(gates) - METRICS.keys():
        raise ValueError(f"Gates support: {', '.join(METRICS)}")
    if not all(positive(v) for v in gates.values()):
        raise ValueError("Gate thresholds must be finite positive numbers")


def load_plan(path):
    raw = tomllib.loads(path.read_text())
    if set(raw) - {"version", "defaults", "profiles", "stages", "limits", "gates"}:
        raise ValueError("Unknown saturation manifest fields")
    if raw.get("version") != 1:
        raise ValueError("Expected saturation manifest version=1")
    limits = raw.get("limits", {})
    if (
        not isinstance(limits, dict)
        or set(limits) - LIVE_KEYS
        or not {"timeout_seconds", "min_host_available_bytes"} <= limits.keys()
        or not all(positive(v) for v in limits.values())
    ):
        raise ValueError("Set positive timeout_seconds and min_host_available_bytes in limits")
    gates = raw.get("gates", {})
    validate_gates(gates)
    profiles, stages = raw.get("profiles"), raw.get("stages")
    for name, rows in (("profiles", profiles), ("stages", stages)):
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Provide at least one [[{name}]] table")
        names = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not row["name"]:
                raise ValueError(f"Each {name} entry needs a nonempty name")
            names.append(row["name"])
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate {name} names")
    defaults = raw.get("defaults", {})
    if not isinstance(defaults, dict) or set(defaults) & {"runtime", "device", "precision"}:
        raise ValueError("Runtime, device and precision belong in profiles")
    if set(defaults) & {"sweep", "name"}:
        raise ValueError("Defaults cannot define names or sweeps; list ordered stages explicitly")
    planned = []
    for profile in profiles:
        if set(profile) - {"name", "runtime", "device", "precision", "threads", "gates"}:
            raise ValueError("Unknown profile field")
        if profile.get("device") not in {"cpu", "cuda", "mps", "gpu"}:
            raise ValueError("Each profile requires an explicit device; auto is forbidden")
        if profile.get("runtime") not in {"torch", "mlx"}:
            raise ValueError("Each profile requires an explicit runtime")
        if (profile["runtime"] == "torch" and profile["device"] == "gpu") or (
            profile["runtime"] == "mlx" and profile["device"] not in {"cpu", "gpu"}
        ):
            raise ValueError("Incompatible profile runtime/device")
        validate_gates(profile.get("gates", {}))
        profile_gates = gates | profile.get("gates", {})
        options = {k: v for k, v in profile.items() if k not in {"name", "gates"}}
        options.setdefault("precision", "fp32")
        cases = []
        for stage in stages:
            if set(stage) - {"name", "options"} or not isinstance(stage.get("options"), dict):
                raise ValueError("Each stage supports only name and an options table")
            if set(stage["options"]) & {"runtime", "device", "precision", "threads", "sweep"}:
                raise ValueError(
                    "Stages cannot override compute profiles or contain unordered sweeps"
                )
            case = defaults | stage["options"] | options | {"name": stage["name"]}
            cases.extend(expand_manifest({"version": 1, "cases": [case]}))
        for case in cases:
            args = parser().parse_args(case["argv"])
            for key in ("batch", "width", "length", "steps", "limit", "threads", "repeats"):
                value = getattr(args, key)
                if value is not None and value < 1:
                    raise ValueError(f"{key} must be positive")
            if args.warmup is not None and args.warmup < 0:
                raise ValueError("warmup must be nonnegative")
            # Writing the same checkpoint at escalating stages would destroy evidence.
            if args.checkpoint is not None and args.mode != "infer":
                raise ValueError(
                    "Saturation training must omit checkpoint (no shared output weights)"
                )
        planned.append({"name": profile["name"], "cases": cases, "gates": profile_gates})
    if sum(len(p["cases"]) for p in planned) > 256:
        raise ValueError("Limit a saturation plan to 256 profile/stage pairs")
    plan = {"version": 1, "limits": limits, "gates": gates, "profiles": planned}
    plan["id"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return plan


def values_at(value, keys):
    if isinstance(value, list):
        return [v for item in value for v in values_at(item, keys)]
    if not keys:
        return [value]
    if not isinstance(value, dict) or keys[0] not in value:
        return [None]
    return values_at(value[keys[0]], keys[1:])


def evaluate(report, gates):
    """Missing/invalid metrics block escalation; every trial must provide evidence."""
    if report.get("status") != "completed":
        return [
            {
                "reason": "out_of_memory"
                if report.get("reason") == "out_of_memory"
                else "execution_error"
            }
        ]
    decisions = []
    for name, threshold in gates.items():
        path, direction = METRICS[name]
        values = values_at(report, path.split("."))
        if not values or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            decisions.append({"reason": "missing_metric", "metric": name, "threshold": threshold})
            continue
        value = min(values) if direction == "min" else max(values)
        crossed = value <= threshold if direction == "min" else value >= threshold
        if crossed:
            decisions.append(
                {"reason": "threshold", "metric": name, "observed": value, "threshold": threshold}
            )
    return decisions


def supervise(command, folder, limits):
    """Bound a fresh child including startup/setup; sampled guards are best effort."""
    available = psutil.virtual_memory().available
    if available <= limits["min_host_available_bytes"]:
        return None, [{"reason": "host_memory_guard", "observed": available}]
    started = time.monotonic()
    with (folder / "console.log").open("w") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            try:
                process = psutil.Process(child.pid)
            except psutil.NoSuchProcess:
                return child.wait(), []
            while child.poll() is None:
                reason = None
                elapsed = time.monotonic() - started
                available = psutil.virtual_memory().available
                if elapsed >= limits["timeout_seconds"]:
                    reason = {"reason": "timeout", "observed": elapsed}
                elif available <= limits["min_host_available_bytes"]:
                    reason = {"reason": "host_memory_guard", "observed": available}
                elif "max_process_rss_bytes" in limits:
                    try:
                        rss = process.memory_info().rss
                    except psutil.NoSuchProcess:
                        break
                    if rss >= limits["max_process_rss_bytes"]:
                        reason = {"reason": "process_memory_guard", "observed": rss}
                if reason:
                    child.kill()
                    child.wait()
                    return child.returncode, [reason]
                time.sleep(0.1)
            return child.wait(), []
        finally:
            if child.poll() is None:
                child.kill()
            child.wait()


def write_index(root, index):
    temporary = root / "index.tmp"
    temporary.write_text(json.dumps(index, indent=2, allow_nan=False) + "\n")
    temporary.replace(root / "index.json")


def prior_stops(path, plan, node, condition):
    """Only carry forward stops; historical successes never qualify new work."""
    if path is None:
        return {}
    prior = json.loads(path.read_text())
    if (
        not isinstance(prior, dict)
        or prior.get("protocol") != "saturation-v1"
        or prior.get("plan") != plan
        or prior.get("node") != node
        or prior.get("condition") != condition
        or prior.get("working_directory") != str(Path.cwd().resolve())
    ):
        raise ValueError(
            "Prior index must match the complete plan, node, condition and working directory"
        )
    return {p["name"]: p for p in prior["profiles"] if p["status"] != "exhausted"}


def baseline_evidence(specs, plan, node, condition):
    """Import matching reports only as conservative stop evidence, never as a pass."""
    evidence = {}
    profiles = {p["name"]: p for p in plan["profiles"]}
    for spec in specs or []:
        name, separator, filename = spec.partition("=")
        if not separator or name not in profiles or name in evidence:
            raise ValueError("Baseline must be unique PROFILE=REPORT for a planned profile")
        path = Path(filename).resolve()
        contents = path.read_bytes()
        report = json.loads(contents)
        profile = profiles[name]
        args = parser().parse_args(profile["cases"][0]["argv"])
        resolve_defaults(args)
        expected = vars(args) | {"node": node, "condition": condition}
        if not isinstance(report, dict):
            raise ValueError(f"Invalid baseline report for {name}")
        config = report.get("config", {})
        if report.get("schema_version") != 1 or not isinstance(config, dict):
            raise ValueError(f"Invalid baseline report for {name}")
        for key, value in expected.items():
            if key == "output" or (key == "checkpoint" and args.mode == "train"):
                continue
            actual = config.get(key)
            if isinstance(value, Path):
                matches = isinstance(actual, str) and Path(actual).resolve() == value.resolve()
            else:
                matches = actual == value
            if not matches:
                raise ValueError(f"Baseline {name} differs from first stage: {key}")
        gates = dict(profile["gates"])
        limits = plan["limits"]
        gates["wall_seconds"] = min(gates.get("wall_seconds", math.inf), limits["timeout_seconds"])
        gates["host_available_bytes"] = max(
            gates.get("host_available_bytes", 0), limits["min_host_available_bytes"]
        )
        if "max_process_rss_bytes" in limits:
            gates["process_rss_bytes"] = min(
                gates.get("process_rss_bytes", math.inf), limits["max_process_rss_bytes"]
            )
        evidence[name] = {
            "source": str(path),
            "sha256": hashlib.sha256(contents).hexdigest(),
            "decisions": evaluate(report, gates),
        }
    return evidence


def execute(plan, output, node, condition, prior=None, baselines=None):
    stops = prior_stops(prior, plan, node, condition)
    baselines = baseline_evidence(baselines, plan, node, condition)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="saturation-", dir=output)).resolve()
    index = {
        "protocol": "saturation-v1",
        "plan": plan,
        "node": node,
        "condition": condition,
        "working_directory": str(Path.cwd().resolve()),
        "status": "running",
        "profiles": [],
    }
    write_index(root, index)
    try:
        for number, profile in enumerate(plan["profiles"], 1):
            row = {"name": profile["name"], "status": "running", "last_passed": None, "stages": []}
            index["profiles"].append(row)
            stopped = profile["name"] in stops
            if stopped:
                row.update(
                    status="stopped",
                    prior_index=str(prior.resolve()),
                    stop=[{"reason": "prior_stop", "evidence": stops[profile["name"]]}],
                )
            if profile["name"] in baselines:
                evidence = baselines[profile["name"]]
                row["baseline_evidence"] = evidence
                if evidence["decisions"]:
                    row.update(status="stopped", first_stopped=profile["cases"][0]["name"])
                    row.setdefault("stop", []).append(
                        {"reason": "prior_baseline", "evidence": evidence}
                    )
                    stopped = True
            for step, case in enumerate(profile["cases"], 1):
                outcome = {
                    "name": case["name"],
                    "argv": case["argv"],
                    "status": "skipped" if stopped else "running",
                }
                row["stages"].append(outcome)
                write_index(root, index)
                if stopped:
                    continue
                folder = root / f"{number:03d}-{step:03d}"
                folder.mkdir()
                command = [
                    sys.executable,
                    "-m",
                    "runtime_bench.cli",
                    "run",
                    *case["argv"],
                    "--output",
                    str(folder),
                    "--node",
                    node,
                    "--condition",
                    condition,
                ]
                print(f"{profile['name']} / {case['name']}", flush=True)
                code, reasons = supervise(command, folder, plan["limits"])
                reports = sorted(folder.glob("*.json"))
                outcome.update(
                    exit_code=code, directory=str(folder), reports=[str(p) for p in reports]
                )
                if not reasons:
                    if len(reports) != 1:
                        reasons = [{"reason": "missing_report"}]
                    else:
                        try:
                            report = json.loads(reports[0].read_text())
                            reasons = evaluate(report, profile["gates"])
                        except (ValueError, TypeError, AttributeError):
                            reasons = [{"reason": "invalid_report"}]
                    if code and not reasons:
                        reasons = [{"reason": "execution_error", "exit_code": code}]
                outcome.update(status="stopped" if reasons else "passed", decisions=reasons)
                if reasons:
                    row.update(status="stopped", stop=reasons, first_stopped=case["name"])
                    stopped = True
                else:
                    row["last_passed"] = case["name"]
                write_index(root, index)
            if not stopped:
                row["status"] = "exhausted"
            write_index(root, index)
        index["status"] = "completed"
    except BaseException:
        index["status"] = "interrupted"
        write_index(root, index)
        raise
    write_index(root, index)
    return root, index


def main():
    p = argparse.ArgumentParser(description="Ordered saturation with per-profile checkpoint gates")
    commands = p.add_subparsers(dest="action", required=True)
    plan_parser = commands.add_parser("plan", help="Validate and print stages without computing")
    plan_parser.add_argument("manifest", type=Path)
    run_parser = commands.add_parser(
        "run", help="Execute until each profile stops or exhausts its stages"
    )
    run_parser.add_argument("manifest", type=Path)
    run_parser.add_argument("--node", required=True)
    run_parser.add_argument("--condition", default="unspecified")
    run_parser.add_argument("--output", type=Path, default=Path("results"))
    run_parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        metavar="PROFILE=REPORT",
        help="Use an existing matching baseline as stop evidence (repeat per profile)",
    )
    run_parser.add_argument(
        "--prior-index", type=Path, help="Carry stops forward from an identical plan"
    )
    status_parser = commands.add_parser(
        "status", help="Inspect recorded boundaries without computing"
    )
    status_parser.add_argument("index", type=Path)
    args = p.parse_args()
    try:
        if args.action == "status":
            index = json.loads(args.index.read_text())
            if not isinstance(index, dict) or index.get("protocol") != "saturation-v1":
                raise ValueError("Expected a saturation-v1 index")
            print(
                json.dumps(
                    {k: index[k] for k in ("status", "node", "condition", "profiles")}, indent=2
                )
            )
        elif args.action == "plan":
            print(json.dumps(load_plan(args.manifest), indent=2))
        else:
            root, index = execute(
                load_plan(args.manifest),
                args.output,
                args.node,
                args.condition,
                args.prior_index,
                args.baseline,
            )
            print(root / "index.json")
            if any(row["status"] == "stopped" for row in index["profiles"]):
                p.exit(1, "Profiles stopped at checkpoint gates; inspect the index.\n")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        p.exit(2, f"saturation failed: {exc}\n")
    except KeyboardInterrupt:
        p.exit(130, "Saturation interrupted; child stopped and index preserved.\n")
