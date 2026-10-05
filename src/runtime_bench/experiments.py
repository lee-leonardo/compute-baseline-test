"""Small TOML experiment runner. Each case executes in a fresh process."""

import argparse
import itertools
import json
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

from .cli import parser


def expand(path):
    """Validate TOML options and expand sweeps before starting any child process."""
    return expand_manifest(tomllib.loads(path.read_text()))


def expand_manifest(manifest):
    """Expand an in-memory manifest using the same validation as a suite file."""
    if set(manifest) - {"version", "defaults", "cases"} or manifest.get("version") != 1:
        raise ValueError("Expected manifest version=1 with defaults and cases")
    defaults = manifest.get("defaults", {})
    cases = manifest.get("cases", [])
    if not isinstance(defaults, dict) or not isinstance(cases, list) or not cases:
        raise ValueError("Provide defaults table and at least one [[cases]] table")
    cli = parser()
    actions = {a.dest: a for a in cli._actions if a.option_strings}
    allowed = set(actions) - {"help", "output", "node", "condition"}
    expanded = []
    names = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each case must be a table")
        case = dict(case)
        name = case.pop("name", None)
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Case names must be unique nonempty strings")
        names.add(name)
        sweep = case.pop("sweep", {})
        config = defaults | case
        task = config.pop("task", None)
        if task not in next(a.choices for a in cli._actions if a.dest == "task"):
            raise ValueError(f"Unknown task in {name}: {task}")
        if set(config) - allowed:
            raise ValueError(f"Unknown or reserved options in {name}: {set(config) - allowed}")
        if not isinstance(sweep, dict) or set(sweep) - {"batch", "length"}:
            raise ValueError("Sweeps support batch and length only")
        if any(
            not isinstance(v, list) or not v or any(type(n) is not int or n < 1 for n in v)
            for v in sweep.values()
        ):
            raise ValueError("Sweep values must be nonempty lists of positive integers")
        for values in itertools.product(*sweep.values()):
            options = config | dict(zip(sweep, values))
            argv = [task]
            for key, value in options.items():
                action = actions[key]
                flag = action.option_strings[0]
                if isinstance(action, argparse._StoreTrueAction):
                    if type(value) is not bool:
                        raise ValueError(f"{key} must be boolean")
                    if value:
                        argv.append(flag)
                else:
                    if isinstance(value, (list, dict, bool)):
                        raise ValueError(f"{key} must be a scalar")
                    argv.extend([flag, str(value)])
            # Validate CLI spelling/types before launching any cases.
            try:
                cli.parse_args(argv)
            except SystemExit as exc:
                raise ValueError(f"Invalid configuration in {name}") from exc
            expanded.append({"name": name, "argv": argv})
            if len(expanded) > 256:
                raise ValueError("Limit a manifest to 256 expanded cases")
    return expanded


def execute(cases, output, node, condition):
    """Run cases sequentially, preserving logs and an incremental outcome index."""
    output.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="suite-", dir=output))
    outcomes = []
    for index, case in enumerate(cases, 1):
        folder = root / f"{index:03d}"
        folder.mkdir()
        command = [
            sys.executable,
            "-m",
            "runtime_bench.cli",
            *case["argv"],
            "--output",
            str(folder),
            "--node",
            node,
            "--condition",
            condition,
        ]
        print(f"[{index}/{len(cases)}] {case['name']}", flush=True)
        with (folder / "console.log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        reports = list(folder.glob("*.json"))
        outcomes.append(
            {
                **case,
                "exit_code": completed.returncode,
                "status": "completed" if completed.returncode == 0 else "failed",
                "reports": [str(p) for p in reports],
                "directory": str(folder),
            }
        )
        # Incremental index survives interruption or a later process crash.
        (root / "index.json").write_text(
            json.dumps(
                {
                    "manifest_version": 1,
                    "node": node,
                    "condition": condition,
                    "planned_cases": len(cases),
                    "outcomes": outcomes,
                },
                indent=2,
            )
            + "\n"
        )
    return root, outcomes


def main():
    """Parse suite arguments; exit 1 for case failures and 2 for invalid manifests."""
    p = argparse.ArgumentParser(description="Run a reproducible experiment manifest sequentially")
    p.add_argument("manifest", type=Path)
    p.add_argument("--node", required=True, help="User-supplied public-safe node label")
    p.add_argument("--condition", default="unspecified", help="E.g. idle or services-running")
    p.add_argument("--output", type=Path, default=Path("results"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    try:
        cases = expand(args.manifest)
        if args.dry_run:
            print(json.dumps(cases, indent=2))
            return
        root, outcomes = execute(cases, args.output, args.node, args.condition)
        print(root / "index.json")
        if any(row["exit_code"] for row in outcomes):
            p.exit(1, "Some cases failed; inspect the index and per-case logs.\n")
    except (OSError, ValueError, TypeError) as exc:
        p.exit(2, f"manifest failed: {exc}\n")


if __name__ == "__main__":
    main()
