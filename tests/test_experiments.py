import copy
import csv
import json

import pytest

from runtime_bench.cli import parser, run, save
from runtime_bench.compare import compare
from runtime_bench.experiments import execute, expand
from runtime_bench.reporting import annotate, export


def test_manifest_expansion_and_rejects_typos(tmp_path):
    path = tmp_path / "experiment.toml"
    path.write_text("""version = 1
[defaults]
steps = 2
[[cases]]
name = "test"
task = "smoke"
[cases.sweep]
batch = [2, 4]
length = [8, 16]
""")
    cases = expand(path)
    assert len(cases) == 4
    assert cases[0]["argv"] == ["smoke", "--steps", "2", "--batch", "2", "--length", "8"]
    path.write_text(path.read_text().replace("steps = 2", "stepz = 2"))
    with pytest.raises(ValueError, match="Unknown"):
        expand(path)


def test_suite_continues_failure_and_exports_trials(tmp_path):
    options = [
        "--device",
        "cpu",
        "--steps",
        "1",
        "--warmup",
        "0",
        "--repeats",
        "2",
        "--threads",
        "1",
        "--width",
        "8",
    ]
    cases = [
        {"name": "bad", "argv": ["smoke", *options, "--precision", "fp16"]},
        {"name": "good", "argv": ["smoke", *options]},
    ]
    root, outcomes = execute(cases, tmp_path, "node-a", "idle")
    assert [o["status"] for o in outcomes] == ["failed", "completed"]
    assert json.loads((root / "index.json").read_text())["planned_cases"] == 2
    paths = sorted(root.glob("*/*.json"))
    assert len(paths) == 2
    output = tmp_path / "trials.csv"
    assert export(paths, output) == 3
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["experiment_id"] == ""
    assert rows[1]["labels.node"] == "node-a"
    assert rows[1]["trial_index"] == "1" and rows[2]["trial_index"] == "2"
    assert rows[1]["trial.loop_samples_per_second"]


def test_identity_and_comparison_modes(tmp_path):
    args = parser().parse_args(
        [
            "smoke",
            "--device",
            "cpu",
            "--steps",
            "1",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--threads",
            "1",
        ]
    )
    baseline = run(args)
    candidate = copy.deepcopy(baseline)
    candidate["config"].update(node="node-b", condition="busy", device="cuda")
    assert annotate(candidate)["experiment_id"] == baseline["experiment_id"]
    assert compare(baseline, candidate)["loop_throughput_gain"] == 1
    candidate["config"]["profile"] = "diagnostic"
    assert annotate(candidate)["experiment_id"] != baseline["experiment_id"]
    with pytest.raises(ValueError, match="profile"):
        compare(baseline, candidate)
    save(baseline, tmp_path)
    # Legacy reports still compare with standard reports.
    candidate = copy.deepcopy(baseline)
    del candidate["config"]["profile"]
    assert compare(baseline, candidate)["loop_throughput_gain"] == 1
