"""Synthetic report fixtures exercise policy decisions, never claim GPU measurements."""

import json
from pathlib import Path

import pytest

from runtime_bench import frontiers, saturation


def fixture_index(tmp_path):
    plan = {
        "version": 1,
        "limits": {"timeout_seconds": 10, "min_host_available_bytes": 1},
        "gates": {"wall_seconds": 5},
        "profiles": [],
    }
    index = {
        "protocol": "saturation-v1",
        "plan": plan,
        "node": "test",
        "condition": "idle",
        "working_directory": str(Path.cwd().resolve()),
        "profiles": [],
    }
    for name, device in [("cpu", "cpu"), ("candidate", "cuda")]:
        profile = {"name": name, "gates": plan["gates"], "cases": []}
        outcome = {"name": name, "stages": []}
        for batch in (2, 8):
            case = {
                "name": f"batch-{batch}",
                "argv": [
                    "smoke",
                    "--device",
                    device,
                    "--batch",
                    str(batch),
                    "--threads",
                    "1",
                    "--repeats",
                    "3",
                ],
            }
            cfg = frontiers.config(case) | {"node": "test", "condition": "idle"}
            rate = 10 if name == "cpu" else 5 if batch == 2 else 20
            report = {
                "schema_version": 1,
                "status": "completed",
                "protocol": "micro-v2",
                "runtime": "torch",
                "device": device,
                "config": cfg,
                "dataset": {"synthetic": True},
                "parameters": 8,
                "git_commit": "synthetic-test-fixture",
                "quality": {},
                "wall_seconds": 1,
                "trials": [
                    {"loop_samples_per_second": rate, "compute_median_ms": 1} for _ in range(3)
                ],
            }
            path = tmp_path / f"{name}-{batch}.json"
            path.write_text(json.dumps(report))
            profile["cases"].append(case)
            outcome["stages"].append(
                {**case, "status": "passed", "reports": [str(path)], "decisions": []}
            )
        plan["profiles"].append(profile)
        index["profiles"].append(outcome)
    return index


def test_crossover_conservative_ranges_and_unknowns(tmp_path):
    index = fixture_index(tmp_path)
    result = frontiers.crossover(index, "cpu", "candidate", "batch")
    assert [r["classification"] for r in result["rows"]] == [
        "baseline_preferred",
        "candidate_beneficial",
    ]
    assert result["observed_transition_intervals"] == [[2, 8]]
    path = tmp_path / "candidate-8.json"
    report = json.loads(path.read_text())
    report["trials"][0]["loop_samples_per_second"] = 5
    path.write_text(json.dumps(report))
    result = frontiers.crossover(index, "cpu", "candidate", "batch")
    assert result["rows"][1]["classification"] == "ambiguous"
    assert result["observed_transition_intervals"] == []
    index["profiles"][1]["stages"][1]["status"] = "skipped"
    assert (
        frontiers.crossover(index, "cpu", "candidate", "batch")["rows"][1]["classification"]
        == "untested"
    )


def test_crossover_rejects_unequal_work_and_provenance(tmp_path):
    index = fixture_index(tmp_path)
    index["plan"]["profiles"][1]["cases"][1]["argv"] += ["--width", "512"]
    with pytest.raises(ValueError, match="only batch"):
        frontiers.crossover(index, "cpu", "candidate", "batch")
    index = fixture_index(tmp_path)
    path = tmp_path / "candidate-8.json"
    report = json.loads(path.read_text())
    report["dataset"]["different"] = True
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="fingerprint"):
        frontiers.crossover(index, "cpu", "candidate", "batch")


def threshold_index(tmp_path):
    index = fixture_index(tmp_path)
    path = tmp_path / "cpu-8.json"
    report = json.loads(path.read_text())
    report["wall_seconds"] = 6
    path.write_text(json.dumps(report))
    stage = index["profiles"][0]["stages"][1]
    stage.update(status="stopped", decisions=[{"reason": "threshold", "metric": "wall_seconds"}])
    return index


def test_boundary_refines_and_confirms_without_retesting_upper(tmp_path, monkeypatch):
    index = threshold_index(tmp_path)
    calls = []

    def fake_execute(plan, output, node, condition):
        value = frontiers.config(plan["profiles"][0]["cases"][0])["batch"]
        calls.append(value)
        decisions = [] if value < 6 else [{"reason": "threshold", "metric": "wall_seconds"}]
        return output, {
            "profiles": [
                {
                    "stages": [
                        {
                            "status": "stopped" if decisions else "passed",
                            "decisions": decisions,
                            "reports": [str(tmp_path / "cpu-2.json")],
                        }
                    ]
                }
            ]
        }

    monkeypatch.setattr(saturation, "execute", fake_execute)
    root, result = frontiers.refine(index, "cpu", "batch", tmp_path)
    assert result["interval"] == [5, 6]
    assert result["status"] == "resolution_reached"
    assert calls == [2, 2, 5, 5, 6, 6]
    assert json.loads((root / "index.json").read_text())["interval"] == [5, 6]


def test_boundary_does_not_refine_errors_or_inconsistent_observations(tmp_path, monkeypatch):
    index = threshold_index(tmp_path)
    calls = []

    def fake_execute(plan, output, node, condition):
        value = frontiers.config(plan["profiles"][0]["cases"][0])["batch"]
        calls.append(value)
        reasons = [{"reason": "threshold", "metric": "wall_seconds"}] if len(calls) == 4 else []
        return output, {
            "profiles": [
                {
                    "stages": [
                        {
                            "status": "stopped" if reasons else "passed",
                            "decisions": reasons,
                            "reports": [str(tmp_path / "cpu-2.json")],
                        }
                    ]
                }
            ]
        }

    monkeypatch.setattr(saturation, "execute", fake_execute)
    _, result = frontiers.refine(index, "cpu", "batch", tmp_path)
    assert result["status"] == "ambiguous"
    assert result["interval"] == [2, 8]
    path = tmp_path / "cpu-8.json"
    report = json.loads(path.read_text())
    report.update(status="failed", reason="out_of_memory")
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="threshold stops"):
        frontiers.refine(index, "cpu", "batch", tmp_path)


def test_boundary_blocks_changed_endpoint_provenance(tmp_path, monkeypatch):
    index = threshold_index(tmp_path)
    path = tmp_path / "fresh.json"
    report = json.loads((tmp_path / "cpu-2.json").read_text())
    report["git_commit"] = "different-synthetic-revision"
    path.write_text(json.dumps(report))

    def fake_execute(plan, output, node, condition):
        return output, {
            "profiles": [
                {"stages": [{"status": "passed", "decisions": [], "reports": [str(path)]}]}
            ]
        }

    monkeypatch.setattr(saturation, "execute", fake_execute)
    _, result = frontiers.refine(index, "cpu", "batch", tmp_path)
    assert result["status"] == "baseline_not_reproduced"
    assert len(result["observations"]) == 1
    assert result["observations"][0]["decisions"][0]["reason"] == "endpoint_provenance_changed"


def test_boundary_budget_and_failed_baseline(tmp_path, monkeypatch):
    index = threshold_index(tmp_path)

    def fake_execute(plan, output, node, condition):
        return output, {
            "profiles": [
                {
                    "stages": [
                        {
                            "status": "passed",
                            "decisions": [],
                            "reports": [str(tmp_path / "cpu-2.json")],
                        }
                    ]
                }
            ]
        }

    monkeypatch.setattr(saturation, "execute", fake_execute)
    _, result = frontiers.refine(index, "cpu", "batch", tmp_path, max_probes=1)
    assert result["status"] == "probe_budget_exhausted"
    assert result["interval"] == [5, 8]
