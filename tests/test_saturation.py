import copy
import json
import sys
from types import SimpleNamespace

import pytest

from runtime_bench import saturation


MANIFEST = """version = 1
[defaults]
task = "smoke"
steps = 1
warmup = 0
repeats = 1
width = 8
[limits]
timeout_seconds = 30
min_host_available_bytes = 1
[gates]
wall_seconds = 20
[[profiles]]
name = "cpu"
runtime = "torch"
device = "cpu"
threads = 1
[[stages]]
name = "baseline"
[stages.options]
batch = 2
[[stages]]
name = "larger"
[stages.options]
batch = 4
"""


def plan_at(tmp_path, text=MANIFEST):
    path = tmp_path / "plan.toml"
    path.write_text(text)
    return saturation.load_plan(path)


def test_plan_preserves_order_and_pins_device(tmp_path):
    plan = plan_at(tmp_path)
    assert [c["name"] for c in plan["profiles"][0]["cases"]] == ["baseline", "larger"]
    assert "--device" in plan["profiles"][0]["cases"][0]["argv"]
    assert plan_at(tmp_path) == plan
    assert plan_at(tmp_path, MANIFEST.replace("batch = 4", "batch = 8"))["id"] != plan["id"]


@pytest.mark.parametrize(
    "before,after",
    [
        ('device = "cpu"', 'device = "auto"'),
        ('device = "cpu"', 'device = "gpu"'),
        ("wall_seconds = 20", "wall_seconds = nan"),
        ("wall_seconds = 20", "vram = 20"),
        ("timeout_seconds = 30", "timeout_seconds = 0"),
        ("min_host_available_bytes = 1", "unknown = 1"),
        ("batch = 4", 'device = "cuda"'),
        ("batch = 4", "batch = -1"),
        ("batch = 4", "batc = 4"),
        ('name = "larger"', 'name = "baseline"'),
        ("batch = 4", 'checkpoint = "shared.pt"'),
    ],
)
def test_invalid_plan_fails_before_compute(tmp_path, before, after):
    with pytest.raises(ValueError):
        plan_at(tmp_path, MANIFEST.replace(before, after))


def test_gate_worst_trial_inclusive_and_missing_data():
    report = {
        "status": "completed",
        "trials": [{"compute_p95_ms": 1}, {"compute_p95_ms": 10}],
        "resources": {"host_available_ram_min_bytes": 100},
    }
    reasons = saturation.evaluate(report, {"compute_p95_ms": 10, "host_available_bytes": 100})
    assert len(reasons) == 2
    assert reasons[0]["observed"] == 10
    assert (
        saturation.evaluate(report, {"cuda_allocated_bytes": 100})[0]["reason"] == "missing_metric"
    )
    report["trials"][1]["compute_p95_ms"] = float("nan")
    assert saturation.evaluate(report, {"compute_p95_ms": 10})[0]["reason"] == "missing_metric"
    assert saturation.evaluate({"status": "failed", "reason": "out_of_memory"}, {}) == [
        {"reason": "out_of_memory"}
    ]


def test_stop_skips_only_affected_profile_and_prior_is_sticky(tmp_path, monkeypatch):
    plan = plan_at(tmp_path)
    second = copy.deepcopy(plan["profiles"][0])
    second["name"] = "second"
    plan["profiles"].append(second)
    calls = []

    def fake(command, folder, limits):
        calls.append(command)
        (folder / "report.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "wall_seconds": 30 if len(calls) == 1 else 1,
                }
            )
        )
        return 0, []

    monkeypatch.setattr(saturation, "supervise", fake)
    root, index = saturation.execute(plan, tmp_path, "node", "idle")
    assert len(calls) == 3
    first, second = index["profiles"]
    assert [s["status"] for s in first["stages"]] == ["stopped", "skipped"]
    assert first["last_passed"] is None
    assert second["status"] == "exhausted"
    assert second["last_passed"] == "larger"
    saturation.execute(plan, tmp_path, "node", "idle", root / "index.json")
    assert len(calls) == 5  # stopped profile skipped, successful profile re-baselined
    with pytest.raises(ValueError, match="Prior index"):
        saturation.execute(plan, tmp_path, "other", "idle", root / "index.json")
    changed = copy.deepcopy(plan)
    changed["limits"]["timeout_seconds"] = 31
    with pytest.raises(ValueError, match="Prior index"):
        saturation.execute(changed, tmp_path, "node", "idle", root / "index.json")


@pytest.mark.parametrize(
    "content,code,reason",
    [
        (None, -9, "missing_report"),
        ("invalid", 0, "invalid_report"),
        ('{"status":"failed","reason":"out_of_memory"}', 2, "out_of_memory"),
        ('{"status":"completed","wall_seconds":1}', 2, "execution_error"),
    ],
)
def test_child_errors_stop_escalation(tmp_path, monkeypatch, content, code, reason):
    plan = plan_at(tmp_path)

    def fake(command, folder, limits):
        if content:
            (folder / "report.json").write_text(content)
        return code, []

    monkeypatch.setattr(saturation, "supervise", fake)
    _, index = saturation.execute(plan, tmp_path, "node", "idle")
    assert index["profiles"][0]["stop"][0]["reason"] == reason
    assert index["profiles"][0]["stages"][1]["status"] == "skipped"


def test_interrupt_preserves_checkpoint(tmp_path, monkeypatch):
    plan = plan_at(tmp_path)

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(saturation, "supervise", interrupt)
    with pytest.raises(KeyboardInterrupt):
        saturation.execute(plan, tmp_path, "node", "idle")
    path = next(tmp_path.glob("saturation-*/index.json"))
    index = json.loads(path.read_text())
    assert index["status"] == "interrupted"
    assert "cpu" in saturation.prior_stops(path, plan, "node", "idle")


def test_live_timeout_and_preflight_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(saturation.psutil, "virtual_memory", lambda: SimpleNamespace(available=100))
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    code, reasons = saturation.supervise(
        command,
        tmp_path,
        {
            "timeout_seconds": 0.1,
            "min_host_available_bytes": 1,
        },
    )
    assert code != 0 and reasons[0]["reason"] == "timeout"
    code, reasons = saturation.supervise(
        command,
        tmp_path,
        {
            "timeout_seconds": 30,
            "min_host_available_bytes": 100,
        },
    )
    assert code is None and reasons[0]["reason"] == "host_memory_guard"


def test_real_cpu_campaign_and_cli_status(tmp_path, monkeypatch, capsys):
    plan = plan_at(tmp_path)
    root, index = saturation.execute(plan, tmp_path, "test-node", "idle")
    assert index["profiles"][0]["status"] == "exhausted"
    reports = list(root.glob("*/*.json"))
    assert len(reports) == 2
    for report_path in reports:
        report = json.loads(report_path.read_text())
        assert report["status"] == "completed"
        assert report["config"]["device"] == "cpu"
        assert report["dataset"]["initial_weights_sha256"]
    monkeypatch.setattr(sys, "argv", ["saturate", "status", str(root / "index.json")])
    saturation.main()
    assert '"last_passed": "larger"' in capsys.readouterr().out


def test_profile_specific_gates(tmp_path):
    text = MANIFEST.replace(
        "threads = 1", "threads = 1\n[profiles.gates]\nprocess_rss_bytes = 1000"
    )
    plan = plan_at(tmp_path, text)
    assert plan["profiles"][0]["gates"] == {"wall_seconds": 20, "process_rss_bytes": 1000}
    assert plan["gates"] == {"wall_seconds": 20}


def test_baseline_only_disqualifies_and_requires_matching_config(tmp_path, monkeypatch):
    plan = plan_at(tmp_path)
    from runtime_bench.cli import parser, run

    args = parser().parse_args(
        [
            *plan["profiles"][0]["cases"][0]["argv"],
            "--node",
            "node",
            "--condition",
            "idle",
        ]
    )
    report = run(args)
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(report))
    spec = [f"cpu={path}"]
    assert saturation.baseline_evidence(spec, plan, "node", "idle")["cpu"]["decisions"] == []
    calls = []

    def measured_again(command, folder, limits):
        calls.append(command)
        (folder / "report.json").write_text(json.dumps(report))
        return 0, []

    monkeypatch.setattr(saturation, "supervise", measured_again)
    _, fresh = saturation.execute(plan, tmp_path, "node", "idle", baselines=spec)
    assert len(calls) == 2 and fresh["profiles"][0]["status"] == "exhausted"
    with pytest.raises(ValueError, match="node"):
        saturation.baseline_evidence(spec, plan, "other", "idle")
    changed = copy.deepcopy(plan)
    changed["profiles"][0]["cases"][0]["argv"] += ["--batch", "100"]
    with pytest.raises(ValueError, match="batch"):
        saturation.baseline_evidence(spec, changed, "node", "idle")
    # Lower a policy threshold below the real measurement; do not invent hardware data.
    plan["profiles"][0]["gates"]["wall_seconds"] = report["wall_seconds"] / 2

    def forbidden(*args):
        pytest.fail("A disqualifying baseline must prevent child execution")

    monkeypatch.setattr(saturation, "supervise", forbidden)
    _, index = saturation.execute(plan, tmp_path, "node", "idle", baselines=spec)
    assert index["profiles"][0]["stop"][0]["reason"] == "prior_baseline"
    assert all(s["status"] == "skipped" for s in index["profiles"][0]["stages"])


def test_live_process_rss_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(saturation.psutil, "virtual_memory", lambda: SimpleNamespace(available=100))
    code, reasons = saturation.supervise(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path,
        {"timeout_seconds": 10, "min_host_available_bytes": 1, "max_process_rss_bytes": 1},
    )
    assert code != 0 and reasons[0]["reason"] == "process_memory_guard"


def test_default_sweep_rejected(tmp_path):
    text = MANIFEST.replace('task = "smoke"', 'sweep = { batch = [2, 4] }\ntask = "smoke"')
    with pytest.raises(ValueError, match="sweeps"):
        plan_at(tmp_path, text)


def test_public_cli_plan_does_not_compute(tmp_path, monkeypatch, capsys):
    from runtime_bench.cli import main

    plan_at(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["runtime-bench", "saturate", "plan", str(tmp_path / "plan.toml")]
    )
    main()
    assert json.loads(capsys.readouterr().out)["profiles"][0]["name"] == "cpu"
