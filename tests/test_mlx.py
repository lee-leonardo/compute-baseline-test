"""Native parity checks are opt-in; platform validation is tested everywhere."""

import copy

import numpy as np
import pytest
import torch

from runtime_bench.cli import parser, run, save
from runtime_bench.mlx_runtime import MLXRuntime, validate
from runtime_bench.torch_runtime import TorchRuntime
from runtime_bench.workloads import build


def options(task="smoke", *extra):
    return parser().parse_args(
        [
            task,
            "--runtime",
            "mlx",
            "--device",
            "gpu",
            "--steps",
            "3",
            "--warmup",
            "1",
            "--repeats",
            "2",
            "--batch",
            "4",
            "--width",
            "16",
            *extra,
        ]
    )


def test_mlx_invalid_requests_and_host(monkeypatch):
    with pytest.raises(ValueError, match="supports coffee and smoke"):
        run(options("news"))
    with pytest.raises(ValueError, match="fp32 only"):
        run(options("smoke", "--precision", "fp16"))
    with pytest.raises(ValueError, match="MLX requires"):
        run(options("smoke", "--device", "cuda"))
    monkeypatch.setattr("runtime_bench.mlx_runtime.platform.system", lambda: "Linux")
    with pytest.raises(ValueError, match="native Apple Silicon"):
        validate(options())
    with pytest.raises(ValueError, match="PyTorch requires"):
        run(parser().parse_args(["smoke", "--device", "gpu"]))


@pytest.mark.mlx
@pytest.mark.parametrize("task", ["smoke", "coffee"])
def test_native_mlx_matches_torch_forward_and_adamw(task, tmp_path):
    args = options(task)
    if task == "coffee":
        path = tmp_path / "coffee.csv"
        path.write_text("label,x,z\n" + "".join(f"{i % 2},{i},{i % 3}\n" for i in range(40)))
        args.data, args.target, args.features = path, "label", "x,z"
    torch.manual_seed(args.seed)
    model, train, _, _ = build(args)
    torch_args = copy.copy(args)
    torch_args.runtime, torch_args.device = "torch", "cpu"
    torch_runtime = TorchRuntime(torch_args, copy.deepcopy(model))
    mlx_runtime = MLXRuntime(args, model)
    torch_runtime.reset()
    x, y = (v[: args.batch] for v in train)
    torch_inputs = torch_runtime.prepare(x, y)
    mlx_inputs = mlx_runtime.prepare(x, y)
    for _ in range(3):
        with torch.no_grad():
            torch_output = torch_runtime.model(torch_inputs[0]).numpy()
        mlx_output = np.array(mlx_runtime.model(mlx_inputs[0]))
        np.testing.assert_allclose(mlx_output, torch_output, atol=3e-5, rtol=3e-4)
        torch_loss = torch_runtime.losses([torch_runtime.step(*torch_inputs)])[0]
        mlx_loss = mlx_runtime.losses([mlx_runtime.step(*mlx_inputs)])[0]
        assert mlx_loss == pytest.approx(torch_loss, abs=3e-5)
    # Post-update outputs exercise gradients and optimizer state, not only copied weights.
    with torch.no_grad():
        torch_output = torch_runtime.model(torch_inputs[0]).numpy()
    np.testing.assert_allclose(
        np.array(mlx_runtime.model(mlx_inputs[0])), torch_output, atol=3e-5, rtol=3e-4
    )
    mlx_runtime.reset()
    torch_runtime.reset()
    with torch.no_grad():
        restored = torch_runtime.model(torch_inputs[0]).numpy()
    np.testing.assert_allclose(
        np.array(mlx_runtime.model(mlx_inputs[0])), restored, atol=3e-5, rtol=3e-4
    )


@pytest.mark.mlx
@pytest.mark.parametrize("mode", ["train", "infer"])
def test_native_mlx_trials_reports_and_cross_runtime_comparison(mode, tmp_path):
    from runtime_bench.compare import compare

    args = options("smoke", "--mode", mode)
    report = run(args)
    assert report["runtime"] == "mlx" and report["device"] == "mlx-gpu"
    assert report["hardware"]["unified_memory"] is True
    assert report["trials"][0]["final_loss"] == report["trials"][1]["final_loss"]
    assert report["trials"][0]["mlx_peak_allocated_bytes"] > 0
    text = save(report, tmp_path).read_text()
    assert "runtime=mlx" in text and "mlx=0.32.3" in text
    args.runtime, args.device = "torch", "cpu"
    baseline = run(args)
    result = compare(baseline, report)
    assert result["compute_latency_speedup"] > 0
    assert result["cross_entropy_change"] == pytest.approx(0, abs=3e-5)
