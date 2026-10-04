import csv

import pytest
import torch

from runtime_bench.cli import parser, run, save
from runtime_bench.hardware import select_device
from runtime_bench.workloads import MemoryTransformer, coffee, tokenize


def options(*extra):
    return parser().parse_args(
        [
            "smoke",
            "--device",
            "cpu",
            "--steps",
            "2",
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


def test_trials_reproduce_and_reports(tmp_path):
    result = run(options())
    assert result["trials"][0]["final_loss"] == result["trials"][1]["final_loss"]
    assert 0 <= result["quality"]["accuracy"] <= 1
    assert result["trials"][0]["compute_samples_per_second"] > 0
    assert save(result, tmp_path).exists()
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_state_has_effect_and_reset_restores_output():
    torch.manual_seed(1)
    model = MemoryTransformer(16, 4).eval()
    x = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        first = model(x)
        second = model(x)
        model.reset()
        restored = model(x)
    assert not torch.allclose(first, second)
    assert torch.equal(first, restored)


def test_coffee_rejects_leakage(tmp_path):
    path = tmp_path / "coffee.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["quality", "x"])
        writer.writerows([["good" if i % 2 else "bad", i] for i in range(40)])
    args = options("--data", str(path), "--target", "quality", "--features", "quality")
    with pytest.raises(ValueError, match="Target"):
        coffee(args)
    args.features = "x"
    _, train, test, meta = coffee(args)
    assert len(train[0]) == 32 and len(test[0]) == 8
    assert meta["data_sha256"]
    assert torch.allclose(train[0].mean(0), torch.zeros(1), atol=1e-6)


def test_bad_precision_and_configuration():
    with pytest.raises(ValueError, match="requires CUDA"):
        run(options("--precision", "fp16"))
    with pytest.raises(ValueError, match="positive"):
        run(options("--steps", "0"))
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="unavailable"):
            select_device("cuda")


def test_tokenization_stable():
    assert tokenize("Coffee GOOD", 4) == tokenize("coffee good", 4)
    assert tokenize("", 4) == [0, 0, 0, 0]


def test_transformer_workloads_and_comparison(tmp_path):
    from runtime_bench.compare import compare

    args = options("--length", "4")
    args.task = "stateful"
    report = run(args)
    assert compare(report, report)["compute_latency_speedup"] == 1
    other = {**report, "config": {**report["config"], "batch": 8}}
    with pytest.raises(ValueError, match="Incompatible"):
        compare(report, other)
    path = tmp_path / "news.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([[i % 4 + 1, f"Headline {i}", "sample news text"] for i in range(24)])
    args.task = "news"
    args.data = path
    args.mode = "infer"
    result = run(args)
    assert result["dataset"]["synthetic"] is False
    assert result["dataset"]["test_rows"] == 5
