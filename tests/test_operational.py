"""Exercise the ladder offline with tiny local Transformers fixtures."""

import json

import pytest
import torch

from runtime_bench.cli import main, parser, run, save
from runtime_bench.compare import capacity_line, compare


@pytest.fixture
def tiny_model(tmp_path):
    transformers = pytest.importorskip("transformers")
    model_path = tmp_path / "model"
    model_path.mkdir()
    vocab = model_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nnews\ncoffee\ngood\nbad\n")
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(vocab))
    tokenizer.save_pretrained(model_path)
    torch.manual_seed(42)
    config = transformers.BertConfig(
        vocab_size=9,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
    )
    transformers.BertForMaskedLM(config).save_pretrained(model_path)
    csv = tmp_path / "news.csv"
    csv.write_text("".join(f"{i % 4 + 1},news coffee,good bad\n" for i in range(24)))
    return model_path, csv


def options(task, *extra):
    return parser().parse_args(
        [
            task,
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
            "--length",
            "8",
            "--threads",
            "2",
            *extra,
        ]
    )


def test_sklearn_cpu_ladder(tmp_path):
    pytest.importorskip("sklearn")
    args = options("classify", "--limit", "100")
    result = run(args)
    assert result["runtime"] == "sklearn" and result["dataset"]["synthetic"] is True
    assert result["trials"][0]["estimator_iterations"] == 2
    assert 0 <= result["quality"]["accuracy"] <= 1
    assert result["resources"]["process_rss_sampled_peak_bytes"] > 0
    assert result["resources"]["gpu_utilization_mean_percent"] is None
    assert compare(result, result)["job_wall_time_speedup"] == 1
    assert "status=completed" in save(result, tmp_path).read_text()


@pytest.mark.parametrize("task", ["embeddings", "infer", "finetune"])
def test_offline_pretrained_ladder(task, tiny_model, tmp_path):
    model, csv = tiny_model
    result = run(options(task, "--model", str(model), "--data", str(csv), "--synthetic-data"))
    assert result["protocol"] == "operational-v1"
    assert result["dataset"]["model_assets_sha256"]
    assert result["dataset"]["initial_weights_sha256"]
    assert result["trials"][0]["samples_per_second"] > 0
    if task == "finetune":
        assert result["trials"][0]["final_loss"] == result["trials"][1]["final_loss"]
        assert 0 <= result["quality"]["accuracy"] <= 1
    else:
        assert result["quality"]["accuracy"] is None
    assert compare(result, result)["loop_throughput_gain"] == 1
    save(result, tmp_path / "results")


def test_failure_artifact_and_capacity_view(monkeypatch, tmp_path):
    from runtime_bench import operational

    def oom(_):
        raise torch.OutOfMemoryError("out of memory (injected test)")

    monkeypatch.setattr(operational, "run", oom)
    monkeypatch.setattr(
        "sys.argv", ["runtime-bench", "infer", "--device", "cpu", "--output", str(tmp_path)]
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    record = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert record["status"] == "failed" and record["reason"] == "out_of_memory"
    assert record["resources"]["process_rss_sampled_peak_bytes"] > 0
    assert "reason=out_of_memory" in capacity_line(record)
    with pytest.raises(ValueError, match="completed trials"):
        compare(record, record)


def test_pipeline_requests_are_explicit():
    with pytest.raises(ValueError, match="fixed mode"):
        run(options("infer", "--mode", "train"))
    with pytest.raises(ValueError, match="sklearn/PyTorch"):
        run(options("embeddings", "--runtime", "mlx"))


def test_news_acquisition_is_separate_and_preserves_inputs(monkeypatch, tmp_path):
    from io import BytesIO
    from runtime_bench.model_specs import fetch_news

    monkeypatch.setattr(
        "runtime_bench.model_specs.urlopen", lambda *a, **k: BytesIO(b"1,news,coffee\n")
    )
    path = fetch_news(tmp_path / "news.csv")
    assert path.read_text() == "1,news,coffee\n"
    provenance = json.loads(path.with_suffix(".source.json").read_text())
    assert provenance["sha256"]
    path.write_text("existing user data")
    assert fetch_news(path).read_text() == "existing user data"


def test_gpu_sampler_uses_actual_uuid_and_reports_unavailable(monkeypatch):
    import sys
    from types import SimpleNamespace
    from runtime_bench.resources import ResourceSampler

    calls = []
    fake = SimpleNamespace(
        nvmlInit=lambda: calls.append("init"),
        nvmlShutdown=lambda: calls.append("shutdown"),
        nvmlDeviceGetHandleByUUID=lambda uuid: calls.append(uuid) or "handle",
        nvmlDeviceGetUtilizationRates=lambda handle: SimpleNamespace(gpu=73),
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=123456),
        NVMLError=RuntimeError,
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid="GPU-selected")
    )
    with ResourceSampler(options("infer", "--device", "cuda")) as sampler:
        pass
    metrics = sampler.summary()
    assert "GPU-selected" in calls and calls[-1] == "shutdown"
    assert metrics["gpu_utilization_mean_percent"] == 73
    assert metrics["gpu_device_used_sampled_peak_bytes"] == 123456
    assert metrics["gpu_utilization_unavailable_reason"] is None

    def broken(_):
        raise RuntimeError("driver unavailable")

    fake.nvmlDeviceGetHandleByUUID = broken
    with ResourceSampler(options("infer", "--device", "cuda")) as unavailable:
        pass
    assert unavailable.summary()["gpu_utilization_mean_percent"] is None
    assert "unavailable" in unavailable.summary()["gpu_utilization_unavailable_reason"]


def test_embedding_alias_cannot_silently_initialize_inference_head():
    with pytest.raises(ValueError, match="without a pretrained fill-mask head"):
        run(options("infer", "--model", "minilm"))


def test_local_encoder_cannot_silently_initialize_inference_head(tiny_model, tmp_path):
    from transformers import BertModel

    model, csv = tiny_model
    encoder_path = tmp_path / "encoder"
    encoder_path.mkdir()
    BertModel.from_pretrained(model).save_pretrained(encoder_path)
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(model).save_pretrained(encoder_path)
    with pytest.raises(ValueError, match="complete pretrained fill-mask checkpoint"):
        run(options("infer", "--model", str(encoder_path), "--data", str(csv)))


@pytest.mark.parametrize("task", ["embeddings", "infer", "finetune"])
def test_diagnostic_phases_preserve_work_and_quality(task, tiny_model):
    model, csv = tiny_model
    args = options(task, "--model", str(model), "--data", str(csv))
    standard = run(args)
    args.profile = "diagnostic"
    diagnostic = run(args)
    assert standard["dataset"] == diagnostic["dataset"]
    assert standard["quality"] == diagnostic["quality"]
    assert standard["trials"][0]["phases"] is None
    for trial in diagnostic["trials"]:
        assert all(v >= 0 for v in trial["phases"].values())
        assert 0 < sum(trial["phases"].values()) <= trial["wall_seconds"]
    with pytest.raises(ValueError, match="profile"):
        compare(standard, diagnostic)


def test_representative_pipeline(tiny_model, tmp_path):
    model, csv = tiny_model
    result = run(options("pipeline", "--model", str(model), "--data", str(csv), "--synthetic-data"))
    assert result["protocol"] == "pipeline-v1"
    assert result["dataset"]["timed_split"] == "test"
    assert result["dataset"]["classifier_training_rows"] == result["dataset"]["train_rows"]
    assert result["dataset"]["synthetic"] is True
    assert result["dataset"]["initial_weights_sha256"]
    assert 0 <= result["quality"]["accuracy"] <= 1
    assert len(result["trials"]) == 2
    for trial in result["trials"]:
        assert trial["samples_per_second"] > 0
        assert set(trial["phases"]) == {
            "tokenize_seconds",
            "transfer_seconds",
            "embed_seconds",
            "return_to_cpu_seconds",
            "classify_seconds",
        }
        assert sum(trial["phases"].values()) <= trial["wall_seconds"]
    assert compare(result, result)["loop_throughput_gain"] == 1
    save(result, tmp_path / "reports")
