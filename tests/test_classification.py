"""Classification lifecycle checks: real artifact reuse, split integrity, and CLI help."""

import copy

import pytest
import torch

from runtime_bench.classification import load_checkpoint
from runtime_bench.cli import classification_parser, main, parser, run
from runtime_bench.compare import compare


def options(mode, checkpoint, *extra):
    return parser().parse_args(
        [
            "classification",
            "--mode",
            mode,
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
            "--steps",
            "4",
            "--warmup",
            "1",
            "--repeats",
            "2",
            "--batch",
            "16",
            "--width",
            "12",
            "--threads",
            "1",
            *extra,
        ]
    )


def test_synthetic_checkpoint_roundtrip(tmp_path):
    checkpoint = tmp_path / "trained.pt"
    train = run(options("train", checkpoint))
    payload = load_checkpoint(checkpoint)
    assert train["dataset"]["initial_weights_sha256"] != payload["weights_sha256"]
    assert train["trials"][0]["final_loss"] == train["trials"][1]["final_loss"]
    assert train["checkpoint"]["saved_trial"] == 2
    before = checkpoint.read_bytes()
    infer = run(options("infer", checkpoint, "--width", "24", "--seed", "7"))
    assert infer["quality"] == train["quality"]
    assert infer["config"]["width"] == 12 and infer["config"]["seed"] == 42
    assert infer["dataset"]["timed_split"] == "test"
    assert checkpoint.read_bytes() == before
    assert infer["trials"][0]["final_loss"] == infer["trials"][1]["final_loss"]
    assert compare(infer, infer)["loop_throughput_gain"] == 1
    # Identical artifacts at different paths remain equivalent experiments.
    copied = tmp_path / "copied.pt"
    copied.write_bytes(before)
    other = run(options("infer", copied))
    assert other["experiment_id"] == infer["experiment_id"]
    assert compare(infer, other)
    with pytest.raises(ValueError, match="already exists"):
        run(options("train", checkpoint))


def test_csv_restores_preprocessing_and_rejects_changed_data(tmp_path, monkeypatch):
    path = tmp_path / "data.csv"
    path.write_text(
        "label,x,z\n" + "".join(f"{i % 2},{i if i % 7 else ''},{i % 3}\n" for i in range(80))
    )
    checkpoint = tmp_path / "csv.pt"
    train = run(
        options("train", checkpoint, "--data", str(path), "--target", "label", "--features", "x,z")
    )
    import runtime_bench.workloads as workloads

    def no_refit(*args, **kwargs):
        raise AssertionError("Inference must not refit preprocessing")

    monkeypatch.setattr(workloads.np, "nanmedian", no_refit)
    infer = run(options("infer", checkpoint, "--data", str(path)))
    assert infer["quality"] == train["quality"]
    assert infer["dataset"]["preprocessing"] == train["dataset"]["preprocessing"]
    path.write_text(path.read_text() + "0,10000,3\n")
    with pytest.raises(ValueError, match="fingerprint differs"):
        run(options("infer", checkpoint, "--data", str(path)))


def test_inference_requires_trained_artifact_and_matching_weights(tmp_path):
    with pytest.raises(ValueError, match="requires --checkpoint"):
        run(parser().parse_args(["classification", "--mode", "infer"]))
    checkpoint = tmp_path / "model.pt"
    run(options("train", checkpoint))
    payload = load_checkpoint(checkpoint)
    payload["state_dict"]["0.weight"].zero_()
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        run(options("infer", checkpoint))
    with pytest.raises(ValueError, match="fp32 only"):
        run(options("infer", checkpoint, "--runtime", "mlx", "--precision", "fp16"))


def test_focused_cli_help_and_defaults(monkeypatch, capsys):
    p = classification_parser()
    args = p.parse_args(["train"])
    assert args.task == "classification" and args.mode == "train"
    with pytest.raises(SystemExit) as exc:
        p.parse_args(["infer", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--checkpoint" in help_text and "--model-cache" not in help_text
    assert "--width" not in help_text and "--length" not in help_text
    monkeypatch.setattr("sys.argv", ["runtime-bench", "--help"])
    main()
    assert "classification" in capsys.readouterr().out


def test_checkpoint_changes_prevent_inference_comparison(tmp_path):
    first, second = tmp_path / "first.pt", tmp_path / "second.pt"
    run(options("train", first))
    run(options("train", second, "--steps", "8"))
    a, b = run(options("infer", first)), run(options("infer", second))
    with pytest.raises(ValueError, match="dataset fingerprint"):
        compare(a, b)
    changed = copy.deepcopy(a)
    changed["config"]["mode"] = "train"
    with pytest.raises(ValueError, match="mode"):
        compare(a, changed)


def test_checkpoint_inference_never_uses_training_step(tmp_path, monkeypatch):
    from runtime_bench.torch_runtime import TorchRuntime

    checkpoint = tmp_path / "model.pt"
    run(options("train", checkpoint))

    def forbidden(*args):
        raise AssertionError("Checkpoint inference must call predict, not the loss/training step")

    monkeypatch.setattr(TorchRuntime, "step", forbidden)
    assert run(options("infer", checkpoint))["status"] == "completed"


def test_imbalanced_quality_exposes_majority_only_predictions():
    from runtime_bench.classification_metrics import summarize

    metrics = summarize([[90, 0], [10, 0]], ["majority", "minority"])
    assert metrics["balanced_accuracy"] == 0.5  # Accuracy alone would be 90%.
    assert metrics["macro_f1"] == pytest.approx((180 / 190) / 2)
    assert metrics["per_class"][1]["precision"] is None
    assert metrics["per_class"][1]["recall"] == 0
    missing = summarize([[5, 0], [0, 0]], ["present", "absent"])
    assert missing["missing_evaluation_labels"] == ["absent"]
    assert missing["per_class"][1]["recall"] is None
    assert missing["balanced_accuracy"] == 1
    assert missing["macro_f1"] == 0.5


@pytest.mark.parametrize(
    "header,row,features,message",
    [
        ("label,x,x", "0,1,2", "x", "headers must be unique"),
        ("label,x", "0,1", "x,x", "nonempty and unique"),
        ("label,x", "0,1,2", "x", "same number of fields"),
        ("label,x", "0", "x", "same number of fields"),
    ],
)
def test_csv_schema_errors_are_actionable(tmp_path, header, row, features, message):
    path = tmp_path / "bad.csv"
    path.write_text(header + "\n" + row + "\n")
    with pytest.raises(ValueError, match=message):
        run(
            options(
                "train",
                tmp_path / "bad.pt",
                "--data",
                str(path),
                "--target",
                "label",
                "--features",
                features,
            )
        )


@pytest.mark.mlx
@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("csv_data", [False, True])
def test_native_mlx_checkpoint_interchange(tmp_path, device, csv_data):
    """Exercise both checkpoint directions, both MLP shapes and both MLX devices."""
    data = []
    training = []
    if csv_data:
        path = tmp_path / "data.csv"
        path.write_text("label,x,z\n" + "".join(f"{i % 3},{i},{i % 7}\n" for i in range(120)))
        data = ["--data", str(path)]
        training = [*data, "--target", "label", "--features", "x,z"]
    torch_path, mlx_path = tmp_path / "torch.pt", tmp_path / "mlx.pt"
    torch_train = run(options("train", torch_path, *training))
    mlx_infer = run(options("infer", torch_path, *data, "--runtime", "mlx", "--device", device))
    assert mlx_infer["quality"]["accuracy"] == torch_train["quality"]["accuracy"]
    assert mlx_infer["quality"]["cross_entropy"] == pytest.approx(
        torch_train["quality"]["cross_entropy"], abs=1e-6
    )
    mlx_train = run(options("train", mlx_path, *training, "--runtime", "mlx", "--device", device))
    assert mlx_train["trials"][0]["final_loss"] == mlx_train["trials"][1]["final_loss"]
    payload = load_checkpoint(mlx_path)
    assert payload["weights_sha256"] != mlx_train["dataset"]["initial_weights_sha256"]
    torch_infer = run(options("infer", mlx_path, *data))
    assert torch_infer["quality"]["confusion_matrix"] == mlx_train["quality"]["confusion_matrix"]
    assert torch_infer["quality"]["cross_entropy"] == pytest.approx(
        mlx_train["quality"]["cross_entropy"], abs=1e-6
    )
    # A canonical checkpoint can be compared across adapters without path-based identity.
    a = run(options("infer", torch_path, *data))
    assert a["experiment_id"] == mlx_infer["experiment_id"]
    assert compare(a, mlx_infer)["balanced_accuracy_change"] == 0


def test_legacy_checkpoint_and_majority_baseline(tmp_path):
    checkpoint = tmp_path / "legacy.pt"
    train = run(options("train", checkpoint))
    distribution = train["dataset"]["class_distribution"]
    assert sum(distribution["train"]) == 800
    assert sum(distribution["test"]) == train["quality"]["evaluation_rows"] == 224
    majority = train["dataset"]["train_majority_class"]
    assert train["quality"]["majority_baseline_accuracy"] == distribution["test"][majority] / 224
    assert train["quality"]["accuracy_above_majority_baseline"] == (
        train["quality"]["accuracy"] - train["quality"]["majority_baseline_accuracy"]
    )
    payload = load_checkpoint(checkpoint)
    for key in ("labels", "class_distribution", "train_majority_class"):
        payload["dataset"].pop(key, None)
    torch.save(payload, checkpoint)
    assert run(options("infer", checkpoint))["quality"] == train["quality"]


def test_corrupt_preprocessing_is_rejected(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("label,x\n" + "".join(f"{i % 2},{i}\n" for i in range(40)))
    checkpoint = tmp_path / "model.pt"
    run(options("train", checkpoint, "--data", str(path), "--target", "label", "--features", "x"))
    payload = load_checkpoint(checkpoint)
    payload["dataset"]["preprocessing"]["mean"] = [float("nan")]
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="non-finite"):
        run(options("infer", checkpoint, "--data", str(path)))


def test_prepare_regenerates_missing_checkpoint_and_reuses_existing(tmp_path, monkeypatch, capsys):
    checkpoint = tmp_path / "model.pt"
    command = [
        "runtime-bench",
        "classification",
        "prepare",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(tmp_path),
        "--device",
        "cpu",
        "--steps",
        "2",
        "--warmup",
        "0",
        "--repeats",
        "1",
        "--threads",
        "1",
    ]
    monkeypatch.setattr("sys.argv", command)
    main()
    original = checkpoint.read_bytes()
    reports = list(tmp_path.glob("*.json"))
    main()
    assert checkpoint.read_bytes() == original
    assert list(tmp_path.glob("*.json")) == reports
    assert "Reusing validated checkpoint" in capsys.readouterr().out
    checkpoint.unlink()
    with pytest.raises(ValueError, match="classification prepare"):
        run(options("infer", checkpoint))
    main()
    assert checkpoint.exists()
