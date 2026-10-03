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
    with pytest.raises(ValueError, match="torch only"):
        run(options("infer", checkpoint, "--runtime", "mlx"))


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
