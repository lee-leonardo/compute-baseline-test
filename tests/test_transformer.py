"""Transformer sequence semantics, padding, reset and portable artifact tests."""

import csv
import shutil

import pytest
import torch

from runtime_bench.cli import parser, run, transformer_parser
from runtime_bench.compare import compare
from runtime_bench.transformer import SequenceTransformer, load_checkpoint, prepare


def options(mode, checkpoint, *extra):
    return parser().parse_args(
        [
            "transformer",
            "--mode",
            mode,
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
            "--width",
            "16",
            "--length",
            "8",
            "--limit",
            "80",
            "--batch",
            "4",
            "--steps",
            "3",
            "--warmup",
            "1",
            "--repeats",
            "2",
            "--threads",
            "1",
            *extra,
        ]
    )


def test_padding_mask_and_order_sensitivity():
    torch.manual_seed(42)
    model = SequenceTransformer(16, 8, 33, 2).eval()
    with torch.inference_mode():
        short = model(torch.tensor([[1, 2, 3, 4]]))
        padded = model(torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0]]))
        reverse = model(torch.tensor([[4, 3, 2, 1]]))
    torch.testing.assert_close(short, padded, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(short, reverse)


def test_synthetic_semantics_and_disjoint_splits(tmp_path):
    args = options("train", tmp_path / "model.pt")
    _, train, test, meta = prepare(args)
    assert not {tuple(x.tolist()) for x in train[0]} & {tuple(x.tolist()) for x in test[0]}
    for x, y in (train, test):
        assert torch.equal((x[:, 0] > x[:, -1]).long(), y)
    assert meta["synthetic"] is True
    assert meta["train_rows"] == 64 and meta["test_rows"] == 16


def test_transformer_checkpoint_roundtrip_and_reset(tmp_path):
    path = tmp_path / "model.pt"
    train = run(options("train", path))
    assert train["protocol"] == "transformer-v1"
    assert (
        train["trials"][0]["padded_tokens_per_second"]
        == train["trials"][0]["loop_samples_per_second"] * 8
    )
    assert train["trials"][0]["final_loss"] == train["trials"][1]["final_loss"]
    assert train["checkpoint"]["weights_sha256"] != train["dataset"]["initial_weights_sha256"]
    original = path.read_bytes()
    infer = run(options("infer", path, "--width", "32", "--length", "16", "--seed", "12"))
    assert infer["quality"] == train["quality"]
    assert infer["config"]["width"] == 16 and infer["config"]["length"] == 8
    assert path.read_bytes() == original
    copy = tmp_path / "copy.pt"
    shutil.copyfile(path, copy)
    other = run(options("infer", copy))
    assert other["experiment_id"] == infer["experiment_id"]
    assert compare(infer, other)["accuracy_change"] == 0
    with pytest.raises(ValueError, match="already exists"):
        run(options("train", path))
    saved = load_checkpoint(copy)
    saved["state_dict"]["head.weight"].zero_()
    torch.save(saved, copy)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        run(options("infer", copy))


def test_news_checkpoint_and_data_validation(tmp_path):
    path, checkpoint = tmp_path / "news.csv", tmp_path / "news.pt"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows((i % 4 + 1, f"news {i}", "example text") for i in range(80))
    train = run(options("train", checkpoint, "--data", str(path), "--synthetic-data"))
    infer = run(options("infer", checkpoint, "--data", str(path)))
    assert train["quality"] == infer["quality"]
    assert train["dataset"]["synthetic"] is True
    path.write_text(path.read_text() + "1,changed,example\n")
    with pytest.raises(ValueError, match="original CSV fingerprint"):
        run(options("infer", checkpoint, "--data", str(path)))
    path.write_text("1,duplicate,words\n" * 20)
    with pytest.raises(ValueError, match="Repeated tokenized"):
        run(options("train", tmp_path / "other.pt", "--data", str(path)))


def test_invalid_requests_and_help(tmp_path, capsys):
    with pytest.raises(ValueError, match="checkpoint missing"):
        run(options("infer", tmp_path / "missing.pt"))
    with pytest.raises(ValueError, match="requires --runtime torch"):
        run(options("train", tmp_path / "model.pt", "--runtime", "mlx"))
    with pytest.raises(ValueError, match="divisible by four"):
        run(options("train", tmp_path / "model.pt", "--width", "7"))
    with pytest.raises(SystemExit) as exc:
        transformer_parser().parse_args(["infer", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--checkpoint" in help_text and "--width" not in help_text
