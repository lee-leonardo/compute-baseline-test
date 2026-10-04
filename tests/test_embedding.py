"""Offline embedding objective, reset, checkpoint and retrieval contract tests."""

import csv
import json
import shutil

import pytest
import torch

from runtime_bench.cli import embedding_parser, parser, run
from runtime_bench.compare import compare
from runtime_bench.embedding import read_pairs, retrieval_quality


@pytest.fixture
def embedding_inputs(tmp_path):
    transformers = pytest.importorskip("transformers")
    path = tmp_path / "base"
    path.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "query", "answer"]
    vocab += [f"topic{i}" for i in range(24)]
    (path / "vocab.txt").write_text("\n".join(vocab) + "\n")
    transformers.BertTokenizerFast(vocab_file=str(path / "vocab.txt")).save_pretrained(path)
    torch.manual_seed(42)
    model = transformers.BertModel(
        transformers.BertConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=32,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
    )
    model.save_pretrained(path)
    data = tmp_path / "pairs.csv"
    with data.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["anchor", "positive"])
        writer.writerows((f"query topic{i}", f"answer topic{i}") for i in range(24))
    return path, data


def options(mode, path, data, *extra):
    return parser().parse_args(
        [
            "embedding",
            "--mode",
            mode,
            "--checkpoint",
            str(path),
            "--data",
            str(data),
            "--device",
            "cpu",
            "--steps",
            "3",
            "--warmup",
            "1",
            "--repeats",
            "2",
            "--batch",
            "4",
            "--length",
            "8",
            "--threads",
            "1",
            "--synthetic-data",
            *extra,
        ]
    )


def test_embedding_training_and_checkpoint_inference(embedding_inputs, tmp_path):
    model, data = embedding_inputs
    checkpoint = tmp_path / "trained"
    train = run(options("train", checkpoint, data, "--model", str(model)))
    saved = json.loads((checkpoint / "benchmark.json").read_text())
    assert saved["weights_sha256"] != train["dataset"]["initial_weights_sha256"]
    assert train["trials"][0]["final_loss"] == train["trials"][1]["final_loss"]
    infer = run(options("infer", checkpoint, data))
    assert infer["quality"] == train["quality"]
    assert infer["quality"]["mean_embedding_norm"] == pytest.approx(1)
    assert 0 <= infer["quality"]["retrieval_recall_at_1"] <= 1
    assert infer["dataset"]["timed_split"] == "test"
    assert compare(infer, infer)["retrieval_mrr_change"] == 0
    copied = tmp_path / "copy"
    shutil.copytree(checkpoint, copied)
    other = run(options("infer", copied, data, "--seed", "15"))
    assert other["experiment_id"] == infer["experiment_id"]
    assert compare(infer, other)
    with pytest.raises(ValueError, match="already exists"):
        run(options("train", checkpoint, data, "--model", str(model)))
    (copied / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="assets fingerprint"):
        run(options("infer", copied, data))
    data.write_text(data.read_text() + "new query,new answer\n")
    with pytest.raises(ValueError, match="data fingerprint"):
        run(options("infer", checkpoint, data))


def test_embedding_diagnostic_preserves_learning(embedding_inputs, tmp_path):
    model, data = embedding_inputs
    a = run(options("train", tmp_path / "a", data, "--model", str(model)))
    b = run(
        options("train", tmp_path / "b", data, "--model", str(model), "--profile", "diagnostic")
    )
    assert a["quality"] == b["quality"]
    assert a["checkpoint"]["weights_sha256"] == b["checkpoint"]["weights_sha256"]
    assert all(0 < sum(t["phases"].values()) <= t["wall_seconds"] for t in b["trials"])
    with pytest.raises(ValueError, match="profile"):
        compare(a, b)


def test_embedding_data_and_batch_contract(embedding_inputs, tmp_path):
    model, data = embedding_inputs
    args = options("train", tmp_path / "out", data, "--model", str(model))
    pairs, train, test = read_pairs(args)
    assert not {text for i in train for text in pairs[i]} & {
        text for i in test for text in pairs[i]
    }
    for batch in (1, 20):
        with pytest.raises(ValueError, match="Contrastive training requires"):
            run(
                options(
                    "train", tmp_path / "out", data, "--model", str(model), "--batch", str(batch)
                )
            )
    data.write_text(data.read_text() + "query topic0,other positive\n")
    with pytest.raises(ValueError, match="unique"):
        read_pairs(args)
    with pytest.raises(ValueError, match="require --runtime torch"):
        run(options("train", tmp_path / "out", data, "--runtime", "mlx"))


def test_retrieval_ranking_and_ties():
    identity = torch.eye(3)
    assert retrieval_quality(identity, identity)["retrieval_recall_at_1"] == 1
    tied = torch.ones(3, 2)
    result = retrieval_quality(tied, tied)
    assert result["retrieval_recall_at_1"] == 0
    assert result["retrieval_mrr"] == pytest.approx(1 / 3)


def test_focused_embedding_help(capsys):
    with pytest.raises(SystemExit) as exc:
        embedding_parser().parse_args(["infer", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--checkpoint" in help_text and "--features" not in help_text


def test_news_pair_preparation_is_explicit_and_preserves_files(tmp_path):
    from runtime_bench.embedding import create_news_pairs

    source, output = tmp_path / "news.csv", tmp_path / "pairs.csv"
    with source.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows((1, f"title {i}", f"description {i}") for i in range(20))
        writer.writerow((1, "title 0", "repeated"))
    create_news_pairs(source, output, 100)
    provenance = json.loads(output.with_suffix(".source.json").read_text())
    assert provenance["pairs"] == 20
    assert provenance["source_sha256"]
    args = options("train", tmp_path / "model", output)
    assert len(read_pairs(args)[0]) == 20
    before = output.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        create_news_pairs(source, output, 100)
    assert output.read_bytes() == before


def test_embedding_missing_checkpoint_is_actionable(embedding_inputs, tmp_path):
    _, data = embedding_inputs
    with pytest.raises(ValueError, match="embedding train first"):
        run(options("infer", tmp_path / "missing", data))
