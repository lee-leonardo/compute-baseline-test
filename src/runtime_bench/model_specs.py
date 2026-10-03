"""Public model revisions: acquisition is separate from measured execution."""

import argparse
import csv
import hashlib
import json
import shutil
from urllib.request import urlopen
from pathlib import Path


MODELS = {
    "minilm": (
        "sentence-transformers/all-MiniLM-L6-v2",
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    ),
    "distilbert": (
        "distilbert/distilbert-base-uncased",
        "12040accade4e8a0f71eabdb258fecc2e7e948be",
    ),
    "bert": ("google-bert/bert-base-uncased", "86b5e0934494bd15c9632b12f734a8a67f723594"),
}


def select_model(task, model=None):
    """Choose assets independently of the execution backend and task head."""
    selected = model or ("minilm" if task == "embeddings" else "distilbert")
    if task == "infer" and selected == "minilm":
        raise ValueError(
            "MiniLM is an embedding checkpoint without a pretrained fill-mask head; "
            "use --model distilbert or --model bert for infer"
        )
    return selected


def resolve(model, cache, download=False):
    from huggingface_hub import snapshot_download

    if model and Path(model).is_dir():
        return Path(model), {"model_id": "local", "model_revision": "local-file-fingerprint"}
    if model not in MODELS:
        raise ValueError(
            "--model must be minilm, distilbert, bert or an existing local model directory"
        )
    repo, revision = MODELS[model]
    try:
        path = snapshot_download(
            repo,
            revision=revision,
            cache_dir=cache,
            local_files_only=not download,
            allow_patterns=["*.json", "*.txt", "*.safetensors"],
            ignore_patterns=["onnx/*", "openvino/*"],
        )
    except OSError as exc:
        raise ValueError(
            f"Model is not cached; run runtime-bench-fetch {model} --cache {cache}"
        ) from exc
    return Path(path), {"model_id": repo, "model_revision": revision}


NEWS_URL = (
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
)


def fetch_news(output):
    """Download an AG News CSV mirror once; retain the exact file hash and source."""
    if output.exists():
        return output  # Never overwrite a supplied dataset.
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.part")
    try:
        with urlopen(NEWS_URL, timeout=30) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        with temporary.open(newline="", encoding="utf-8") as handle:
            rows = csv.reader(handle)
            first = next(rows, [])
            if len(first) < 3 or first[0] not in ("1", "2", "3", "4"):
                raise ValueError("Downloaded news file is not an AG News CSV")
        fingerprint = hashlib.sha256(temporary.read_bytes()).hexdigest()
        temporary.replace(output)
        output.with_suffix(".source.json").write_text(
            json.dumps({"source_url": NEWS_URL, "sha256": fingerprint}, indent=2) + "\n"
        )
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output


def main():
    """Acquire assets separately from measured runs; exit 2 on acquisition errors."""
    p = argparse.ArgumentParser(
        description="Fetch a pinned model before running the workload ladder"
    )
    p.add_argument("model", choices=[*MODELS, "news"])
    p.add_argument("--cache", type=Path, default=Path("data/models"))
    p.add_argument("--data-output", type=Path, default=Path("data/ag-news.csv"))
    args = p.parse_args()
    try:
        if args.model == "news":
            print(fetch_news(args.data_output))
        else:
            print(resolve(args.model, args.cache, download=True)[0])
    except (ImportError, OSError, ValueError) as exc:
        p.exit(2, f"model acquisition failed: {exc}; install --extra pipeline if needed\n")
