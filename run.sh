#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash example-matrix.sh cpu
#   bash example-matrix.sh apple
#   bash example-matrix.sh cuda
#
# Optional coffee data:
#   COFFEE_CSV=data/coffee.csv \
#   COFFEE_TARGET=quality_category \
#   COFFEE_FEATURES=acidity,aroma,body \
#     bash example-matrix.sh apple
#
# Increase work after the small reference runs succeed:
#   BATCH=8 LENGTH=128 STEPS=100 REPEATS=3 \
#     bash example-matrix.sh apple

profile="${1:-cpu}"

BATCH="${BATCH:-4}"
LENGTH="${LENGTH:-32}"
STEPS="${STEPS:-3}"
REPEATS="${REPEATS:-1}"
LIMIT="${LIMIT:-128}"

command -v uv >/dev/null || {
  echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
}
[[ -f pyproject.toml ]] || {
  echo "Run this script from the repository root."
  exit 1
}

# Extras install packages; device/runtime arguments select execution.
case "$profile" in
  cpu)
    extras=(--extra cpu --extra pipeline)
    devices=(cpu)
    ;;
  apple)
    extras=(--extra cpu --extra pipeline --extra mlx)
    devices=(cpu mps)
    ;;
  cuda)
    extras=(--extra cuda --extra pipeline)
    devices=(cpu cuda)
    ;;
  *)
    echo "Profile must be cpu, apple, or cuda."
    exit 1
    ;;
esac

uv sync --locked "${extras[@]}" --group dev
UV=(uv run --locked "${extras[@]}")

# Fail early if the requested hardware profile is unavailable.
"${UV[@]}" python - "$profile" <<'PY'
import platform
import sys
import torch

profile = sys.argv[1]
if profile == "apple":
    assert platform.system() == "Darwin" and platform.machine() == "arm64", \
        "The apple profile requires native Apple Silicon macOS."
    assert torch.backends.mps.is_available(), "PyTorch MPS is unavailable."
    import mlx.core as mx
    assert mx.metal.is_available(), "MLX Metal GPU is unavailable."
elif profile == "cuda":
    assert torch.cuda.is_available(), "CUDA is unavailable; check the host driver."
print("Hardware profile available:", profile)
PY

# Validate the harness and adapters.
"${UV[@]}" ruff check .
"${UV[@]}" ruff format --check .
if [[ "$profile" == apple ]]; then
  "${UV[@]}" pytest --run-mlx
else
  "${UV[@]}" pytest
fi

# Acquisition happens before benchmarking. No model server is started.
# Reuses the local cache when already downloaded.
for asset in news minilm distilbert bert; do
  "${UV[@]}" runtime-bench-fetch "$asset"
done

# Make a small real-data subset for the learning "news" task.
# That task reads the whole supplied CSV; --limit does not constrain it.
NEWS_SMALL="$("${UV[@]}" python - "$LIMIT" <<'PY'
import csv
import itertools
import sys
import tempfile

with open("data/ag-news.csv", newline="", encoding="utf-8") as source:
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8",
        dir="data", prefix="news-example-", suffix=".csv", delete=False
    ) as target:
        csv.writer(target).writerows(
            itertools.islice(csv.reader(source), int(sys.argv[1]))
        )
        print(target.name)
PY
)"

# Separate directory for this invocation, including logs and JSON reports.
mkdir -p results
OUT="$(mktemp -d "results/example-${profile}-XXXXXX")"
SUMMARY="$OUT/status.tsv"
printf 'case\tstatus\n' > "$SUMMARY"

common=(
  --batch "$BATCH"
  --length "$LENGTH"
  --width 32
  --steps "$STEPS"
  --warmup 1
  --repeats "$REPEATS"
  --limit "$LIMIT"
  --seed 42
)

failures=0

# Keep running after an individual workload failure.
# Logs retain the error; the CLI also writes a failure report when possible.
run_case() {
  local name="$1"
  shift
  mkdir -p "$OUT/$name"
  echo "Running: $name"

  if "${UV[@]}" runtime-bench "$@" \
      "${common[@]}" --output "$OUT/$name" \
      >"$OUT/$name/console.log" 2>&1; then
    printf '%s\tPASS\n' "$name" >> "$SUMMARY"
  else
    printf '%s\tFAIL\n' "$name" >> "$SUMMARY"
    failures=$((failures + 1))
    echo "  Failed; inspect $OUT/$name/console.log"
  fi
}

coffee_args=()
if [[ -n "${COFFEE_CSV:-}" ]]; then
  : "${COFFEE_TARGET:?Set COFFEE_TARGET to a categorical target column}"
  : "${COFFEE_FEATURES:?Set COFFEE_FEATURES to comma-separated numeric columns}"
  coffee_args=(
    --data "$COFFEE_CSV"
    --target "$COFFEE_TARGET"
    --features "$COFFEE_FEATURES"
  )
fi

# sklearn is CPU-only, regardless of the installed accelerator packages.
run_case classify-synthetic classify --runtime torch --device cpu
if [[ ${#coffee_args[@]} -gt 0 ]]; then
  run_case classify-coffee classify \
    --runtime torch --device cpu "${coffee_args[@]}"
else
  echo "Skipping coffee cases: no COFFEE_CSV supplied."
fi

for device in "${devices[@]}"; do
  precisions=(fp32)

  if [[ "$device" == cuda ]]; then
    precisions+=(fp16)
    if "${UV[@]}" python -c \
      'import torch; raise SystemExit(0 if torch.cuda.is_bf16_supported() else 1)'
    then
      precisions+=(bf16)
    else
      echo "Skipping BF16: selected CUDA device does not support it."
    fi
  fi

  for precision in "${precisions[@]}"; do
    backend=(--runtime torch --device "$device" --precision "$precision")

    # Pretrained embeddings: all three encoders, with mean pooling.
    for model in minilm distilbert bert; do
      run_case "embeddings-$model-$device-$precision" embeddings \
        "${backend[@]}" --model "$model" --data data/ag-news.csv
    done

    # Pretrained fill-mask inference: MiniLM is deliberately excluded.
    for model in distilbert bert; do
      run_case "infer-$model-$device-$precision" infer \
        "${backend[@]}" --model "$model" --data data/ag-news.csv
    done

    # Full-model training with a new four-class classification head.
    for model in minilm distilbert bert; do
      run_case "finetune-$model-$device-$precision" finetune \
        "${backend[@]}" --model "$model" --data data/ag-news.csv
    done

    # Learning examples use freshly initialized models, not fetched weights.
    for mode in train infer; do
      for task in smoke stateful; do
        run_case "$task-$mode-$device-$precision" "$task" \
          "${backend[@]}" --mode "$mode"
      done

      run_case "news-$mode-$device-$precision" news \
        "${backend[@]}" --mode "$mode" --data "$NEWS_SMALL"

      if [[ ${#coffee_args[@]} -gt 0 ]]; then
        run_case "coffee-$mode-$device-$precision" coffee \
          "${backend[@]}" --mode "$mode" "${coffee_args[@]}"
      fi
    done
  done
done

# MLX supports only the matched learning MLPs, in FP32.
if [[ "$profile" == apple ]]; then
  for device in cpu gpu; do
    for mode in train infer; do
      run_case "smoke-$mode-mlx-$device" smoke \
        --runtime mlx --device "$device" --precision fp32 --mode "$mode"

      if [[ ${#coffee_args[@]} -gt 0 ]]; then
        run_case "coffee-$mode-mlx-$device" coffee \
          --runtime mlx --device "$device" --precision fp32 \
          --mode "$mode" "${coffee_args[@]}"
      fi
    done
  done
fi

cat "$SUMMARY"
echo "Reports and logs: $OUT"
echo "Failed workload cases: $failures"
[[ "$failures" -eq 0 ]]
