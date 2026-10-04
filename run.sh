#!/usr/bin/env bash
# Build a reproducible, machine-local performance profile. This script expands
# only supported runtime/device/precision combinations; an explicit accelerator
# is never replaced with CPU.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./run.sh [options]

Create an isolated performance-profile matrix for one machine. The matrix
includes lifecycle train/infer pairs, operational NLP workloads, and the small
learning workloads supported by the selected hardware profile.

Required selection:
  --profile PROFILE        cpu, apple, or cuda (default: cpu)

Matrix selection:
  --families LIST          Comma-separated: classification,transformer,embedding,
                           operational,micro,all (default: all)
  --precisions SET         all or fp32. "all" adds fp16 and supported bf16 on CUDA.
  --measurement MODE       standard or diagnostic NLP phase timing (default: standard)

Work budget:
  --batch N                Batch size (default: 4)
  --length N               Sequence length (default: 32)
  --width N                Learning-model width (default: 32)
  --steps N                Measured batches per trial (default: 3)
  --warmup N               Discarded warmup batches (default: 1)
  --repeats N              Independently reset trials (default: 1)
  --limit N                Maximum news rows/pairs (default: 128)
  --seed N                 Reproducibility seed (default: 42)

Labels and files:
  --node LABEL             Public-safe machine label (default: unspecified)
  --condition LABEL        User-managed operating condition (default: unspecified)
  --output DIR             Parent for one new matrix directory (default: results)
  --model-cache DIR        Cached model directory passed to NLP runs
  --coffee-csv PATH        Optional tabular CSV for coffee cases
  --coffee-target NAME     Categorical target column for --coffee-csv
  --coffee-features LIST   Comma-separated numeric feature columns for --coffee-csv

Control:
  --skip-checks            Skip ruff/pytest checks; benchmark hardware is still validated.
  --dry-run                Print the expanded plan without syncing, downloading, or running.
  -h, --help               Show this help.

Examples:
  ./run.sh --profile cpu --node cpu-lab-a --condition idle
  ./run.sh --profile apple --families classification,micro --batch 32 --steps 50
  ./run.sh --profile cuda --precisions all --measurement diagnostic --repeats 3

Downloads and paired-data creation occur before measured cases. Reports retain
the actual runtime, device, data fingerprint, precision, and synthetic-data label.
Static TOML suites and report comparison/export intentionally remain separate CLI
operations: they need user-selected manifests or report paths rather than a machine
profile permutation.
EOF
}

die() { echo "error: $*" >&2; exit 2; }
require_value() { [[ $# -ge 2 ]] || die "$1 requires a value"; }
is_positive_integer() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
is_nonnegative_integer() { [[ "$1" =~ ^[0-9]+$ ]]; }

profile=cpu; families=all; precision_set=all; measurement=standard
batch=4; length=32; width=32; steps=3; warmup=1; repeats=1; limit=128; seed=42
node=unspecified; condition=unspecified; output_parent=results; model_cache=""
coffee_csv=""; coffee_target=""; coffee_features=""; skip_checks=false; dry_run=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile|--families|--precisions|--measurement|--batch|--length|--width|--steps|--warmup|--repeats|--limit|--seed|--node|--condition|--output|--model-cache|--coffee-csv|--coffee-target|--coffee-features)
      require_value "$@"
      case "$1" in
        --profile) profile="$2" ;; --families) families="$2" ;;
        --precisions) precision_set="$2" ;; --measurement) measurement="$2" ;;
        --batch) batch="$2" ;; --length) length="$2" ;; --width) width="$2" ;;
        --steps) steps="$2" ;; --warmup) warmup="$2" ;; --repeats) repeats="$2" ;;
        --limit) limit="$2" ;; --seed) seed="$2" ;; --node) node="$2" ;;
        --condition) condition="$2" ;; --output) output_parent="$2" ;;
        --model-cache) model_cache="$2" ;; --coffee-csv) coffee_csv="$2" ;;
        --coffee-target) coffee_target="$2" ;; --coffee-features) coffee_features="$2" ;;
      esac
      shift 2 ;;
    --skip-checks) skip_checks=true; shift ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (use --help)" ;;
  esac
done

case "$profile" in cpu|apple|cuda) ;; *) die "--profile must be cpu, apple, or cuda" ;; esac
case "$precision_set" in all|fp32) ;; *) die "--precisions must be all or fp32" ;; esac
case "$measurement" in standard|diagnostic) ;; *) die "--measurement must be standard or diagnostic" ;; esac
for value in "$batch" "$length" "$width" "$steps" "$warmup" "$repeats" "$limit"; do
  is_positive_integer "$value" || die "work-budget values must be positive integers"
done
is_nonnegative_integer "$seed" || die "--seed must be a non-negative integer"
if [[ -n "$coffee_csv" ]]; then
  [[ -n "$coffee_target" && -n "$coffee_features" ]] || die "--coffee-csv requires --coffee-target and --coffee-features"
elif [[ -n "$coffee_target$coffee_features" ]]; then
  die "--coffee-target and --coffee-features require --coffee-csv"
fi

if [[ "$families" == all ]]; then
  families=classification,transformer,embedding,operational,micro
else
  IFS=',' read -r -a selected_families <<< "$families"
  [[ ${#selected_families[@]} -gt 0 ]] || die "--families cannot be empty"
  for family in "${selected_families[@]}"; do
    case "$family" in classification|transformer|embedding|operational|micro) ;; *) die "unsupported family in --families: $family" ;; esac
  done
fi
has_family() { [[ ",$families," == *",$1,"* ]]; }

command -v uv >/dev/null || die "install uv first: https://docs.astral.sh/uv/getting-started/installation/"
[[ -f pyproject.toml ]] || die "run this script from the repository root"
case "$profile" in
  cpu) extras=(--extra cpu --extra pipeline); torch_devices=(cpu); mlx_devices=() ;;
  apple) extras=(--extra cpu --extra pipeline --extra mlx); torch_devices=(cpu mps); mlx_devices=(cpu gpu) ;;
  cuda) extras=(--extra cuda --extra pipeline); torch_devices=(cpu cuda); mlx_devices=() ;;
esac
UV=(uv run --locked "${extras[@]}")

if [[ "$dry_run" == true ]]; then
  out="${output_parent%/}/matrix-${profile}-DRY-RUN"
else
  mkdir -p "$output_parent"
  out="$(mktemp -d "${output_parent%/}/matrix-${profile}-XXXXXX")"
fi
summary="$out/status.tsv"; news_data=data/ag-news.csv; pairs_data="$out/news-pairs.csv"; checkpoint_dir="$out/checkpoints"
common=(--batch "$batch" --length "$length" --width "$width" --steps "$steps" --warmup "$warmup" --repeats "$repeats" --limit "$limit" --seed "$seed" --node "$node" --condition "$condition")
nlp_common=(--profile "$measurement")
[[ -n "$model_cache" ]] && nlp_common+=(--model-cache "$model_cache")
coffee_args=(); [[ -n "$coffee_csv" ]] && coffee_args=(--data "$coffee_csv" --target "$coffee_target" --features "$coffee_features")

if [[ "$dry_run" == false ]]; then
  uv sync --locked "${extras[@]}" --group dev
  printf 'case\tstatus\n' > "$summary"
  # An explicit accelerator request fails here; it is never silently downgraded.
  "${UV[@]}" python - "$profile" <<'PY'
import platform
import sys
import torch
profile = sys.argv[1]
if profile == "apple":
    assert platform.system() == "Darwin" and platform.machine() == "arm64", "The apple profile requires native Apple Silicon macOS."
    assert torch.backends.mps.is_available(), "PyTorch MPS is unavailable."
    import mlx.core as mx
    assert mx.metal.is_available(), "MLX Metal GPU is unavailable."
elif profile == "cuda":
    assert torch.cuda.is_available(), "CUDA is unavailable; check the host driver."
print(f"Hardware profile available: {profile}")
PY
  printf 'accelerator-preflight\tPASS\n' >> "$summary"
  if [[ "$skip_checks" == false ]]; then
    echo "Running static checks and CPU-oriented tests (separate from native accelerator cases)."
    "${UV[@]}" ruff check .; "${UV[@]}" ruff format --check .; "${UV[@]}" pytest
    printf 'development-checks\tPASS\n' >> "$summary"
    if [[ "$profile" == apple ]]; then
      echo "Running opt-in native MLX verification separately."
      "${UV[@]}" pytest --run-mlx
      printf 'native-mlx-tests\tPASS\n' >> "$summary"
    fi
  fi
  # Acquisition and paired-data creation are deliberately outside timed cases.
  # A partial family selection does not fetch unrelated public checkpoints.
  if has_family embedding || has_family operational || has_family micro; then
    "${UV[@]}" runtime-bench fetch news
  fi
  if has_family embedding; then
    "${UV[@]}" runtime-bench fetch minilm
    "${UV[@]}" runtime-bench embedding pairs --data "$news_data" --output "$pairs_data" --limit "$limit"
  fi
  if has_family operational; then
    for asset in minilm distilbert bert; do "${UV[@]}" runtime-bench fetch "$asset"; done
  fi
  mkdir -p "$checkpoint_dir"
else
  echo "Dry run: no environment, hardware, assets, or workloads will be touched."
  mkdir -p "$out"; printf 'case\tstatus\n' > "$summary"
fi

failures=0
run_case() {
  local name="$1"; shift
  local destination="$out/$name"
  if [[ "$dry_run" == true ]]; then
    printf 'PLAN\t%s\n' "${UV[*]} runtime-bench $* ${common[*]} --output $destination"
    printf '%s\tPLAN\n' "$name" >> "$summary"; return
  fi
  mkdir -p "$destination"; echo "Running: $name"
  if "${UV[@]}" runtime-bench "$@" "${common[@]}" --output "$destination" >"$destination/console.log" 2>&1; then
    printf '%s\tPASS\n' "$name" >> "$summary"
  else
    printf '%s\tFAIL\n' "$name" >> "$summary"; failures=$((failures + 1)); echo "  Failed; inspect $destination/console.log"
  fi
}
torch_precisions() {
  local device="$1"
  if [[ "$device" != cuda || "$precision_set" == fp32 ]]; then printf '%s\n' fp32; return; fi
  printf '%s\n' fp32 fp16
  if [[ "$dry_run" == true ]] || "${UV[@]}" python -c 'import torch; raise SystemExit(not torch.cuda.is_bf16_supported())'; then printf '%s\n' bf16; else echo "Skipping BF16: selected CUDA device does not support it."; fi
}

for device in "${torch_devices[@]}"; do
  while IFS= read -r precision; do
    backend=(--runtime torch --device "$device" --precision "$precision"); key="torch-${device}-${precision}"
    if has_family classification; then
      checkpoint="$checkpoint_dir/classification-$key.pt"
      run_case "classification-train-$key" classification --mode train "${backend[@]}" --checkpoint "$checkpoint"
      run_case "classification-infer-$key" classification --mode infer "${backend[@]}" --checkpoint "$checkpoint"
    fi
    if has_family transformer; then
      checkpoint="$checkpoint_dir/transformer-$key.pt"
      run_case "transformer-train-$key" transformer --mode train "${backend[@]}" --checkpoint "$checkpoint"
      run_case "transformer-infer-$key" transformer --mode infer "${backend[@]}" --checkpoint "$checkpoint"
    fi
    if has_family embedding; then
      checkpoint="$checkpoint_dir/embedding-$key"
      run_case "embedding-train-$key" embedding --mode train "${backend[@]}" --data "$pairs_data" --checkpoint "$checkpoint" --model minilm "${nlp_common[@]}"
      run_case "embedding-infer-$key" embedding --mode infer "${backend[@]}" --data "$pairs_data" --checkpoint "$checkpoint" "${nlp_common[@]}"
    fi
    if has_family operational; then
      if [[ "$device" == cpu && "$precision" == fp32 ]]; then
        run_case classify-synthetic classify --runtime torch --device cpu --precision fp32
        [[ ${#coffee_args[@]} -gt 0 ]] && run_case classify-coffee classify --runtime torch --device cpu --precision fp32 "${coffee_args[@]}"
      fi
      for model in minilm distilbert bert; do run_case "embeddings-$model-$key" embeddings "${backend[@]}" --model "$model" --data "$news_data" "${nlp_common[@]}"; done
      for model in distilbert bert; do
        run_case "infer-$model-$key" infer "${backend[@]}" --model "$model" --data "$news_data" "${nlp_common[@]}"
        run_case "finetune-$model-$key" finetune "${backend[@]}" --model "$model" --data "$news_data" "${nlp_common[@]}"
      done
    fi
    if has_family micro; then
      for mode in train infer; do
        for task in smoke stateful; do run_case "$task-$mode-$key" "$task" "${backend[@]}" --mode "$mode"; done
        run_case "news-$mode-$key" news "${backend[@]}" --mode "$mode" --data "$news_data"
        [[ ${#coffee_args[@]} -gt 0 ]] && run_case "coffee-$mode-$key" coffee "${backend[@]}" --mode "$mode" "${coffee_args[@]}"
      done
    fi
  done < <(torch_precisions "$device")
done

if [[ "$profile" == apple ]]; then
  for device in "${mlx_devices[@]}"; do
    backend=(--runtime mlx --device "$device" --precision fp32); key="mlx-${device}-fp32"
    if has_family classification; then
      checkpoint="$checkpoint_dir/classification-$key.pt"
      run_case "classification-train-$key" classification --mode train "${backend[@]}" --checkpoint "$checkpoint"
      run_case "classification-infer-$key" classification --mode infer "${backend[@]}" --checkpoint "$checkpoint"
    fi
    if has_family micro; then
      for mode in train infer; do
        run_case "smoke-$mode-$key" smoke "${backend[@]}" --mode "$mode"
        [[ ${#coffee_args[@]} -gt 0 ]] && run_case "coffee-$mode-$key" coffee "${backend[@]}" --mode "$mode" "${coffee_args[@]}"
      done
    fi
  done
fi

cat "$summary"
echo "Reports, logs, checkpoints, and paired-data fingerprint: $out"
echo "Failed workload cases: $failures"
[[ "$failures" -eq 0 ]]
