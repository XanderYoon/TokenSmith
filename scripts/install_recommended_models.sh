#!/usr/bin/env bash
set -Eeuo pipefail

PROFILE="${1:-all}"
MODELS_DIR="${2:-models}"

download_model() {
  local repo_id="$1"
  local filename="$2"
  python -m huggingface_hub download \
    "$repo_id" \
    "$filename" \
    --local-dir "$MODELS_DIR" \
    --local-dir-use-symlinks False
}

mkdir -p "$MODELS_DIR"

case "$PROFILE" in
  baseline)
    download_model "Qwen/Qwen3-Embedding-4B-GGUF" "Qwen3-Embedding-4B-Q5_K_M.gguf"
    download_model "Qwen/Qwen2.5-3B-Instruct-GGUF" "qwen2.5-3b-instruct-q8_0.gguf"
    ;;
  gpu)
    download_model "Qwen/Qwen3-Embedding-4B-GGUF" "Qwen3-Embedding-4B-Q8_0.gguf"
    download_model "lmstudio-community/Qwen2.5-7B-Instruct-GGUF" "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    ;;
  all)
    download_model "Qwen/Qwen3-Embedding-4B-GGUF" "Qwen3-Embedding-4B-Q5_K_M.gguf"
    download_model "Qwen/Qwen2.5-3B-Instruct-GGUF" "qwen2.5-3b-instruct-q8_0.gguf"
    download_model "Qwen/Qwen3-Embedding-4B-GGUF" "Qwen3-Embedding-4B-Q8_0.gguf"
    download_model "lmstudio-community/Qwen2.5-7B-Instruct-GGUF" "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    ;;
  *)
    echo "Usage: $0 [baseline|gpu|all] [models_dir]" >&2
    exit 1
    ;;
esac

echo "Downloaded $PROFILE model profile into $MODELS_DIR"
