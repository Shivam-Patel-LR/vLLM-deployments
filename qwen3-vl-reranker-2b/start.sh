#!/bin/sh
set -eu

: "${MODEL_ID:=Qwen/Qwen3-VL-Reranker-2B}"
: "${MODEL_REVISION:=main}"
: "${CHAT_TEMPLATE_NAME:=reranker}"
: "${HF_HOME:=/workspace/hf}"
: "${HF_HUB_CACHE:=$HF_HOME/hub}"

REPO_SLUG="$(printf "%s" "${MODEL_ID}" | sed 's#/#--#g')"
REPO_CACHE="${HF_HUB_CACHE}/models--${REPO_SLUG}"

hf download "${MODEL_ID}" --revision "${MODEL_REVISION}" \
  --exclude "*.safetensors" \
  --exclude "*.bin" \
  --exclude "*.pt" \
  --exclude "*.pth" \
  --exclude "*.ckpt" \
  --exclude "*.onnx" \
  --exclude "*.msgpack" \
  --exclude "*.h5" \
  --exclude "*.ot"

if [ "${MODEL_REVISION}" = "main" ]; then
  REV="$(cat "${REPO_CACHE}/refs/main")"
else
  REV="${MODEL_REVISION}"
fi

TEMPLATE="${REPO_CACHE}/snapshots/${REV}/additional_chat_templates/${CHAT_TEMPLATE_NAME}.jinja"

if [ ! -f "${TEMPLATE}" ]; then
  echo "Missing chat template: ${TEMPLATE}" >&2
  exit 1
fi

exec vllm serve "${MODEL_ID}" --config /app/config.yaml --chat-template "${TEMPLATE}"
