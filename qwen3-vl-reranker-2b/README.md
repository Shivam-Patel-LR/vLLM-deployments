# Qwen3-VL-Reranker-2B (vLLM)

Deployment of `Qwen/Qwen3-VL-Reranker-2B` - a 2.13B-parameter multimodal reranker built on Qwen3-VL-2B-Instruct - via the vLLM OpenAI-compatible `/v1/score` API on port **8002**. Pinned to **GPU 1**.

Scoring is produced by the "yes"/"no" classifier head applied on top of the Qwen3-VL base via the `hf-overrides` injection (see `config.yaml`).

Reference: [HF model card](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B), [vLLM reranker template](https://github.com/vllm-project/vllm/blob/main/examples/pooling/score/template/qwen3_vl_reranker.jinja).

## Files

- `docker-compose.yml` - `vllm-qwen3-vl-rerank` service, GPU 1, runs the upstream `vllm/vllm-openai:v0.20.0` image
- `config.yaml` - pooling runner, bfloat16, 32k context, hf-overrides for classifier head
- `start.sh` - resolves the repo-native reranker template from the HF cache, then launches `vllm serve`

## Chat template source

The service now refreshes the non-weight files for
`Qwen/Qwen3-VL-Reranker-2B` with `hf download` during container startup,
resolves the selected revision, and passes the repo-native
`additional_chat_templates/reranker.jinja` file to `vllm serve` via
`--chat-template`.

This avoids keeping a drift-prone local copy of the reranker template in this
repo. The `hf download` step excludes weight formats so it refreshes template
and config files without proactively re-pulling the model blobs; if vLLM later
needs a weight blob for a new snapshot and the blob hash is already present in
the HF cache, Hugging Face will reuse the cached blob rather than fetch it
again.

The startup script is cache-root agnostic. Set `HF_HOME` and `HF_HUB_CACHE`
to wherever your platform persists the Hugging Face cache.

## Requirements

- 1x NVIDIA L40S (BF16 weights ~4.5 GB + KV / activations)
- `../.env` with `HUGGING_FACE_HUB_TOKEN`

## Bring up

```bash
docker compose up -d
docker compose logs -f vllm-qwen3-vl-rerank
```

This compose file mounts the host HF cache at `/workspace/hf` inside the
container so the same layout works locally and on RunPod Pods.

## RunPod Pods

RunPod Pods persist data in `/workspace`, so use that as the cache root:

```bash
MODEL_ID=Qwen/Qwen3-VL-Reranker-2B
MODEL_REVISION=main
CHAT_TEMPLATE_NAME=reranker
HF_HOME=/workspace/hf
HF_HUB_CACHE=/workspace/hf/hub
```

Use the contents of `start.sh` as the Pod start command, or mount this repo
and run:

```bash
/bin/sh /app/start.sh
```

If you want fully reproducible startup on RunPod, pin `MODEL_REVISION` to the
exact HF commit hash instead of `main`.

## Smoke test

```bash
curl http://localhost:8002/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-Reranker-2B",
    "query": "A woman playing with her dog on a beach at sunset.",
    "documents": [
      "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset.",
      "A city skyline at night with neon lights reflecting off skyscrapers."
    ]
  }'
```

Returns a relevance score per document (higher = more relevant).

## Tear down

```bash
docker compose down
```
