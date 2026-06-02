# Qwen3-VL-Embedding-2B (vLLM)

Deployment of `Qwen/Qwen3-VL-Embedding-2B` - a 2.13B-parameter multimodal embedding model (text, image, video) built on Qwen3-VL-2B-Instruct - via the vLLM OpenAI-compatible embeddings API on port **8001**. Pinned to **GPU 0**.

Reference: [HF model card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B).

## Files

- `docker-compose.yml` - `vllm-qwen3-vl-embed` service, GPU 0, `vllm/vllm-openai:v0.19.0`
- `config.yaml` - pooling runner, bfloat16, 32k context, shm MM cache

## Requirements

- 1x NVIDIA L40S (BF16 weights ~4.5 GB + KV / activations)
- `../.env` with `HUGGING_FACE_HUB_TOKEN`

## Bring up

```bash
docker compose up -d
docker compose logs -f vllm-qwen3-vl-embed
```

## Smoke test

Text embedding:

```bash
curl http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-Embedding-2B",
    "input": "A woman playing with her dog on a beach at sunset."
  }'
```

Output is a 2048-dim vector (configurable 64-2048 at training time).

## Tear down

```bash
docker compose down
```
