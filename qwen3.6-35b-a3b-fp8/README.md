# Qwen3.6-35B-A3B-FP8 (vLLM)

Deployment of `Qwen/Qwen3.6-35B-A3B-FP8` — a multimodal gated-delta MoE (35.95B total, 3B active; 256 experts, 8 routed + 1 shared) — via the vLLM OpenAI-compatible API on port **8000**, tensor-parallel across **both L40S GPUs**.

Reference: [vLLM Qwen3.5 / Qwen3.6 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html), [HF model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8).

## Files

- `docker-compose.yml` - `vllm-qwen` service, both GPUs, `vllm/vllm-openai:v0.19.0`
- `config.yaml` - TP=2, 64K context, qwen3 reasoning parser, qwen3_coder tool parser, MM shm cache

## Requirements

- 2x NVIDIA L40S (FP8 weights ~36 GB; KV + activations need the rest across both GPUs)
- `../.env` with `HUGGING_FACE_HUB_TOKEN`
- First boot downloads ~36 GB — allow 10-20 min before healthy

## Bring up

```bash
docker compose up -d
docker compose logs -f vllm-qwen
```

## Smoke test

Text chat:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.6-35B-A3B-FP8",
    "messages": [{"role": "user", "content": "In three sentences, what is quantization-aware distillation?"}],
    "max_tokens": 512
  }'
```

Vision / document-intelligence:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.6-35B-A3B-FP8",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png"}},
        {"type": "text", "text": "Extract all text from this image as markdown."}
      ]
    }],
    "max_tokens": 2048
  }'
```

## Tuning

- To disable thinking mode by default: add `--default-chat-template-kwargs '{"enable_thinking": false}'` to the entrypoint, or send `chat_template_kwargs: {"enable_thinking": false}` per-request.
- To push context beyond 64K: raise `max-model-len` in `config.yaml` (native is 262144). Beyond that, use YaRN (see recipe).
- If KV-cache OOMs at boot: drop `max-model-len` to 32768 or lower `gpu-memory-utilization` to 0.88.

## Tear down

```bash
docker compose down
```
