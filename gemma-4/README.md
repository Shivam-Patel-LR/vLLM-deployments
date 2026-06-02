# Gemma-4 31B IT (vLLM)

Docker-compose deployment of `google/gemma-4-31B-it` served via the vLLM OpenAI-compatible API on port 8000. Uses tensor parallelism across both GPUs.

## Files

- `dockerfile` - builds on `vllm/vllm-openai:gemma4` and bakes in `config.yaml` + entrypoint
- `docker-compose.yml` - single `vllm-gen` service, all GPUs, reads `../.env`
- `config.yaml` - vLLM serve flags (TP=2, 128k context, FP8 KV cache, tool + reasoning parsers)
- `init_local_deployment.sh` - prepares `~/.cache` and runs `docker compose up -d`
- `vllm_curl.sh` - quick chat-completion smoke test
- `test_vllm.py` - Python test harness hitting the OpenAI endpoint

## Requirements

- 2x NVIDIA L40S (or equivalent, ~90 GB total VRAM)
- NVIDIA Container Toolkit
- `../.env` with `HUGGING_FACE_HUB_TOKEN` (gated model)

## Bring up

```bash
./init_local_deployment.sh
docker compose logs -f vllm-gen    # watch startup
./vllm_curl.sh                     # smoke test once healthy
```

Model load + CUDA graph capture takes several minutes; the healthcheck allows 15 min.

## Tear down

```bash
docker compose down
```
