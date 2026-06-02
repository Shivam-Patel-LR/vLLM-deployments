# MinerU2.5 Extraction (vLLM)

Docker-compose deployment of `opendatalab/MinerU2.5-Pro-2604-1.2B` (PDF page
image -> layout + content blocks) on the vLLM OpenAI-compatible API, port 8000.
Reaches **~7.4 pages/s total / ~3.7 pages/s/GPU on 2x L40S**.

## Files

- `dockerfile` - builds on `vllm/vllm-openai:v0.21.0`, bakes in `mineru-vl-utils[vllm]` + `config.yaml` + entrypoint
- `docker-compose.yml` - single `vllm-mineru` service, all GPUs, reads `../.env`
- `config.yaml` - vLLM serve flags (data-parallel=2, MinerU logits processor, renderer pool)

## Requirements

- 2x NVIDIA L40S (or equivalent)
- NVIDIA Container Toolkit
- `../.env` with `HUGGING_FACE_HUB_TOKEN`

## Bring up

```bash
docker compose up -d --build
docker compose logs -f vllm-mineru     # watch startup
curl -s localhost:8000/v1/models       # ready check once healthy
```

First boot downloads the model + CUDA graph capture; the healthcheck allows 15 min.

## Load-bearing config

Each flag in `config.yaml` is required, not tuning. `hf-overrides` re-ties the
LM head (else garbage output); `logits-processors` is the colon module:class
form from `mineru-vl-utils`; `data-parallel-size 2` runs one 1.2B replica per
GPU (do not tensor-parallel); `renderer-num-workers 16` keeps the GPU fed and
requires `mm-processor-cache-gb 0` (that cache is not thread-safe). Leave
`max-num-seqs` at its default - raising it over-batches and reduces throughput.

## Client (throughput)

The two-step layout -> per-block protocol fires ~26 small requests/page; a
single asyncio client starves the engine. Run **~24 client processes over
disjoint page shards** to saturate the GPU (~80% power at the ceiling).

```python
from mineru_vl_utils import MinerUClient
client = MinerUClient(backend="http-client", server_url="http://localhost:8000",
                      model_name="MinerU2.5-Pro-2604-1.2B", max_concurrency=64)
results = client.concurrent_two_step_extract(images)  # list[PIL.Image]
```

## Tear down

```bash
docker compose down
```
