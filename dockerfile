FROM vllm/vllm-openai:latest@sha256:b168cbb0101f51b2491047345d0a83f5a8ecbe56a6604f2a2edb81eca55ebc9e

ENV VLLM_ATTENTION_BACKEND="FLASH_ATTN"
ENV VLLM_USE_TRITON_FLASH_ATTN=1
ENV VLLM_FLASH_ATTN_VERSION=2
ENV VLLM_USE_TRITON_AWQ=1
ENV VLLM_MARLIN_USE_ATOMIC_ADD=1

RUN pip install --no-cache-dir flash-attn

WORKDIR /app

EXPOSE 8000

# ENTRYPOINT ["vllm", "serve", "gaunernst/gemma-3-27b-it-int4-awq", "--enable-reasoning", "--reasoning-parser", "deepseek_r1", "--gpu-memory-utilization", "0.95", "--dtype", "auto", "--max-model-len", "32768", "--quantization", "awq_marlin", "--max-seq-len-to-capture", "8192", "--enable-chunked-prefill"]

ENTRYPOINT ["vllm", "serve", "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ", "--gpu-memory-utilization", "0.95", "--dtype", "auto", "--max-model-len", "32768", "--quantization", "awq_marlin", "--max-seq-len-to-capture", "8192", "--enable-chunked-prefill"]