FROM  vllm/vllm-openai:v0.10.1.1@sha256:d731ee65c044ae0977421eed3d93f931d4b7d79614394184c939db35b8f28fc2

# ENV VLLM_ATTENTION_BACKEND="FLASH_ATTN"
# ENV VLLM_USE_TRITON_FLASH_ATTN=1
# ENV VLLM_FLASH_ATTN_VERSION=2
# ENV VLLM_USE_TRITON_AWQ=1
# ENV VLLM_MARLIN_USE_ATOMIC_ADD=1

# RUN pip install --no-cache-dir flash-attn

WORKDIR /app

COPY ./config.yaml /app/config.yaml
COPY ./chat_template.jinja /app/chat_template.jinja
# COPY ./qwen_3_prompt_template.jinja /app/qwen_3_prompt_template.jinja

EXPOSE 8000

# ENV VLLM_LOGGING_LEVEL=DEBUG
# ENV VLLM_TRACE_FUNCTION=1
# ENV VLLM_LOG_REQUESTS=1

# ENTRYPOINT ["vllm", "serve", "Qwen/Qwen3-32B-AWQ", "--config", "config.yaml"]
ENTRYPOINT [ "vllm", "serve", "cpatonn/Hermes-4-70B-AWQ-4bit", "--config", "config.yaml"]