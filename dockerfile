FROM  vllm/vllm-openai:v0.9.1@sha256:0b51ec38fb965b44f6aa75d8d847c5f21bc062b7140e1d83444b39b67fc4a2ea

ENV VLLM_ATTENTION_BACKEND="FLASH_ATTN"
ENV VLLM_USE_TRITON_FLASH_ATTN=1
ENV VLLM_FLASH_ATTN_VERSION=2
ENV VLLM_USE_TRITON_AWQ=1
ENV VLLM_MARLIN_USE_ATOMIC_ADD=1

# RUN pip install --no-cache-dir flash-attn

WORKDIR /app

COPY ./config.yaml /app/config.yaml
COPY ./chat_template.jinja /app/chat_template.jinja
# COPY ./qwen_3_prompt_template.jinja /app/qwen_3_prompt_template.jinja

EXPOSE 8000

ENV VLLM_LOGGING_LEVEL=DEBUG
ENV VLLM_TRACE_FUNCTION=1
ENV VLLM_LOG_REQUESTS=1

ENTRYPOINT ["vllm", "serve", "Qwen/Qwen3-32B-AWQ", "--config", "config.yaml"]