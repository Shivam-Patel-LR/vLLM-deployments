#! /bin/bash
if [ ! -d ~/.cache ]; then
    mkdir -p ~/.cache
fi
sudo chown -R $USER:$USER ~/.cache
chmod u+w ~/.cache
if [ ! -f ~/.cache/Qwen2.5-Coder-32B-Instruct-Q6_K.gguf ]; then
    wget -P ~/.cache https://huggingface.co/bartowski/Qwen2.5-Coder-32B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-32B-Instruct-Q6_K.gguf
fi
docker compose up --remove-orphans