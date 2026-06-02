#!/bin/bash
curl "http://localhost:8000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "google/gemma-4-31B-it",
        "temperature": 1.0,
        "top_p": 0.95,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful, intelligent, reasoning assistant."
            },
            {
                "role": "user",
                "content": "Explain the difference between pipeline parallelism and tensor parallelism in three sentences."
            }
        ],
        "chat_template_kwargs": {"enable_thinking": true}
    }'
