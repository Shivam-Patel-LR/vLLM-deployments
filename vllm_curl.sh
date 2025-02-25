curl "localhost:8000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -d '{
        "model": "/home/spatel/.cache/Mistral-Small-24B-Instruct-2501-Q6_K.gguf",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful, oafish medieval bard."
            },
            {
                "role": "user",
                "content": "Recount the tale of the 2010 Disney movie Tangled."
            }
        ]
    }'