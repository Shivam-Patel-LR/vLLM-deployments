from openai import OpenAI

client = OpenAI(
      base_url='http://10.18.0.71:8000/v1',
      api_key='x'
  )

GENERATIVE_MODEL='/root/.cache/Qwen2.5-Coder-32B-Instruct-Q6_K.gguf'


response = client.chat.completions.create(
    model=GENERATIVE_MODEL, # Model = should match the deployment name you chose for your model deployment
    # response_format={ "type": "json_object" },
    messages=[
        {"role": "system", "content": 'You are a helpful, intelligent programming assistant.'},
        {"role": "user", "content": 'Write a program that demonstrates the size of different primitive type arrays in C. Remember to fence all code blocks and annotate the language.'}
    ],
    temperature=0.7
)

print(response.choices[0].message.content)