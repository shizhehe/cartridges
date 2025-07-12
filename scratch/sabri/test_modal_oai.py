import openai

openai.api_key = "your-api-key"

for idx in range(10):
    response = openai.ChatCompletion.create(
        model="Qwen/Qwen3-0.6B",
        messages=[
            {"role": "user", "content": f"Hello, how are you? {idx}"}
        ],
        max_tokens=100,
        logprobs=10
    )

    print(response.choices[0].message['content'])

breakpoint()