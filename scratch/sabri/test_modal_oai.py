import requests

url = "https://fu-edh8vpwma2wnhmy8chmv8k.us-west.modal.direct/v1/chat/completions"

for idx in range(10):
    headers = {
        "Content-Type": "application/json",
        "X-Modal-Flash-Upstream": f"testing{idx%2}"
    }
    data = {
        "model": "Qwen/Qwen3-0.6B",
        "messages": [
            {"role": "user", "content": f"Hello, how are you? {idx}"}
        ],
        "max_tokens": 100,
        "logprobs": True,
        "top_logprobs": 10
    }

    response = requests.post(url, headers=headers, json=data)


print(response.json())
breakpoint()