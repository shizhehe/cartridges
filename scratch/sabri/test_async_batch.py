from cartridges.clients.tokasaurus import TokasaurusClient
import asyncio
import requests


def main():
    
    base_url="http://0.0.0.0:10210/v1"
    batch_request = {
        "requests": [
            {
                "model": "Qwen/Qwen3-4b",
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
                "max_completion_tokens": 20,
                "temperature": 0.0,
                "apply_chat_template_overrides": {
                    "enable_thinking": True,
                },
                "logprobs_in_fingerprint": True,
                "logprobs": True,
                "top_logprobs": 10,
            },
        ] * 128
    }
    
    # Make request to our custom endpoint
    print("Sending request")
    response = requests.post(
        f"{base_url}/batch/chat/completions",
        json=batch_request,
        headers={"Authorization": f"Bearer sk-proj-1234567890"}
    )

    print(response.status_code)
    print("Responses received")
    # responses = await asyncio.gather(*tasks)
    # print("Responses received")

if __name__ == "__main__":
    main()