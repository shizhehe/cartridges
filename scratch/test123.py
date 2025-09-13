import asyncio
from cartridges.clients.tokasaurus import TokasaurusClient


client = TokasaurusClient.Config(
    url="https://hazyresearch--toka-qwen3-8b-1xh100-cartridges-serve.modal.run",
    model_name="Qwen/Qwen3-8B",
    # cartridges=cartridges
).instantiate()

response = asyncio.run(client.chat(
    chats=[
        [
            {"role": "user", "content": "What is the capital of France?"}
        ]
    ],
    max_completion_tokens=1024,
    temperature=0.0,
    enable_thinking=False,
))
print(response)