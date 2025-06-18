from cartridges.clients.tokasaurus_batch import TokasaurusBatchClient
import asyncio

client = TokasaurusBatchClient.Config(
    url="https://hazyresearch--tksrs-entry-capsules-3b-1xh100-min0-max64-serve.modal.run",
    ports=None,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    max_retries=1,
).instantiate()

out = asyncio.run(client.chat_async(
    chats=[
        [
            {"role": "user", "content": "Hello, how are you?"}
        ]
    ],
    max_completion_tokens=1024,
    temperature=0.6,
    top_logprobs=20,
))

print(out)