import asyncio

from cartridges.clients.base import CartridgeConfig
from cartridges.clients.tokasaurus import TokasaurusClient

client = TokasaurusClient.Config(
    url="https://hazyresearch--toka-llama-3-2-3b-1xh100-cartridges-serve.modal.run",
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    cartridges=[
        CartridgeConfig(
            id="wv3q9fae",
            source="wandb",
            force_redownload=False
        ),
    ]
).instantiate()

TEXT = """Can you tell me about your context?"""

response = client.chat(
    chats=[[{"role": "user", "content": TEXT}]],
    max_completion_tokens=50,
    temperature=0.0,
    cartridges=None
)

response = asyncio.run(response)
print(response.samples[0].text)
