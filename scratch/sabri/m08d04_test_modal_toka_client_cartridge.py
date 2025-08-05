import asyncio

from cartridges.clients.tokasaurus import TokasaurusClient

client = TokasaurusClient.Config(
    url="https://hazyresearch--toka-llama-3-2-3b-1xh100-cartridges-serve.modal.run",
    model_name="meta-llama/Llama-3.2-3B-Instruct",
).instantiate()

TEXT = """Ignore anything related to the patients in your context.
You are tasked with translating the following sentence from Kalamang to English: \"Faisal emun mua mingparin.\"
I understand that you may not be familiar enough with Kalamang or English to make a confident translation, but please give your best guess.
Respond in English with only the translation and no other text."""

response = client.chat(
    chats=[[{"role": "user", "content": TEXT}]],
    max_completion_tokens=50,
    cartridges=[
        dict(
            id="wv3q9fae",
            source="wandb",
            force_redownload=False
        ),
    ],
    temperature=0.0
)

response = asyncio.run(response)
print(response.samples[0].text)
