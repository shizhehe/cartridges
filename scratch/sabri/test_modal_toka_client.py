from cartridges.clients.tokasaurus import TokasaurusClient
import asyncio

async def send_request(client, chat_data):
    return await client.chat(
        chats=chat_data,
        max_completion_tokens=1024,
        top_logprobs=32,
        temperature=0.,
    )

async def main():
    client = TokasaurusClient.Config(
        # url="https://hazyresearch--toka-qwen3-4b-1xh100-min0-serve.modal.run",
        url="http://0.0.0.0:10210",
        # model_name="Qwen/Qwen3-4b",
        model_name="meta-llama/Llama-3.2-3B-Instruct"
    ).instantiate()

    chat_data = [
        [{"role": "user", "content": "Hello, how are you?" * 1024}]
    ] * 128

    print("Awaiting responses...")
    await send_request(client, chat_data) 
    print("Responses received")
    # responses = await asyncio.gather(*tasks)
    # print("Responses received")


asyncio.run(main())
