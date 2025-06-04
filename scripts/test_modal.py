from cartridges.clients.sglang import SGLangClient
from cartridges.clients.tokasaurus_batch import TokasaurusBatchClient

client = SGLangClient.Config(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    url="https://hazyresearch--sglang-llama-3-2-3b-instruct-h100-serve.modal.run",
).instantiate()

# client = TokasaurusBatchClient.Config(
#     url="https://hazyresearch--tksrs-entry-capsules-3b-1xh100-min0-max64-serve.modal.run",
#     ports=None,
#     model_name="meta-llama/Llama-3.2-3B-Instruct",
# ).instantiate()


print("Sending request")
response = client.chat(
    chats=[
        [
            {"role": "user", "content": "Hello, how are you?" * 8},
        ] * 10,
    ],
    top_logprobs=20,
    max_completion_tokens=1,
    logprobs_start_message=0,
)
print(response.samples[0].top_logprobs.top_ids.shape)

response = client.chat(
    chats=[
        [
            {"role": "user", "content": "Hello, how are you?" * 8},
        ] * 10,
    ],
    top_logprobs=20,
    max_completion_tokens=1,
    logprobs_start_message=4,
)
print(response.samples[0].top_logprobs.top_ids.shape)
