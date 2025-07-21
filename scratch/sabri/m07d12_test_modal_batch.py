from cartridges.clients.tokasaurus_batch import TokasaurusBatchClient




if __name__ == "__main__":
    client = TokasaurusBatchClient.Config(
        url="https://hazyresearch--tksrs-batch-qwen3-0-6b-1xa100-80gb-min1-serve.modal.run/",
        model_name="Qwen/Qwen3-0.6B",
    ).instantiate()

    out = client.chat(
        chats=[
            [{"role": "user", "content": "Hello, how are you?"}],
        ],
        max_completion_tokens=100,
        top_logprobs=10,
    )
breakpoint()