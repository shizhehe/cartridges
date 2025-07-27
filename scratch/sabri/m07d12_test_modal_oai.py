import openai

from cartridges.data.longhealth.resources import LongHealthResource

openai.api_key = "your-api-key"

client = openai.OpenAI(
    base_url="https://hazyresearch--vllm-qwen3-4b-1xh100-serve.modal.run/v1"
)

NUM_PATIENTS = 10
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]

resource = LongHealthResource.Config(
    patient_ids=patient_ids,
).instantiate()

for idx in range(10):
    response = client.chat.completions.create(
        model="Qwen/Qwen3-4b",
        messages=[
            {"role": "user", "content": resource.to_string()}
        ],
        max_tokens=100,
        
    )

    breakpoint()

breakpoint()


