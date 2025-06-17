import datasets

from cartridges.clients.openai_batch import TokasaurusClient

DATASET_NAME: str = "ScalingIntelligence/mmlu-llama3b-generations"


def make_prompt(question, choices):
    return f"""
Follow these instructions exactly:
1. Provide a detailed explanation of your reasoning.
2. Output your final answer (one of A, B, C, or D) between the tokens <answer> and </answer>.
3. Format your answer exactly as specified to enable easy auto-grading.

Now, solve the following multiple-choice question.

Question:
{question}

Choices:
A: {choices[0]}
B: {choices[1]}
C: {choices[2]}
D: {choices[3]}

Think through the problem and output your final answer (one of A, B, C, or D) between <answer> and </answer>.
"""


def main():
    rows = datasets.load_dataset("cais/mmlu", "all")["test"]

    print(f"Running on {len(rows)} rows")
    
    questions = []
    answers = []

    for row in rows:
        questions.append(
            [
                {
                    "role": "user",
                    "content": make_prompt(
                        row["question"],
                        row["choices"],
                    ),
                }
            ]
        )
        answers.append(row["answer"])

    client = TokasaurusClient.Config(
        url="http://localhost:8012/v1", model_name="not_empty"
    ).instantiate()

    response = client.chat(
        chats=questions,
        temperature=0.6,
        top_logprobs=1,
        max_completion_tokens=512,
    )
    print("Got response")

    dataset_dicts = [
        {
            "question": row["question"],
            "prompt": chat[0]['content'],
            "gt_answer": answer,
            "answer": sample.text,
        }
        for row, sample, answer, chat in zip(
            rows,
            response.samples,
            answers,
            questions
        )
    ]
    dataset = datasets.Dataset.from_list(dataset_dicts, split="test")
    dataset.push_to_hub(DATASET_NAME, private=True, max_shard_size="256MB")


if __name__ == "__main__":
    main()
