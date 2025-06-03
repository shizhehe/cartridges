from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from capsules.clients.anthropic import AnthropicClient
from capsules.generate.context_convo_generators.base import (
    ContextConvoGenerator,
    QuestionData,
)
from capsules.generate.structs import ContextConvo, Message
from capsules.utils.parsing import extract_last_json


QUESTION_INSTRUCTION_PROMPT = f"""Generate exactly {{samples}} distinct questions about this document.
Here are instructions that specify what kind of questions should be generated:
<instructions>
{{instructions}}
</instructions>

These questions should test a one's knowledge of the information in the document when asked in a closed book setting.

Generate every question according to these instructions.
Remember, it is important that you generate exactly {{samples}} questions.

First, think and brainstorm options.
Then, output a single json block (between triple backticks - ```) that contains a string list, where the elements of the list are the questions.
"""


class AnthropicQandA(ContextConvoGenerator):
    class Config(ContextConvoGenerator.Config):
        question_system_prompt_template: str
        answer_system_prompt_template: str
        instructions: str
        answer_prompt_template: str
        client: AnthropicClient.Config
        question_temperature: float
        max_paralleism: int = 3

    def get_answer(self, client: AnthropicClient, question: str) -> str:
        assert isinstance(self.config, AnthropicQandA.Config)
        prompt_system = self.config.answer_system_prompt_template.format(
            title=self.context.title,
            content=self.context.text,
        )

        response = client.chat(
            [
                [
                    {
                        "role": "system",
                        "content": prompt_system,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "role": "user",
                        "content": self.config.answer_prompt_template.format(
                            question=question
                        ),
                    },
                ]
            ],
            temperature=0.0,
            max_completion_tokens=8192,
        )
        assert len(response.samples) == 1
        return response.samples[0].text

    def sample_convos(self, start_idx: int, end_idx: int) -> list[ContextConvo]:
        num_convos = end_idx - start_idx
        assert num_convos > 0
        # generate questions
        assert isinstance(self.config, AnthropicQandA.Config)

        prompt_system = self.config.question_system_prompt_template.format(
            title=self.context.title,
            content=self.context.text,
        )

        prompt_instruction = QUESTION_INSTRUCTION_PROMPT.format(
            instructions=self.config.instructions,
            samples=num_convos,
        )

        client = self.config.client.instantiate()
        response = client.chat(
            chats=[
                [
                    {
                        "content": prompt_system,
                        "role": "system",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"content": prompt_instruction, "role": "user"},
                ]
            ],
            temperature=self.config.question_temperature,
        )

        content = response.samples[0].text
        questions = extract_last_json(content)

        try:
            assert isinstance(questions, list)
        except:
            breakpoint()
            raise
        
        if len(questions) != num_convos:
            breakpoint()

        assert len(questions) == num_convos
        for q in questions:
            assert isinstance(q, str)

        # generate answers
        # TODO(RE): don't warm cache if there's a single example
        # warm cache with one questions
        answers: list[str | None] = [None] * num_convos
        answers[0] = self.get_answer(client, questions[0])

        # Process the rest of the questions in parallel.
        if num_convos > 1:
            futures = {}
            with ThreadPoolExecutor(max_workers=self.config.max_paralleism) as executor:
                # Submit futures for questions 1..num_convos-1
                for question_idx in range(1, num_convos):
                    futures[
                        executor.submit(
                            self.get_answer,
                            client,
                            questions[question_idx],
                        )
                    ] = question_idx
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Generating answers"
                ):
                    question_idx = futures[future]
                    answers[question_idx] = future.result()

        convos = []
        for question, answer in zip(questions, answers):
            assert isinstance(answer, str)
            messages = [
                Message(content=question, role="user", sample=None),
                Message(content=answer, role="assistant", sample=None),
            ]
            convo = ContextConvo(
                messages=messages, type="anthropic_q_and_a", metadata={}
            )
            convos.append(convo)

        return convos
 