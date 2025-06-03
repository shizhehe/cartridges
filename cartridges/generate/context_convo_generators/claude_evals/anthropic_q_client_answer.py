from capsules.clients.anthropic import AnthropicClient
from capsules.generate.context_convo_generators.base import (
    ConvoGeneratorWithLLMAnswer,
    QuestionData,
)
from capsules.utils.parsing import extract_last_json


QUESTION_INSTRUCTION_PROMPT = f"""Generate exactly {{samples}} distinct questions about this document. Here are instructions for each question:
<instructions>
{{instructions}}
</instructions>

Think first and brainstorm options. Output a single json block (between triple backticks - ```) that contains a string list (the elements being the questions)
"""

class AnthropicQClientAnswer(ConvoGeneratorWithLLMAnswer):
    class Config(ConvoGeneratorWithLLMAnswer.Config):
        system_prompt_template: str
        instructions: str
        client: AnthropicClient.Config
        temperature: float

    def get_questions(self, num_samples: int) -> list[QuestionData]:
        
        prompt_system = self.config.system_prompt_template.format(
            title=self.context.title, 
            content=self.context.text, 
        )

        prompt_instruction = QUESTION_INSTRUCTION_PROMPT.format(
            instructions=self.config.instructions,
            samples=num_samples,
        )

        client = self.config.client.instantiate()
        response = client.chat(
            chats=[
                [
                    {
                        "type": "text", 
                        "content": prompt_system, 
                        "role": "system",
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "content": prompt_instruction, 
                        "role": "user"
                    }
                ]
            ], 
            temperature=self.config.temperature
        )

        content = response.samples[0].text
        questions_raw = extract_last_json(content)

        try:    
            assert isinstance(questions_raw, list)
        except:
            breakpoint()
        # assert len(questions_raw) == num_samples

        questions = []
        for q in questions_raw:
            assert isinstance(q, str)
            questions.append(
                QuestionData(question=q, sample=None, metadata={}, chunk=None)
            )

        return questions

