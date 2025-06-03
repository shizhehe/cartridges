import json
import random
import os
from capsules.clients.base import Client, ClientConfig
from capsules.generate.structs import Context
from capsules.generate.chunk import Chunker
from capsules.generate.context_convo_generators.base import (
    ConvoGeneratorWithLLMQuestionAndAnswer
   
)


def prompt(chunk: str, in_context_exes: str):
    return f"""Please generate a question about the following excerpt of text:

<begin excerpt>
{chunk}
<end excerpt>

The question should have each of the following attributes:
1. Specificity: Does the question ask for a specific output?
2. Excerpt-specific: Does the question focus primarily on information from this specific excerpt and test understanding of key concepts and details from this excerpt? 
3. Complexity: Does the question have multiple levels of reasoning, components, or variables?
4. Problem-Solving: Does the question directly involve the AI to demonstrate active problem-solving skills?
5. Creativity: Does the question involve a level of creativity in approaching the problem?
6. Technical Accuracy: Does the question require technical accuracy in the response?
7. Real-world Application: Does the question relate to real-world applications?

Here are some examples: {in_context_exes}

Generate a single question that matches the provided examples in terms of difficulty, creativity, and specificity, but ensure the question is directly tailored to the given document. The question should primarily focus on the content of this excerpt, though drawing on broader context from the document is acceptable when necessary. The question should be clear, engaging, and designed to test understanding of key details, themes, or concepts presented in the excerpt.
Generate only the question, with no other text or explanation."""


class ArenaHardInContextQuestion(ConvoGeneratorWithLLMQuestionAndAnswer):
    class Config(ConvoGeneratorWithLLMQuestionAndAnswer.Config):
        chunker: Chunker.Config
    
    
    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.chunker: Chunker = config.chunker.instantiate()
    
    def sample_arena_examples(self, n = 3): 
        questions = []
        arena_path = os.environ["CAPSULES_DIR"] + "/capsules/generate/context_convo_generators/arena_hard.jsonl"
        with open(arena_path, 'r', encoding='utf-8') as file:
            for line in file:
                obj = json.loads(line.strip())
                # limiting the prompt length to 200 characters
                if len(obj['turns'][0]['content']) < 200: 
                    questions.append(obj['turns'][0]['content'])

        chosen = random.sample(questions, n)
        text = ""
        for i in range(n):  
            text += f"\nExample {i+1}: {chosen[i]}\n"
        # print("In context examples: ", text)
        return text

    def sample_generate_question_prompt(self):
        chunk = self.chunker(self.context)
        in_context_exes = self.sample_arena_examples()

        return prompt(chunk.chunk, in_context_exes), {"chunk_metadata": chunk.metadata}, chunk.chunk
