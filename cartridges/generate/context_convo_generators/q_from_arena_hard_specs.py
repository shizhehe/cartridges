import random
from capsules.clients.base import Client, ClientConfig
from capsules.generate.structs import Context
from capsules.generate.chunk import Chunker
from capsules.generate.context_convo_generators.base import (
    ConvoGeneratorWithLLMQuestionAndAnswer
   
)


def prompt(chunk: str):
    return f"""Please generate a question about the following excerpt of text:

{chunk}

The question should have each of the following attributes:
1. Specificity: Does the question ask for a specific output?
2. Excerpt-specific: Does the question focus primarily on information from this specific excerpt and test understanding of key concepts and details from this excerpt? 
3. Complexity: Does the question have multiple levels of reasoning, components, or variables?
4. Problem-Solving: Does the question directly involve the AI to demonstrate active problem-solving skills?
5. Creativity: Does the question involve a level of creativity in approaching the problem?
6. Technical Accuracy: Does the question require technical accuracy in the response?
7. Real-world Application: Does the question relate to real-world applications?

It's okay if answering the question requires some context from elsewhere in the document, but the core focus should be on the content shown in this excerpt.

Generate only the question, with no other text or explanation."""


class ArenaHardSpecsQuestion(ConvoGeneratorWithLLMQuestionAndAnswer):
    class Config(ConvoGeneratorWithLLMQuestionAndAnswer.Config):
        chunker: Chunker.Config
    
    
    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.chunker: Chunker = config.chunker.instantiate()

    def sample_generate_question_prompt(self):
        chunk = self.chunker(self.context)

        return prompt(chunk.chunk), {"chunk_metadata": chunk.metadata}
