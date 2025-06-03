import random
from capsules.clients.base import Client, ClientConfig
from capsules.generate.structs import Context
from capsules.generate.chunk import Chunker, DocumentChunker
from capsules.generate.context_convo_generators.base import (
    ConvoGeneratorWithLLMQuestionAndAnswer
   
)


PROMPT = """Please generate a question about the following excerpt of text:

{chunk}

The question should:
1. Focus primarily on information from this specific excerpt
2. Test understanding of key details or concepts from the excerpt
3. Be clear, specific, and ask for an answer that is unambiguous
4. Be written as a direct question (starting with words like What, How, Why, etc.)

It's okay if answering the question requires some context from elsewhere in the document, but the core focus should be on the content shown in this excerpt.

Generate only the question, with no other text or explanation."""





class SimpleQuestionFromChunk(ConvoGeneratorWithLLMQuestionAndAnswer):
    class Config(ConvoGeneratorWithLLMQuestionAndAnswer.Config):
        chunker: Chunker.Config

        # This will be used to format the question prompt. It must contain
        # the {chunk} variable. If not provided, the question without any
        # formatting will be used.
        question_prompt_template: str = PROMPT
    
    
    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.config = config
        self.chunker: Chunker = config.chunker.instantiate()

    def sample_generate_question_prompt(self):
        chunk = self.chunker(self.context)

        return (
            self.config.question_prompt_template.format(chunk=chunk.chunk),
            {"chunk_metadata": chunk.metadata},
            chunk.chunk
        )




