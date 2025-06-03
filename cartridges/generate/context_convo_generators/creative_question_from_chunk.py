import random
from capsules.clients.base import Client, ClientConfig
from capsules.generate.structs import Context
from capsules.generate.chunk import Chunker
from capsules.generate.context_convo_generators.base import (
    ConvoGeneratorWithLLMQuestionAndAnswer
   
)


import random
from capsules.clients.base import Client, ClientConfig
from capsules.generate.structs import Context
from capsules.generate.chunk import Chunker, DocumentChunker
from capsules.generate.context_convo_generators.base import (
    ConvoGeneratorWithLLMQuestionAndAnswer
   
)

def prompt(chunk: str):
    return f"""Please generate a question about the following excerpt of text:

{chunk}

The question should:

1. Focus primarily on information from this specific excerpt.
2. Test understanding of key details or concepts from the excerpt.
3. Encourage a playful, creative response—such as explaining the concept through a short science fiction story, composing a limerick, or using an analogy from an unexpected domain (e.g., cooking, space travel, or wizardry).

It's okay if answering the question requires some context from elsewhere in the document, but the core focus should be on the content shown in this excerpt.

Generate only the question, with no additional text or explanation."""



class CreativeQuestionFromChunk(ConvoGeneratorWithLLMQuestionAndAnswer):
    class Config(ConvoGeneratorWithLLMQuestionAndAnswer.Config):
        chunker: Chunker.Config
    
    
    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.chunker: Chunker = config.chunker.instantiate()

    def sample_generate_question_prompt(self):
        chunk = self.chunker(self.context)

        return prompt(chunk=chunk.chunk), {"chunk_metadata": chunk.metadata}, chunk.chunk

