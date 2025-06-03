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

The question should:

1. Focus primarily on information from this specific excerpt.
2. Test understanding of key details or concepts from the excerpt.
3. Encourage the responder to develop a poem, story, or other creative interpretation related to the concepts in the text.
4. Prompt a unique perspective or imaginative exploration that deepens understanding of the excerpt.
5. Avoid including analogies within the question itself—rather, it should invite the responder to create their own.
6. Go beyond simple factual recall or surface-level creativity to inspire meaningful exploration.

It's okay if answering the question requires some context from elsewhere in the document, but the core focus should be on the content shown in this excerpt.

Only generate the question itself, without any introductory phrases or explanations."""

"""

1. Focus primarily on information from this specific excerpt.
2. Encourage an original and imaginative response while staying relevant to the core concepts in the excerpt.
3. Involve emotional or artistic expression relating to the excerpt. 
4. Request unique perspectives or interpretative responses to the excerpt. 
5. Request writing that goes beyond factual reporting or analysis

1. Focus primarily on information from this specific excerpt.
2. Encourage an imaginative response while staying relevant to the core concepts in the text.
3. Prompt a unique perspective or interpretative response that deepens understanding of the topic.
4. Inspire connections between technical concepts and creative storytelling, metaphor, or analogy.
5. Go beyond simple factual recall or surface-level creativity to encourage meaningful exploration.

"""


class ArenaCreativeQuestionFromChunk(ConvoGeneratorWithLLMQuestionAndAnswer):
    class Config(ConvoGeneratorWithLLMQuestionAndAnswer.Config):
        chunker: Chunker.Config
    
    
    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.chunker: Chunker = config.chunker.instantiate()

    def sample_generate_question_prompt(self):
        document_idx = random.randint(0, len(self.context.documents) - 1)
        document = self.context.documents[document_idx]
        chunk = self.chunker(self.context)

        return prompt(chunk.chunk), {"chunk_metadata": chunk.metadata}, chunk.chunk
