import random
from capsules.clients.base import Client, ClientConfig
from capsules.generate.structs import Context
from capsules.generate.chunk import Chunker, DocumentChunker
from capsules.generate.context_convo_generators.base import (
    AnswerSystemPromptGenerator,
    ConvoGeneratorWithLLMQuestionAndAnswer,
    QuestionData,
    QuestionSystemPromptGenerator
)


class QuestionSystemPromptForTesting(QuestionSystemPromptGenerator):
    class Config(QuestionSystemPromptGenerator.Config):
        pass

    def __call__(self, context: Context, metadata: dict) -> str:
        return f"""Your task is to evaluate a someone's understanding of a document by generating a challenging exam question.

This question will be used to test a someone's ability to comprehend and analyze the document's content.
This question can be a factual recall of specific details, but could also be a question that requires critical thinking, interpretation, or synthesis of the material.

The title of the document is: {context.title}

Below is the full content of the document, enclosed between <content> tags.

<content>
{context.to_string()}
</content>
"""


class AnswerSystemPromptForTesting(AnswerSystemPromptGenerator):
    class Config(AnswerSystemPromptGenerator.Config):
        include_chunk: bool = False

    def __call__(self, context: Context, question_data: QuestionData):
        chunk_data = ""

        if self.config.include_chunk:
            if question_data.chunk is None:
                assert "chunk_metadata" in question_data.metadata
                chunk_metadata = question_data.metadata["chunk_metadata"]
                doc_idx = chunk_metadata['doc_idx']
                start_idx = chunk_metadata['start_idx']
                end_idx = chunk_metadata['end_idx']
                chunk = context.documents[doc_idx].content[start_idx:end_idx]
            else:
                chunk = question_data.chunk

            chunk_data = f"""The question focuses on this excerpt from the document:
<excerpt>
{chunk}
</excerpt>
"""

        prompt = f"""Your role is to answer a question about the a document (that I will provide soon).

Make sure your response is firmly grounded in the content provided. If necessary, it should include thoughtful analysis, critical reasoning, and references to specific parts of the document.
If the question is just a factual recall, analysis is not necessary.

The document's title is: {context.title}
The complete document content is provided below, enclosed between <content> tags.

<content>
{context.to_string()}
</content>

{chunk_data}

Now, I will provide the question. Please respond with your answer to the question.
"""
        return prompt


class EvaluationFromDoc(ConvoGeneratorWithLLMQuestionAndAnswer):
    class Config(ConvoGeneratorWithLLMQuestionAndAnswer.Config):
        pass

    def sample_generate_question_prompt(self):
        prompt = f"""Now, please generate a testing question that assesses someone's understanding of the document.

This question could be a straightforward factual recall of specific details, or something more challenging that requires analysis, interpretation, or synthesis of the content.

Please provide only the question, with no additional commentary or explanation."""
        return prompt, {}, None


class EvaluationFromDocAndChunk(ConvoGeneratorWithLLMQuestionAndAnswer):
    class Config(ConvoGeneratorWithLLMQuestionAndAnswer.Config):
        chunker: Chunker.Config

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.chunker: Chunker = config.chunker.instantiate()

    def sample_generate_question_prompt(self):
        chunk = self.chunker(self.context)

        prompt = f"""Now, please generate a testing question that focuses on the following excerpt from the document:

<excerpt>
{chunk.chunk}
</excerpt>

The question should allow for straightforward factual recall of details from this excerpt, or encourage advanced analysis, interpretation, or synthesis of its content in relation to the document as a whole.

Please provide only the question, with no additional text or explanation."""
        return prompt, {"chunk_metadata": chunk.metadata}, chunk.chunk