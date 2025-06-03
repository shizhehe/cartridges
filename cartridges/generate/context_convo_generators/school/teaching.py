import random

from capsules.generate.structs import Context
from capsules.generate.chunk import Chunker
from capsules.generate.context_convo_generators.base import (
    AnswerSystemPromptGenerator,
    ConvoGeneratorWithLLMQuestionAndAnswer,
    MetadataT,
    QuestionData,
    QuestionSystemPromptGenerator,
)


class QuestionSystemPromptForTeaching(QuestionSystemPromptGenerator):
    class Config(QuestionSystemPromptGenerator.Config):
        pass

    def __call__(self, context: Context, metadata: MetadataT) -> str:
        return f"""Your job is to help generate a curriculum for teaching a downstream system about a document.

This curriculum will be structured as a series of question/answer pairs.
You are responsible for generating a single question.
You do not need to generate the response or anything else: your job is to just create the question.

These questions can also be more like tasks (e.g write a summary, explain the relationship between two things), than simply factual answering.
But they can also be more traditional questions.

Be creative!

I am going to give you the document, and I want you to come up with a question about it.
The answer to this question should contain information about the document.
This question can be general and open-ended or more specific and targeted.

The title of the document is: {context.title}

Here is the content of the document, between <content> and </content> tags.

<content>
{context.to_string()}
</content>
"""


class AnswerSystemPromptForTeaching(AnswerSystemPromptGenerator):
    class Config(AnswerSystemPromptGenerator.Config):
        include_chunk: bool = False

    def __call__(self, context: Context, question_data: QuestionData):

        chunk_data = ""

        if self.config.include_chunk:
            if question_data.chunk is None:
                assert "chunk_metadata" in question_data.metadata
                chunk_metadata = question_data.metadata["chunk_metadata"]
                # TODO(ryan): this isn't super clean right now
                doc_idx = chunk_metadata["doc_idx"]
                start_idx = chunk_metadata["start_idx"]
                end_idx = chunk_metadata["end_idx"]

                chunk = context.documents[doc_idx].content[start_idx:end_idx]
            else:
                chunk = question_data.chunk

            chunk_data = f"""Here is the excerpt of the document that the question is focused on.
<excerpt>
{chunk}
</excerpt>
"""

        prompt = f"""Your job is to help generate a curriculum for teaching a downstream system about a document.

This curriculum will be structured as a series of question/response pairs.
You are going to be given a question, and your job is to generate the answer, based on the information in the document.

You are not responsible for generating the question or doing anything other than responding.
Information in your answer is what's going to be used to teach the downstream system, so be thoughtful about what you include.

When generating your answer, focus on the information in the provided document.
The answer to question will come from the document.

The title of the document is: {context.title}
The prompt will refer to this document in a unambiguous manner.

Here is the content of the document, between <content> and </content> tags.

<content>
{context.to_string()}
</content>

{chunk_data}

In your answer, include specific information from the document. You can include specific information like text directly from the document and specific facts.
I am going to provide you the question now. Please answer it well."""

        return prompt


class QuestionFromDoc(ConvoGeneratorWithLLMQuestionAndAnswer):
    class Config(ConvoGeneratorWithLLMQuestionAndAnswer.Config): ...

    def sample_generate_question_prompt(self):
        prompt = f"""Now, please generate a question about this document.

This question can be general or specific, and can include information from anywhere in the document.
It can also focus on relating different sections of the document to each other.

It is okay to ask for specific facts, but it also okay to ask general questions.
Be creative.

Ensure your question is fully self-contained by clearly stating that it pertains to this document. Do not use generic placeholders like 'the document' without the title.

Now, generate only the question, with no other text or explanation."""

        return prompt, {}, None


class QuestionFromDocAndChunk(ConvoGeneratorWithLLMQuestionAndAnswer):
    class Config(ConvoGeneratorWithLLMQuestionAndAnswer.Config):
        chunker: Chunker.Config

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.chunker: Chunker = config.chunker.instantiate()

    def sample_generate_question_prompt(self):
        chunk = self.chunker(self.context)

        prompt = f"""Now, please generate a question about this document.

This question can be general or specific. It should focus on the following excerpt of text:

<excerpt>
{chunk.chunk}
</excerpt>

The question should:
1. Focus primarily on information from this specific excerpt, or how it relates to the document
2. Test understanding of key details, facts, or concepts from the document or excerpt.
3. Either inquire about specific facts, ask for an entire section verbatim, or do something else entirely. 

It's okay if answering the question requires some context from elsewhere in the document, but the core focus should be on the content shown in this excerpt.

Now, generate only the question following these guidelines, with no additional text or explanation."""

        return prompt, {"chunk_metadata": chunk.metadata}, chunk.chunk
