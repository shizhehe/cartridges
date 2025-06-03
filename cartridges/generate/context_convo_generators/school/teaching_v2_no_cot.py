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
You do not need to generate the answer or anything else: your job is to just create the question.
I am going to give you the document, and I want you to come up with a question about it.

The question can also be more like a task (e.g write a summary, explain the relationship between two things), than just simply factual answering.
But it can also could be a more traditional questions, like recalling a passage or simple fact answering. Be creative!

The answer to this question should contain information about the document.
This question can be general and open-ended or more specific and targeted.

In your question, it's important to refer to the document and section of the document unambiguously (e.g by its title).
Your questions could be mixed with questions from a number of other sources about other documents, so be sure that its clear what document you are referring to.

The title of the document is: {context.title}

Here is the content of the document, between <content> and </content> tags.

<content>
{context.to_string()}
</content>"""


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

        prompt = f"""Here is a document that may be helpful in answering questions that the user has.

The title of the document is: {context.title} The prompt will refer to this document in a unambiguous manner.
Here is the content of the document, between <content> and </content> tags.

<content>
{context.to_string()}
</content>

{chunk_data}

In your answer, include specific information from the document. You can include specific information like text directly from the document.
I am going to provide you the question now. Please answer it well."""

        return prompt


class QuestionFromDoc(ConvoGeneratorWithLLMQuestionAndAnswer):
    class Config(ConvoGeneratorWithLLMQuestionAndAnswer.Config): ...

    def sample_generate_question_prompt(self):
        prompt = f"""Now, please generate a question about this document.

This question can be general or specific. It should focus on the document as a whole.

The answerer to this question will have access to the entire document.

You can ask summative questions. You can also ask direct factual recall questions or questions that require integrating information from multiple parts of the document.

The question should:
1. Focus primarily on information from the document as a whole
2. Test understanding of key details, facts, concepts, or themes from the document
3. Either inquire about specific facts, ask for summaries, comparisons, analyses, or other higher-level thinking about the document

Remember, it is very important to specify which document you're asking questions about in your question.
The title of the document is: {self.context.title}.

Now, generate only the question following these guidelines, with no additional text or explanation."""

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

The answerer to this question will not know which excerpt this is. If its necessary to answer the question, explain where in the text it is.

For longer excerpts, (eg paragraphs or more), you can ask more summative questions. You can also ask direct factual recall questions for these larger excerpts.

The question should:
1. Focus primarily on information from this specific excerpt, or how it relates to the document
2. Test understanding of key details, facts, or concepts from the document or excerpt.
3. Either inquire about specific facts, ask for an entire section verbatim, or do something else entirely. 

It's okay if answering the question requires some context from elsewhere in the document, but the core focus should be on the content shown in this excerpt.

Remember, it is very important to specify which document you're asking questions about in your question.
The title of the document is: {self.context.title}.

Now, generate only the question following these guidelines, with no additional text or explanation."""

        return prompt, {"chunk_metadata": chunk.metadata}, chunk.chunk
