import math
from typing import Optional

from transformers import AutoTokenizer

from capsules.generate.chunk import Chunk
from capsules.generate.context_convo_generators.base import (
    AnswerSystemPromptGenerator,
    MetadataT,
    QuestionData,
    QuestionSystemPromptGenerator,
)
from capsules.generate.structs import Context


class QuestionSystemPrompt(QuestionSystemPromptGenerator):

    class Config(QuestionSystemPromptGenerator.Config):
        max_tokens: Optional[int] = None
        tokenizer: Optional[str] = None

        # The prompt template can contain {title} and {context} variables.
        prompt_template: str

    def __init__(self, config: Config):
        self.config = config

        if self.config.max_tokens is not None:
            assert self.config.tokenizer is not None
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
        else:
            self.tokenizer = None

    def __call__(self, context: Context, metadata: MetadataT) -> str:
        context_str = context.to_string()
        if self.config.max_tokens is not None:
            assert self.tokenizer is not None
            context_str = self.tokenizer.decode(
                self.tokenizer.encode(
                    context_str, max_length=self.config.max_tokens, truncation=True
                ),  # suppress maximum sequence length warning
            )

        return self.config.prompt_template.format(
            title=context.title, context=context_str,
        )


class AnswerSystemPrompt(AnswerSystemPromptGenerator):
    class Config(AnswerSystemPromptGenerator.Config):
        max_tokens: Optional[int] = None
        tokenizer: Optional[str] = None

        # The prompt template can contain {chunk} and {context} variables.
        prompt_template: str

    def __init__(self, config: Config):
        self.config = config

        if self.config.max_tokens is not None:
            assert self.config.tokenizer is not None
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
        else:
            self.tokenizer = None

    def __call__(self, context: Context, question_data: QuestionData) -> str:
        context_str = context.to_string()
        if self.config.max_tokens is not None:
            assert self.tokenizer is not None
            context_str = self.tokenizer.decode(
                self.tokenizer.encode(
                    context_str, max_length=self.config.max_tokens, truncation=True
                ),  # suppress maximum sequence length warning
            )

        return self.config.prompt_template.format(
            title=context.title, context=context.to_string(), chunk=question_data.chunk
        )


class QuestionSystemPromptWithEntireContext(QuestionSystemPromptGenerator):
    class Config(QuestionSystemPromptGenerator.Config):
        pass

    def __call__(self, context: Context, metadata: MetadataT) -> str:
        return f"""Here is a document that may be useful to the user.
Title: {context.title}

Here is the content of the document, between <content> and </content> tags.
<content>
{context.to_string()}
</content>
        """


class QuestionSystemPromptWithDoc(QuestionSystemPromptGenerator):
    class Config(QuestionSystemPromptGenerator.Config):
        pass

    def __call__(self, context: Context, metadata: MetadataT) -> str:
        doc = context.documents[metadata["chunk_metadata"]["doc_idx"]]
        return f"""Here is a document that may be useful to the user.
It's part of a larger collection of documents called: {context.title}

Here is the content of the document:
{doc.to_string()}
        """


class QuestionSystemPromptWithTitle(QuestionSystemPromptGenerator):
    class Config(QuestionSystemPromptGenerator.Config):
        pass

    def __call__(self, context: Context, metadata: MetadataT) -> str:
        return f"""Please generate a question grounded in a snippet from a document titled {context.title}"""


class AnswerSystemPromptWithEntireContext(AnswerSystemPromptGenerator):
    class Config(AnswerSystemPromptGenerator.Config):
        pass

    def __call__(self, context: Context, metadata: QuestionData) -> str:
        return f"""You are an expert at answering questions about the following document.
 
Title: {context.title}

Here is the content of the document, between <content> and </content> tags.
<content>
{context.to_string()}
</content>
"""


class AnswerSystemPromptWithDoc(AnswerSystemPromptGenerator):
    class Config(AnswerSystemPromptGenerator.Config):
        pass

    def __call__(self, context: Context, metadata: QuestionData) -> str:
        doc = context.documents[metadata.metadata["chunk_metadata"]["doc_idx"]]
        return f"""Here is a document that may be useful to the user.
It's part of a larger collection of documents called: {context.title}

Here is the content of the document:
{doc.to_string()}
"""


class AnswerSystemPromptWithChunk(AnswerSystemPromptGenerator):
    class Config(AnswerSystemPromptGenerator.Config):
        pass

    def __call__(self, context: Context, question_data: QuestionData) -> str:
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

        return f"""You are an expert at answering questions about the following context.
 
<context>
{chunk}
</context>
"""


ELIDED = "... [content elided] ..."


class AnswerSystemPromptWithEntireContextTruncated(AnswerSystemPromptGenerator):
    class Config(AnswerSystemPromptGenerator.Config):
        # TODO(RE): this could be tokens, but characters is probably fine for now
        max_chars: int

    def __call__(self, context: Context, metadata: QuestionData) -> str:
        context_str = context.to_string()

        if len(context_str) > self.config.max_chars:
            context_str = (
                context_str[: self.config.max_chars // 2]
                + ELIDED
                + context_str[-self.config.max_chars // 2 :]
            )

        return f"""Please use the information in the following document to answer the user's questions.
 
Title: {context.title}

Here is the content of the document, between <content> and </content> tags.
<content>
{context_str}
</content>
"""
