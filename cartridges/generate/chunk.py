import abc
from collections import defaultdict
from pathlib import Path
import random
import re
from typing import Optional
from dataclasses import dataclass

from transformers import AutoTokenizer
from pydrantic import ObjectConfig

from capsules.generate.structs import Context, Document


# Todo: eventually use generics properly
# @dataclass
# class Chunk[T]:
#     chunk_metadata: T
#     chunk: str


@dataclass
class Chunk(abc.ABC):
    metadata: dict
    chunk: str


class Chunker(abc.ABC):
    class Config(ObjectConfig):
        _pass_as_config: bool = True

    @abc.abstractmethod
    def __call__(self, context: Context) -> Chunk:
        raise NotImplementedError()


class SimpleCharacterChunker(Chunker):

    class Config(Chunker.Config):
        min_chunk_size_in_chars: Optional[int] = None
        max_chunk_size_in_chars: Optional[int] = None

    def __init__(self, config: Config):
        self.config = config

        assert (
            config.min_chunk_size_in_chars is not None
            or config.max_chunk_size_in_chars is not None
        )
        if (
            config.min_chunk_size_in_chars is not None
            and config.max_chunk_size_in_chars is not None
        ):
            assert config.min_chunk_size_in_chars <= config.max_chunk_size_in_chars

    def __call__(self, context: Context) -> Chunk:
        document_idx = random.randint(0, len(context.documents) - 1)
        document = context.documents[document_idx]

        min_chunk_size = (
            0
            if self.config.min_chunk_size_in_chars is None
            else self.config.min_chunk_size_in_chars
        )
        max_chunk_size = (
            len(document)
            if self.config.max_chunk_size_in_chars is None
            else self.config.max_chunk_size_in_chars
        )

        # dont run over the end
        max_chunk_size = min(len(document.content), max_chunk_size)
        min_chunk_size = min(len(document.content), min_chunk_size)

        chunk_size = random.randint(
            min_chunk_size,
            max_chunk_size,
        )

        start_idx = random.randint(0, len(document.content) - chunk_size)
        end_idx = start_idx + chunk_size

        return Chunk(
            chunk=document.content[start_idx:end_idx],
            metadata={
                "start_idx": start_idx,
                "end_idx": end_idx,
                "doc_idx": document_idx,
            },
        )

class SimpleTokenChunker(Chunker):

    class Config(Chunker.Config):
        min_chunk_size_in_tokens: Optional[int] = None
        max_chunk_size_in_tokens: Optional[int] = None

        tokenizer: str
    
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)


        assert (
            config.min_chunk_size_in_tokens is not None
            or config.max_chunk_size_in_tokens is not None
        )
        if (
            config.min_chunk_size_in_tokens is not None
            and config.max_chunk_size_in_tokens is not None
        ):
            assert config.min_chunk_size_in_tokens <= config.max_chunk_size_in_tokens


    
    def __call__(self, context: Context) -> Chunk:
        raise NotImplementedError()

class DocumentChunker(Chunker):

    class Config(Chunker.Config):
        section_delimiter: Optional[str] = None
        min_sections_per_chunk: int = 4
        max_documents_per_chunk: int = 1

    def __init__(self, config: Config):
        self.config = config

    def __call__(self, context: Context) -> Chunk:
        num_documents = random.randint(1, self.config.max_documents_per_chunk)
        chunk = []
        for _ in range(num_documents):
            document_idx = random.randint(0, len(context.documents) - 1)
            document = context.documents[document_idx]

            if self.config.section_delimiter is not None:
                sections = re.split(self.config.section_delimiter, document.content)
                num_sections = random.randint(
                    self.config.min_sections_per_chunk, len(sections)
                )
                start_idx = random.randint(0, len(sections) - num_sections)
                prefix = "<" if start_idx > 0 else ""
                suffix = "..." if start_idx + num_sections < len(sections) else ""
                content = self.config.section_delimiter.join(
                    sections[start_idx : start_idx + num_sections]
                )
                document = Document(
                    title=document.title,
                    content=prefix + content + suffix,
                    path=document.path,
                )

            chunk.append(document.to_string())

        return Chunk(
            chunk="\n---\n".join(chunk),
            metadata={
                # FIXME: add metadata
                # "start_idx": start_idx,
                # "end_idx": end_idx,
                # "doc_idx": document_idx,
            },
        )


class LeveledDocumentChunker(Chunker):

    class Config(Chunker.Config):
        level_to_weight: Optional[dict[int, float]] = None
        max_tokens: Optional[int] = None
        tokenizer: Optional[str] = None

    def __init__(self, config: Config):
        self.config = config
        if self.config.tokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
        else:
            self.tokenizer = None
        if self.config.max_tokens is not None:
            assert self.tokenizer is not None

    def __call__(self, context: Context) -> Chunk:
        level_to_documents = defaultdict(list)
        for document in context.documents:
            level_to_documents[len(document.path.split("/"))].append(document)
        levels = list(level_to_documents.keys())

        if self.config.level_to_weight is not None:
            weights = [self.config.level_to_weight.get(level, 0) for level in levels]
            level = random.choices(levels, weights=weights, k=1)[0]
        else:
            level = random.choice(levels)

        documents = level_to_documents[level]
        document: Document = random.choice(documents)
        text = document.to_string()

        if self.config.max_tokens is not None:
            text = self.tokenizer.decode(
                self.tokenizer.encode(
                    text, max_length=self.config.max_tokens, truncation=True
                )
            )  # suppress maximum sequence length warning

        text = f"<title>{document.title}</title>\n<document>{text}</document>"

        return Chunk(
            chunk=text,
            metadata={
                # FIXME: add metadata
                # "start_idx": start_idx,
                # "end_idx": end_idx,
                # "doc_idx": document_idx,
            },
        )
