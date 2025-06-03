import abc
import random
from typing import List, Optional, Union
from capsules.generate.structs import Context, Section, SectionedContext
from pydrantic import ObjectConfig
from transformers import AutoTokenizer

from capsules.generate.utils import unopionated_section_maker


class SubcontextGenerator(abc.ABC):

    class Config(ObjectConfig):
        _pass_as_config: bool = True

    def __init__(self, config: Config, context: SectionedContext):
        raise NotImplementedError()

    def __getitem__(self, idx: int) -> str:
        return self.subcontexts[idx]

    def __len__(self) -> int:
        return len(self.subcontexts)

SECTION_BUFFER_TOKENS = 50

def context_to_sections(
    context: Context, 
    tokenizer,
    max_tokens_per_section: int,
) -> List[Section]:

    sections = []
    current_pages: list[tuple[int, str]] = []
    current_tokens = 0

    # 1: to exclude BOS token
    count_tokens = lambda str_: len(tokenizer.encode(str_)) - 1

    def make_section():
        assert len(current_pages)

        content = "\n".join([page_content for (_, page_content) in current_pages])

        section = Section(
            content=content,
            tokens=0,
            desc=f"Pages {current_pages[0][0] + 1} through {current_pages[-1][0] + 1}",
        )
        section.tokens = count_tokens(str(section))
        if not section.tokens <= max_tokens_per_section:
            breakpoint()
        assert section.tokens <= max_tokens_per_section
        sections.append(section)
        current_pages.clear()
        nonlocal current_tokens
        current_tokens = 0

    pages = context.documents
    for page_idx, page in enumerate(pages):
        page_content = page.to_string()
        page_tokens = count_tokens(page_content)
        assert page_tokens < max_tokens_per_section

        if (
            current_tokens + page_tokens + SECTION_BUFFER_TOKENS
            > max_tokens_per_section
        ):
            make_section()

        current_pages.append((page_idx, page_content))
        current_tokens += page_tokens

    make_section()
    return sections


def content_to_sections(
    context: str, 
    tokenizer,
    max_tokens_per_section: int,
) -> List[Section]:

    sections = []
    current_pages: list[tuple[int, str]] = []
    current_tokens = 0

    # 1: to exclude BOS token
    count_tokens = lambda str_: len(tokenizer.encode(str_)) - 1

    def make_section():
        assert len(current_pages)

        content = "\n".join([page_content for (_, page_content) in current_pages])

        section = Section(
            content=content,
            tokens=0,
            desc=f"Pages {current_pages[0][0] + 1} through {current_pages[-1][0] + 1}",
        )
        section.tokens = count_tokens(str(section))
        if not section.tokens <= max_tokens_per_section:
            breakpoint()
        assert section.tokens <= max_tokens_per_section
        sections.append(section)
        current_pages.clear()
        nonlocal current_tokens
        current_tokens = 0

    page_content = context
    page_tokens = count_tokens(page_content)
    assert page_tokens < max_tokens_per_section
    if (
        current_tokens + page_tokens + SECTION_BUFFER_TOKENS
        > max_tokens_per_section
    ):
        make_section()
    current_pages.append((0, page_content))
    current_tokens += page_tokens

    make_section()
    return sections


class RandomizedSlidingWindowGenerator(SubcontextGenerator):
    """
    Generates subcontexts using a sliding window approach where both the
    window size and the step size are chosen randomly within specified bounds
    for each step. Allows multiple passes over the document chunks.

    It's tolerant of windows at the end being shorter than the minimum size
    if the remaining chunks are insufficient.
    """

    class Config(SubcontextGenerator.Config):
        tokenizer: str = "meta-llama/Llama-3.2-3B-Instruct"

        min_window_size: int
        max_window_size: int

        max_tokens_per_section: Optional[int] = None

        min_step_size: int
        max_step_size: int

        num_passes: int
        seed: int

    def __init__(self, config: Config, context: Union[SectionedContext, Context]):
        self.config = config
        assert isinstance(self.config, RandomizedSlidingWindowGenerator.Config)

        assert 0 < self.config.min_window_size <= self.config.max_window_size
        assert 0 < self.config.min_step_size <= self.config.max_step_size
        assert 1 <= self.config.num_passes

        random.seed(self.config.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)

        subcontexts: list[str] = []

        if isinstance(context, SectionedContext):
            sections = context.sections
        else:
            assert self.config.max_tokens_per_section is not None
            breakpoint()
            sections = unopionated_section_maker(
                [line for page in context.documents for line in page.content.split("\n")], 
                title=context.title,
                tokenizer=self.tokenizer, 
                max_tokens_per_section=self.config.max_tokens_per_section
            )

        num_sections = len(sections)

        if num_sections == 0:
            return []  # No chunks to process

        for _ in range(self.config.num_passes):
            current_section_index = 0
            while current_section_index < num_sections:
                current_window_size = random.randint(
                    self.config.min_window_size, self.config.max_window_size
                )

                section_end = min(
                    num_sections, current_section_index + current_window_size
                )

                assert section_end > current_section_index

                next_subcontext_sections = sections[
                    current_section_index:section_end
                ]
                subcontexts.append(
                    "\n".join([str(section) for section in next_subcontext_sections])
                )

                current_step_size = random.randint( 
                    self.config.min_step_size, self.config.max_step_size
                )

                current_section_index += current_step_size

        self.subcontexts = subcontexts
        print(f"{len(subcontexts)} subcontexts")



class RandomSubcontextGenerator(SubcontextGenerator):
    class Config(SubcontextGenerator.Config):
        tokenizer: str = "meta-llama/Llama-3.2-3B-Instruct"

        min_num_chunks: int
        max_num_chunks: int
        num_contexts: int

        separator: str = "\n"

        seed: int

    def __init__(self, config: Config, context: Union[SectionedContext, Context]):
        self.config = config

        assert isinstance(self.config, RandomSubcontextGenerator.Config)

        random.seed(self.config.seed)
        if isinstance(context, SectionedContext):
            sections = [str(section) for section in context.sections]
        else:
            sections = [document.to_string() for document in context.sections]
        self.sections = sections

    def __getitem__(self, idx: int) -> str:
        return self.config.separator.join(
            random.choices(
                self.sections,
                k=random.randint(
                    self.config.min_num_chunks,
                    self.config.max_num_chunks,
                ),
            )
        )
    
    def __len__(self) -> int:
        return self.config.num_contexts


# TODO (SE): Fix the below


# def format_page_xml(page: Page, page_index: int) -> str:
#     """
#     Formats a single Page object into an XML string representation.

#     Args:
#         page: The Page object containing the content.
#         page_index: The original index number of the page in the document.

#     Returns:
#         An XML string representing the page.
#     """
#     page_str = f"""\
# <page number="{page_index}">
# {page.content}
# </page>"""
#     return page_str


# def describe_subcontext(context: list[Chunk]) -> str:
#     """
#     Formats a list of Chunks into an XML string representation suitable for an LLM,
#     using assertions and escaping content instead of CDATA. Uses a helper
#     function to format individual pages.

#     Args:
#         context: A list of Chunk objects representing parts of a document.

#     Returns:
#         A string formatted with XML tags describing the document context.
#     """
#     if not context:
#         raise ValueError("Context list cannot be empty.")

#     first_chunk = context[0]
#     document_title = first_chunk.document.title
#     document_content = first_chunk.document  # Assumes PagedDocument is accessible

#     chunk_strings = []

#     for i, chunk in enumerate(context):
#         # Ensure all chunks belong to the same document.
#         assert chunk.document.title == document_title, (
#             f"Chunk {i} belongs to a different document ('{chunk.document.title}') "
#             f"than the first chunk ('{document_title}'). Context must contain chunks from a single document."
#         )

#         pages = "\n".join(
#             [
#                 format_page_xml(document_content.pages[page_index], page_index)
#                 for page_index in range(chunk.start_page, chunk.end_page + 1)
#             ]
#         )

#         # Use f-string for chunk structure
#         chunk_str = f"""\
# <chunk index="{i}" start_page="{chunk.start_page}" end_page="{chunk.end_page}">
# {pages}
# </chunk>"""
#         chunk_strings.append(chunk_str)

#     # Use f-string for the final document structure
#     return f"""\
# <document title="{document_title}">
#   <context>
# {'\n'.join(chunk_strings)}
#   </context>
# </document>"""


SUBCONTEXT_INSTRUCTIONS = """You'll an excerpt from the document formatted in XML. Here's how to understand the structure:

1.  `<document title="...">`: The main container. The `title` attribute tells you the name of the source document.

2.  `<context>`: Inside `<document>`, this tag groups the specific document sections (chunks) provided to you.

3.  `<chunk index="..." start_page="..." end_page="...">`:
    Found inside `<context>`. Each `<chunk>` represents a continuous block of pages from the original document.

    Important: The chunks themselves might be presented out of their original document order within the `<context>`.

    Attributes Explained:
    - `index`: The sequence number of this chunk as it appears in this specific XML context (0, 1, 2...). This index does not guarantee the chunk represents earlier pages than a chunk with a higher index. For example, chunk `index="0"` could contain pages 50-60, while chunk `index="1"` contains pages 10-20.
    - `start_page`: The page number where this chunk begins in the original document. Use this and `end_page` to know the actual position in the source.
    - `end_page`: The page number where this chunk ends in the original document.

4.  `<page number="...">`:
    Found inside each `<chunk>`. This tag represents one specific page from the range defined by its parent chunk.

    Attribute Explained:
    - `number`: The actual page number of this page from the original document. This number always reflects the true page order.

The actual text from the document pages is located only inside the `<page number="...">` tags.
The text between `<page number="X">` and `</page>` is the literal content from page `X`.
Outer tags (`<document>`, `<context>`, `<chunk>`) define the structure and identify which pages are included; they do not contain the page text themselves. Rely on `start_page`, `end_page`, and `page number` for ordering information relative to the original document.

Example:


<document title="MyReport">
  <context>
    <chunk index="0" start_page="50" end_page="51"> <page number="50">
        Content of page fifty.
      </page>
      <page number="51">
        Content of page fifty-one.
      </page>
    </chunk>
    <chunk index="1" start_page="5" end_page="6">   <page number="5">
        Content of page five.
      </page>
      <page number="6">
        Content of page six.
      </page>
    </chunk>
  </context>
</document>
"""
