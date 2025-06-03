from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union
from abc import ABC

from pydantic import BaseModel
from pydrantic import BaseConfig


if TYPE_CHECKING:
    from cartridges.structs import Context
    from bs4 import Tag

class BaseContextConfig(BaseConfig, ABC):
    """This should be subclassed by different tasks to specify the parameters
    and method for instantiating a Context object.
    For example, see LongHealthContextConfig in tasks/longhealth/__init__.py
    """
    def instantiate(self) -> Union[Context, StructuredContext]:
        raise NotImplementedError("Subclasses must implement this method")    


class StructuredContext(BaseModel, ABC):
    
    @property
    def text(self) -> str:
        """This is a special property that all context objects must implement. 
        It provides the default text representation of the context object.
        """
        return str(self)

    def to_string(self) -> str:
        """This is for backward compatability with the old context format."""
        return self.text
    
    def to_yaml(self, path: str):
        """This is for backward compatability with the old context usage as well."""
        import yaml

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f)
    

def list_nested_contexts(
    ctx: StructuredContext, 
    ctxs: Optional[List[StructuredContext]]=None, 
    path: str="/", 
    leaves_only: bool = False
) -> list[StructuredContext]:    
    ctxs = [] if ctxs is None else ctxs
    initial_len = len(ctxs)
    for field_name, field in ctx.model_fields.items():
        value = getattr(ctx, field_name)
        new_path = os.path.join(path, field_name)
        if isinstance(value, StructuredContext):
            list_nested_contexts(value, ctxs=ctxs, path=new_path)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                list_nested_contexts(item, ctxs=ctxs, path=os.path.join(new_path, f"{i}"))
        elif isinstance(value, dict):
            for k, v in value.items():
                list_nested_contexts(v, ctxs=ctxs, path=os.path.join(new_path, k))
    
    is_leaf = len(ctxs) == initial_len
    if not leaves_only or is_leaf:
        ctxs.append((path, ctx))

    return ctxs


class HTMLElement(StructuredContext):
    tag: str

    attributes: Optional[Dict[str, Any]] = None
    
    children: List["HTMLElement"]
    
    raw: str

    @property
    def text(self) -> str:
        return self.raw
    

    @classmethod
    def from_bs4(cls, soup: "Tag", max_depth: int = 0):
        if max_depth == 0:
            children = []
        else:
            children = [
                cls.from_bs4(child, max_depth - 1) for child in soup.contents
            ]

        return cls(
            tag=soup.name,
            attributes=soup.attrs,
            children=children,
            raw=str(soup),
        )

class HTMLDocument(StructuredContext):

    title: Optional[str] = None

    head: HTMLElement
    body: HTMLElement

    raw: str

    @property
    def text(self) -> str:
        return self.raw
    
    @classmethod
    def from_string(
        cls, 
        html: str,
        max_depth: int = 1
    ) -> "HTMLDocument":
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        return cls(
            raw=html,
            title=soup.title.text if soup.title else None,
            head=HTMLElement.from_bs4(soup.head),
            body=HTMLElement.from_bs4(soup.body),
        )
    

    
class TexChapter(StructuredContext):
    title: str
    chapter_idx: int
    preface: Optional[str] = None
    sections: list[TexSection]
    label: Optional[str] = None

    @property
    def text(self) -> str:
        return self.preface + "\n" + "\n".join([section.text for section in self.sections])

class TexSection(StructuredContext):
    title: str
    preface: Optional[str] = None
    subsections: list[TexSubSection]
    label: Optional[str] = None

    @property
    def text(self) -> str:
        return self.preface + "\n" + "\n".join([subsection.text for subsection in self.subsections])

class TexSubSection(StructuredContext):
    title: str
    preface: Optional[str] = None
    label: Optional[str] = None

    @property
    def text(self) -> str:
        return self.preface


class TexDocument(StructuredContext):
    chapters: list[TexChapter]
    preface: Optional[str] = None
    title: str
    author: str
    label: Optional[str] = None

    @property
    def text(self) -> str:
        return self.preface + "\n" + "\n".join([chapter.text for chapter in self.chapters])

    @classmethod
    def from_string(
        cls, 
        tex: str,
    ) -> "TexDocument":
        import re
        # Define regex patterns for chapters, sections, and subsections
        chapter_pattern = r'\\chapter\{((?:[^\{\}]+|\{(?:[^\{\}]+|\{[^\{\}]*\})*\})*)\}'
        section_pattern = r'\\section\{((?:[^\{\}]+|\{(?:[^\{\}]+|\{[^\{\}]*\})*\})*)\}'
        subsection_pattern = r'\\subsection\{((?:[^\{\}]+|\{(?:[^\{\}]+|\{[^\{\}]*\})*\})*)\}'

        # Split the tex file into chapters
        chapters = re.split(chapter_pattern, tex)

        # Initialize a list to hold the structured data
        structured_data = []

        def get_label(content: str) -> Optional[str]:
            # Use regex to find a label anywhere in a tex section or chapter
            label_match = re.search(r'\\label\{(.+?)\}', content)
            if label_match:
                return label_match.group(1)
            return None

        # Iterate over the chapters
        doc_preface = chapters.pop(0)
        for i in range(0, len(chapters) - 1 , 2):
            chapter_title = chapters[i]
            chapter_content = chapters[i + 1]
            
            # Split the chapter content into sections
            sections = re.split(section_pattern, chapter_content)

            # no label for the no_chapter section
            chapter_preface = sections.pop(0)
            chapter_label = get_label(chapter_preface)
            chapter_sections = []
            for j in range(0, len(sections) - 1, 2):
                section_title = sections[j]
                section_content = sections[j + 1]
                
                # Split the section content into subsections
                subsections = re.split(subsection_pattern, section_content)
                section_subsections = []
                section_preface = subsections.pop(0)
                section_label = get_label(section_preface)
                for k in range(0, len(subsections) - 1, 2):
                    subsection_title = subsections[k]
                    subsection_content = subsections[k + 1]
                    subsection_label = get_label(subsection_content) if k != 0 else None
                    section_subsections.append(TexSubSection(
                        title=subsection_title,
                        preface=subsection_content,
                        label=subsection_label
                    ))
                
                chapter_sections.append(TexSection(
                    title=section_title,
                    subsections=section_subsections,
                    label=section_label,
                    preface=section_preface
                ))
            
            structured_data.append(TexChapter(
                title=chapter_title,
                preface=chapter_preface,
                chapter_idx=i // 2,
                sections=chapter_sections,
                label=chapter_label
            ))
        return TexDocument(
            chapters=structured_data,
            title="Kalamang Grammar",
            preface=doc_preface,
            raw=tex,
            author="",
        )

