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
    

