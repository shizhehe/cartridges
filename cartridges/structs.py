from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np

from cartridges.clients.base import TopLogprobs
from cartridges.utils import get_logger


logger = get_logger(__name__)


@dataclass
class TrainingExample:
    @dataclass
    class Message:
        content: str
        role: Literal["user", "assistant", "system"]
        token_ids: Optional[List[int]]
        
        top_logprobs: Optional[TopLogprobs] = None
        mask: Optional[np.ndarray] = None
    """
    Attrs:
        messages: conversation as openai text.
        metadata: arbitrary metadata
        type: type of this context convo
    """
    messages: list[TrainingExample.Message]
    system_prompt: str

    type: str
    metadata: dict



    def _repr_html_(self) -> str:
        import markdown

        html = """
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        <div class='context-convo p-4'>
        """
        for message in self.messages:
            if message.role == "user":
                role_class = "bg-blue-100 text-blue-800"
            else:
                role_class = "bg-green-100 text-green-800"
            role_display = f"<strong style='font-size: 1.5em;'>{message.role.capitalize()}</strong>"
            content_html = markdown.markdown(message.content)
            html += f"""
            <div class='p-2 my-2 rounded {role_class}'>
                {role_display} {content_html}
            </div>
            """
        html += "</div>"
        return html

    def to_html(self) -> str:
        return self._repr_html_()


