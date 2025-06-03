from typing import List, Optional
from pydantic import BaseModel

from capsules.context import TexDocument, TexChapter, TexSection
from capsules.generate.run import BaseContextConfig

from capsules.context import StructuredContext
from capsules.tasks.codehop.code_hop_synth import CodeHopSynthConfig, make_code_hop, serialize_file

class CodeHopStructuredContextConfig(BaseContextConfig):
    
    code_hop_config: CodeHopSynthConfig
    
    
    def instantiate(self) -> StructuredContext:
        code_hop = make_code_hop(self.code_hop_config)

        return Repo(
            files=[
                File(name=file.name, content=serialize_file(file))
                for file in code_hop.files
            ]
        )

class File(StructuredContext):
    name: str
    content: str

    @property
    def text(self) -> str:
        return f"# File: {self.name}.py\n\n{self.content}"


class Repo(StructuredContext):
    files: List[File]
    
    @property
    def text(self) -> str:
        return "\n\n---------\n\n".join([file.text for file in self.files])
    