

import os
from typing import List

from capsules.context import StructuredContext
from capsules.generate.structs import Context, Section
from capsules.generate.run import BaseContextConfig


class CudaFile(StructuredContext):
    desc: str
    content: str


class Cuda(StructuredContext):
    files: List[CudaFile]


def load_cuda_dataset() -> List[str]:

    filepath = "/home/simarora/code/cuda/"
    sections = []
    for file in os.listdir(filepath):
        with open(os.path.join(filepath, file), "r") as f:
            data = f.read()
            sections.append(
                CudaFile(
                    desc=file,
                    content=data
                )
            )

    return sections

class CUDAContextConfig(BaseContextConfig):
    mode: str = "train"

    def instantiate(self) -> Context:
        print(f"Loading CUDA context in {self.mode} mode")

        context = load_cuda_dataset()
        if self.mode == 'train':
            cuda = Cuda(files = context)
        elif self.mode == 'icl':
            full_context = []
            for file in context:
                full_context.append(file.desc)
                full_context.append(file.content)

            section = Section(
                desc="Information about CUDA",
                content="\n\n".join(full_context),
            )

            cuda = Context(
                title="Information about CUDA",
                sections=[section],
            )
        return cuda

