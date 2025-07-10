


import os
import random
from typing import List
from .base import Resource


def code_to_text(root_dir: str, num_to_sample: int) -> str:
    """ 
    We assume each code session is in its own directory
    A session may contain multiple files (code, profiling results, correctness results, compilation errors, etc.)
    """

    fpath = f"{root_dir}"
    dirs = os.listdir(fpath)

    dirs = random.sample(dirs, num_to_sample)
    contents = []

    for dir in dirs:
        files = os.listdir(f"{fpath}/{dir}")
        texts = {}
        for file in files:
            try:
                with open(f"{fpath}/{dir}/{file}", "r") as f:
                    texts[file] = f.read()
            except Exception as e:
                print(f"Error reading file {file}: {e}")
                continue
        
        content = "\n\n".join([f"{file}\n{text}" for file, text in texts.items()])

        contents.append(content)

    return contents


class CodeResource(Resource):

    class Config(Resource.Config):
        root_dir: str = "database/code"
        prompts: List[str] = []


    def __init__(self, config: Config):
        self.config = config
        self.prompts = CODE_PROMPTS

    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        text = code_to_text(root_dir=self.config.root_dir, num_to_sample=1)[0]
        return text, random.choices(self.prompts, k=batch_size)


CODE_PROMPTS = [
    (
        "You are analyzing the user's code attempt for writing a GEMM kernel in the ThunderKittens repository."
        "Please instruct an LLM to summarize how well this code performs (speed, compilation, correctness, etc.)."
    )
]



