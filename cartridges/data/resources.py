from __future__ import annotations
from abc import ABC, abstractmethod
import abc
import asyncio
import os
import random
import time
from typing import Any, Dict, List, Optional, Literal, Callable
from pydantic import BaseModel
from pydrantic import ObjectConfig
import aiofiles

from cartridges.data.chunkers import Chunker
from cartridges.utils import get_logger

logger = get_logger(__name__)

class Resource(abc.ABC):

    class Config(ObjectConfig):
        _pass_as_config = True
    
    async def setup(self):
        """This is an optional method that can be used to setup the resource.
        It is called before the first call to sample_prompt.
        """
        pass
    
    @abc.abstractmethod
    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        raise NotImplementedError()
    
    
    def to_string(self) -> str:
        raise NotImplementedError("This resource does not implement a string representation.")

SEED_TYPES = Literal[
    "structuring", "summarization", "question", "use_case", "creative", 'generic'
] | Callable[[Any], str]


class TextResource(Resource):
    
    class Config(Resource.Config):
        text: str
        chunker: Chunker.Config

        seed_prompts: List[SEED_TYPES]

    def __init__(self, config: Config):
        self.config = config
        self.text = self.config.text
        self.chunker = None
    
    async def setup(self):
        self.chunker = self.config.chunker.instantiate(text=self.text)
    
    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        if self.chunker is None:
            raise ValueError("Chunker not initialized. Call setup() first.")
        
        chunk = self.chunker.sample_chunk()
        seed_prompts = sample_seed_prompts(self.config.seed_prompts, batch_size)
        return chunk, seed_prompts

class TextFileResource(TextResource):

    class Config(Resource.Config):
        path: str
        seed_prompts: List[SEED_TYPES]
        chunker: Chunker.Config

    def __init__(self, config: Config):
        self.config = config
        self.chunker = None
    
    async def setup(self):
        self.text = open(self.config.path).read()
        await super().setup()

async def load_directory_files(path: str, included_extensions: List[str], recursive: bool=True):
    # Get all files based on recursive setting
    if recursive:
        file_paths = []
        for root, dirs, files in os.walk(path):
            for file_name in files:
                if any(file_name.endswith(ext) for ext in included_extensions):
                    file_paths.append(os.path.join(root, file_name))
    else:
        all_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        file_paths = [
            os.path.join(path, f) for f in all_files 
            if any(f.endswith(ext) for ext in included_extensions)
        ]
    
    # Load all files in parallel
    async def load_single_file(file_path: str) -> tuple[str, str]:
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
        except UnicodeDecodeError:
            # If file can't be decoded as UTF-8, try with latin-1 or skip
            try:
                async with aiofiles.open(file_path, 'r', encoding='latin-1') as f:
                    content = await f.read()
            except Exception:
                content = f"[Unable to read file {os.path.basename(file_path)}]"
        
        # Use relative path from base directory as key for consistent naming
        rel_path = os.path.relpath(file_path, path)
        return rel_path, content
    
    # Load all files concurrently
    tasks = [load_single_file(file_path) for file_path in file_paths]
    results = await asyncio.gather(*tasks)
    
    return dict(results)


class DirectoryResource(Resource):

    class Config(Resource.Config):
        path: str
        included_extensions: List[str] = [".py", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".xml"]
        recursive: bool = False
        
        chunker: Optional[Chunker.Config] = None
        
        seed_prompts: List[SEED_TYPES]

    def __init__(self, config: Config):
        self.config = config
        self.file_contents: Dict[str, str | Chunker] = {}
    
    async def _load_files(self):
        # Get all files based on recursive setting
        return await load_directory_files(
            path=self.config.path,
            included_extensions=self.config.included_extensions,
            recursive=self.config.recursive
        )

    async def setup(self):
        t0 = time.time()
        # Preload all file contents
        self.file_contents = await self._load_files()
        for file_name, content in self.file_contents.items():
            if self.config.chunker is not None:
                self.file_contents[file_name] = self.config.chunker.instantiate(text=content)
            else:
                self.file_contents[file_name] = content
        logger.info(f"Loaded {len(self.file_contents)} files from {self.config.path} in {time.time() - t0:.2f} seconds")
        

    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        if not self.file_contents:
            raise ValueError("No files found in directory. Make sure to call setup() first and check that the directory contains files with the specified extensions.")
        
        if len(self.file_contents) == 0:
            raise ValueError("No files found in directory.")

        # Select a random file
        selected_file = random.choice(list(self.file_contents.keys()))
        
        # Get preloaded content
        content = self.file_contents[selected_file]

        if isinstance(content, Chunker):
            content = content.sample_chunk()
        
        # Create context with file information
        context = self._format_file(selected_file, content)
        
        # Generate seed prompts
        seed_prompts = sample_seed_prompts(self.config.seed_prompts, batch_size)
        
        return context, seed_prompts
    
    def _format_file(self, file_name: str, content: str) -> str:
        if file_name.endswith(".py"):
            return f"File: {file_name}\n\n```python\n{content}\n```"
        else:
            return f"File: {file_name}\n\n{content}"
    
    def to_string(self) -> str:
        if not self.file_contents:
            raise ValueError("Make sure to call setup() first and check that the directory contains files with the specified extensions.")
        
        if len(self.file_contents) == 0:
            raise ValueError("No files found in directory.")
        
        formatted_files = []
        for file_name, content in self.file_contents.items():
            if isinstance(content, Chunker):
                # For chunkers, we'll use the full text representation
                content_str = content.to_string() if hasattr(content, 'to_string') else str(content)
            else:
                content_str = content
            formatted_files.append(self._format_file(file_name, content_str))
        
        return "\n---end of file---\n".join(formatted_files)

class BaseStructuredResource(Resource, ABC):
    """This base class is to be used for resources that can be structured as a nested 
    object containing lists and dictionaries (e.g. JSON objects).
    """

    class Config(Resource.Config):
        seed_prompts: List[SEED_TYPES]
        leaves_only: bool = False
    
    def __init__(self, config: Config):
        self.config = config
        self.data = self._load_data()
        self.ctxs = self._list_nested_data(self.data)
    
    @abc.abstractmethod
    def _load_data(self) -> Any:
        raise NotImplementedError()
    
    def _list_nested_data(self, data: Any, _path: str = "") -> List[(str, str)]:
        """This function creates a string representation of 
        
        Return:
            (path, representation) where path is a string of the path within the object
            to the representation (e.g. "abc/0/def/1") and the string representation of the data. 
        """
        result = []
        
        if isinstance(data, dict):
            # Include the dict itself if not leaves_only
            if not self.config.leaves_only:
                result.append((_path, str(data)))
            
            for key, value in data.items():
                new_path = f"{_path}/{key}" if _path else key
                if isinstance(value, (dict, list)):
                    result.extend(self._list_nested_data(value, new_path))
                else:
                    result.append((new_path, str(value)))
        elif isinstance(data, list):
            # Include the list itself if not leaves_only
            if not self.config.leaves_only:
                result.append((_path, str(data)))
            
            for i, item in enumerate(data):
                new_path = f"{_path}/{i}" if _path else str(i)
                if isinstance(item, (dict, list)):
                    result.extend(self._list_nested_data(item, new_path))
                else:
                    result.append((new_path, str(item)))
        else:
            # For non-dict, non-list data, return the current path and string representation
            result.append((_path, str(data)))
        
        return result
        
    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        path, obj_str = random.choice(self.ctxs)
        ctx = f"The following is located at {path}: {obj_str}"
        seed_prompts = sample_seed_prompts(self.config.seed_prompts, batch_size)
        return ctx, seed_prompts

class JSONResource(BaseStructuredResource):
    class Config(BaseStructuredResource.Config):
        path: str
    
    def _load_data(self):
        import json
        return json.load(open(self.config.path))

# --- begin seed prompt generators  ---

def structuring_seed_prompt(**kwargs):
    DATA_FORMATS = [
        "JSON",
        "YAML",
        "TOML",
        "INI",
        "XML",
        "plain text",
    ]

    data_format = random.choice(DATA_FORMATS)

    EXAMPLES = [
        (
            "Can you structure the information in {{subsection}} of {{document}} related to {{something specific}} "
            f"in the following format: {data_format}? "
            "Be sure to include precise information like any dates, times, names, and numerical values.'"
        ),
        (
            "Can you structure the information in {{subsection}} of {{document}} "
            f"in the following format: {data_format}? "
            "Be sure to include precise information like any dates, times, names, and numerical values.'"
        ),
    ]

    example = random.choice(EXAMPLES)

    return (
        f"Please generate a single chat message instructing an LLM to structure the information in {data_format}. "
        "Output only the chat message itself and absolutely nothing else. "
        "Make sure it is clear what section and document you are asking about. "
        f"The message can follow the following template, filling in details from the corpus: \n\n'{example}'"
    )


def summarization_seed_prompt(**kwargs):
    prompts = [
        (
            "Please generate a single chat message instructing an LLM to summarize part of the corpus. "
            "Make sure the instruction is very explicit about the section of the corpus that you want to summarize. "
            "Include details (ids, names, titles, dates, etc.) that make it clear what you are asking about. "
        ),
        (
            "Please generate a single chat message instructing an LLM to summarize a section. "
            "Make sure the instruction is explicit about the section that should be summarized and the document it is from."
        ),
    ]
    prompt = random.choice(prompts)
    return prompt


def question_seed_prompt(**kwargs):
    prompts = [
        (
            "Generate a question for an LLM that will test its knowledge of the information in the corpus above. "
            "In your question be sure to include details (ids, names, titles, dates, etc.) that make it clear what you are asking about. "
            "Output only a single question. Do NOT include any other text or explanation other than the question."
        ),
        (
            "Generate a message for an LLM that will test its knowledge of the information in the corpus above."
            "Be sure to include details (ids, names, titles, dates, etc.) in the question so that it can be answered without access to the corpus (i.e. closed-book setting). "
            "Output only a single question. Do NOT include any other text or explanation other than the question."
        ),
        (
            "You are helping to quiz a user about the information in the corpus. "
            "Please generate a question about the subsection of the corpus above. "
            "Be sure to include details (ids, names, titles, dates, etc.) in the question to make it clear what you are asking about. "
            "Answer only with the question, do not include any other text."
        ),
    ]
    prompt = random.choice(prompts)
    return prompt


def use_case_seed_prompt(**kwargs):
    prompt = (
        "You are working to train a language model on the information in the following corpus. "
        "Your primary goal is to think about practical, real-world tasks or applications that someone could achieve using the knowledge contained within this corpus. "
        "Consider how a user might want to apply this information, not just recall it. "
        "After considering potential use cases, your task will be to generate a sample question that reflects one of these downstream applications. "
        "This question/instruction/task should be something a user, who has access to this corpus, might ask when trying to accomplish their specific goal. "
        "Output only a single question. Do NOT include any other text or explanation other than the question."
    )
    return prompt


def creative_seed_prompt(**kwargs):
    prompt = [
        (
            "You are having a creative conversation inspired by the information in the corpus. "
            "Please generate a question for your conversation partner to start off the discussion. "
            "Answer only with the question, do not include any other text."
        ),
    ]
    return random.choice(prompt)


def generic_seed_prompt(**kwargs):
    return (
        f"Please generate a single chat message to begin a conversation about the information in the corpus. Ask a question about the corpus or make a request."
    )



SEED_PROMPT_REGISTRY: dict[SEED_TYPES, Callable] = {
    "structuring": structuring_seed_prompt,
    "summarization": summarization_seed_prompt,
    "question": question_seed_prompt,
    "use_case": use_case_seed_prompt,
    "creative": creative_seed_prompt,
    "generic": generic_seed_prompt,
}

def sample_seed_prompts(seed_types: List[SEED_TYPES], batch_size: int) -> List[str]:
    seed_types = random.choices(seed_types, k=batch_size)
    return [SEED_PROMPT_REGISTRY[seed_type]() for seed_type in seed_types]

# --- end generators for 