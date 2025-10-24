

import json
from dataclasses import dataclass
import random
import time
from typing import Dict, List, Literal, Optional, Set

import ast
from cartridges.data.resources import DirectoryResource, Resource
from cartridges.utils import get_logger

logger = get_logger(__name__)


def _parse_imports(tree: ast.AST) -> Set[str]:
    "Returns a mapping of file names to the set of modules they import."
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:  # absolute import
                imports.add(node.module.split(".")[0])
            elif node.level > 0:                  # relative import
                # We skip relative imports for simplicity
                continue
    return imports

def _topological_sort(imports: Dict[str, Set[str]]) -> tuple[List[str], List[List[str]]]:
    """
    Topologically sort a dictionary of imports and return levels.
    imports maps filename -> set of modules that file imports
    Returns (sorted_files, levels) where levels[i] contains files with no dependencies 
    on files in levels[i] or higher.
    """
    # Kahn's algorithm for topological sorting with level tracking
    in_degree = {
        file: len([dep for dep in deps if dep in imports]) 
        for file, deps  in imports.items()
    }
    
    # Initialize queue with files that have no dependencies (in-degree 0)
    curr_level = [file for file, degree in in_degree.items() if degree == 0]
    levels = []
    
    while curr_level:
        levels.append(curr_level)
        # Process all files at current level
        next_level = []

        for curr_file in curr_level:
            # For each file that the current file depends on, reduce dependency count
            for dependent_file in imports:
                if curr_file in imports[dependent_file]:
                    in_degree[dependent_file] -= 1
                    if in_degree[dependent_file] == 0:
                        next_level.append(dependent_file)
        curr_level = next_level
    
    # Check for cycles
    if sum(len(level) for level in levels) != len(imports):
        raise ValueError(f"Cycle detected in imports: {imports.keys() - set(sum(levels, []))}")
    
    return levels

def _file_to_module_name(file_name: str) -> str:
    return file_name.replace("/", ".").replace(".py", "")

def _module_to_file_name(module_name: str) -> str:
    return module_name.replace(".", "/") + ".py"

def _file_with_header(file_name: str, content: str) -> str:
    return f"```\n # File: {file_name} \n # You can use this file with \"import {_file_to_module_name(file_name)}\" \n\n {content} \n```"

class PythonRepositoryResource(DirectoryResource):

    class Config(Resource.Config):
        path: str
        included_extensions: List[str] = [".py"]
        recursive: bool = False
        max_level: Optional[int] = None


    def __init__(self, config: Config):
        self.config = config
    
    async def setup(self):
        t0 = time.time()
        self.files: Dict[str, str] = await self._load_files()
        logger.info(f"Loaded {len(self.files)} files in {time.time() - t0:.2f} seconds")

        self.module_to_content = {
            _file_to_module_name(file_name): content for file_name, content in self.files.items()
        }
        
        t0 = time.time()
        self.module_to_asts = {
            module_name: ast.parse(content) for module_name, content in self.module_to_content.items()
        }
        logger.info(f"Parsed {len(self.module_to_asts)} files in {time.time() - t0:.2f} seconds")

        t0 = time.time()
        self.module_to_imports = {
            module_name: _parse_imports(ast) for module_name, ast in self.module_to_asts.items()
        }
        logger.info(f"Parsed {len(self.module_to_imports)} imports in {time.time() - t0:.2f} seconds")

        t0 = time.time()
        self.levels = _topological_sort(self.module_to_imports)
        logger.info(f"Topologically sorted {len(self.levels)} levels in {time.time() - t0:.2f} seconds")
                
    
    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        if not self.module_to_content:
            raise ValueError("No files available. Make sure to call setup() first.")

        if self.config.max_level is not None:
            candidate_modules = sum(self.levels[:self.config.max_level], [])
        else:
            candidate_modules = sum(self.levels, [])

        module = random.choice(candidate_modules)
        
        context = self.module_to_content[module]
        context = f"\n\n{_file_with_header(_module_to_file_name(module), context)}"
        seed_prompts = random.choices(CODE_SEED_PROMPTS, k=batch_size)
        return context, seed_prompts





CODE_SEED_PROMPTS = [
    (
        "Generate a question for an LLM that will test its knowledge of the information in this python file. "
        "In your question be sure to include the full qualified name of the methods and classes in the file "
        "(e.g. if the file is `foo/bar.py` and the function is `baz`, the full qualified name is `foo.bar.baz`) "
        "Output only a single question. Do NOT include any other text or explanation other than the question."
    ),
    (
        "Generate a question for an LLM that will test its knowledge of the information in this python file. "
        "In your question be sure to include the name of the file that the function is defined in since the responder "
        " will have access to the many files in the repository and may not know which one is being referenced. "
        "Output only a single question. Do NOT include any other text or explanation other than the question."
    ),
    (
        "You are helping to quiz a user about the code in this one python file in a larger repository. "
        "Please generate a question about the code in the file. "
        "The user will not know which exact file the question is about, so be sure to include the full " 
        "qualified name of the methods and classes in the file (e.g. if the file is `foo/bar.py` and the "
        " function is `baz`, the full qualified name is `foo.bar.baz`)."
        "Answer only with the question, do not include any other text."
    ),
    (
        "Please generate a single chat message instructing an LLM to summarize the code in this Python file. "
        "Make sure the instruction is very explicit about the module/file that you want to summarize. "
        "Include the name of the module/file in the instruction. "
        "Answer only with the instruction, do not include any other text."
    ),
    (
        "Generate a single question that assesses an LLM’s understanding of the code in this specific Python file. "
        "The question must reference functions and classes by their fully qualified names based on the file path "
        "(e.g., if the file is ⁠pkg/subpkg/module.py and the function is ⁠run, the full qualified name is ⁠pkg.subpkg.module.run). "
        "Output only the question, with no extra commentary."
    ),
    (
        "Create one question that tests knowledge of this Python file’s contents. "
        "Your question must explicitly state the exact file name and path where the referenced function or class is defined, "
        "since the responder may be looking across multiple files. "
        "Output only the question, nothing else."
    ),
    (
        "Produce a single quiz question about this Python file that mentions the fully qualified names for all referenced symbols "
        "(classes, methods, or functions), derived from the module path (e.g., if the file is ⁠foo/bar/utils.py, use "
        "⁠foo.bar.utils.ClassName.method). "
        "Respond with only the question and no additional text."
    ),
    (
        "Write one instruction asking an LLM to explain the purpose and behavior of the code in this module. "
        "The instruction must clearly identify the module by its file path and name. "
        "Answer only with the instruction, and do not include any extra explanation."
    ),
    (
        "Generate exactly one question that verifies understanding of a particular function or class in this Python file. "
        "Require the responder to cite the fully qualified name (e.g., ⁠app.core.handlers.process_request) and include the file name in the question. "
        "Output only the question and no other text."
    ),
] + [
    (
        f"Please generate a single chat message instructing an LLM to structure the code in this Python file in a {data_format}. "
        "Output only the chat message itself and absolutely nothing else. "
        "Make sure it is clear what module/file you are asking about. "
        f"For example: \n\n"
        "Can you describe how the code in this file works by writing a {data_format} file?"
        "Make sure to include the full qualified name of the methods and classes in the file "
        "(e.g. if the file is `foo/bar.py` and the function is `baz`, the full qualified name is `foo.bar.baz`)."
    ) for data_format in ["JSON", "YAML", "TOML"]
]