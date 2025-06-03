import random
import os
import re
import sys
import torch
from torch.utils.cpp_extension import load_inline
import subprocess
from pathlib import Path
from datasets import load_dataset
from dataclasses import dataclass
from typing import List
from transformers import PreTrainedTokenizerFast
from pydrantic import ObjectConfig
from capsules.datasets import ( CapsuleGenerateDataset, CapsuleGenerateDatasetElement, TEMPLATE, )


from capsules.tasks.thunderkittens.kernelbench_utils import ( 
    KernelExecResult, 
    eval_kernel_against_ref, 
    check_metadata_serializable_all_types, 
    create_compilation_file_kernel_exec_result,
)

from capsules.tasks.thunderkittens.tk_strings import (
    prompt_generate_custom_cuda_from_prompt_template,
    prompt_generate_custom_thunderkitten_from_prompt_template,
    create_tk_makefile
)

REPO_TOP_DIR = "/home/simarora/code/capsules/capsules/tasks/thunderkittens/"


def extract_code_blocks_of_type(text, code_language_type: str) -> str:
    '''
    Extract code blocks with specific language type from text, combine them to return as a single string
    '''
    pattern = f"```{code_language_type}\n(.*?)```"
    if not text.endswith("```"):
        text = text + "```"
    if not text.startswith("```"):
        text = "```python" + text
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Join all matched code blocks with newlines and strip whitespace
    return "\n".join(match.strip() for match in matches)


@dataclass
class KernelBenchQuestion:
    kernelbench_id: int
    kernelbench_name: str
    question: str
    answer: str
    dir_path: str


class KernelBenchGenerateDataset(CapsuleGenerateDataset):
    class Config(ObjectConfig):
        _pass_as_config = True
        max_questions: int = 1
        mode: str = 'cuda' # or 'tk'

        dataset_name: str = "ScalingIntelligence/KernelBench" 
        level: int = 1

        # Evaluation
        max_tokens: int = 4096
        temperature: float = 0.0
        gpu_arch: List[str] = ["Hopper"]

        # Specify where to write and build the kernels locally
        kernel_builds_dir: str = f"{REPO_TOP_DIR}/kernels/"
        

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):
        self.config = config
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]

        assert os.environ.get("THUNDERKITTENS_ROOT"), "THUNDERKITTENS_ROOT environment variable is not set, please run source env.src in the ThunderKitten repo"
        
        questions = []
        for elt in curr_level_dataset:
            
            kernel_dir = os.path.join(self.config.kernel_builds_dir, f"level_{self.config.level}_problem_{elt["problem_id"]}_mode_{self.config.mode}")

            if self.config.mode == 'cuda': 
                question_prompt = prompt_generate_custom_cuda_from_prompt_template(elt['code'])
            elif self.config.mode == 'tk':
                question_prompt = prompt_generate_custom_thunderkitten_from_prompt_template(elt['code'])
            
            question = KernelBenchQuestion(
                kernelbench_id = elt["problem_id"],
                kernelbench_name = elt["name"],
                question = question_prompt,
                answer = elt["code"],
                dir_path = str(kernel_dir),
            )
            questions.append(question)
        
        random.Random(42).shuffle(questions)
        
        if self.config.max_questions is not None:
            questions = questions[: self.config.max_questions]
        self.questions = questions
        self.tokenizer = tokenizer


    def __getitem__(self, index: int) -> CapsuleGenerateDatasetElement:
        question: KernelBenchQuestion = self.questions[index]

        input_ids = self.tokenizer.apply_chat_template(
            ( [] )
            + [ {
                "role": "user",
                "content": question.question,
            } ],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=TEMPLATE,
        )

        # save prompt
        kernel_dir = Path(question.dir_path)
        os.makedirs(kernel_dir, exist_ok=True)
        with open(os.path.join(kernel_dir, "prompt.txt"), "w") as f:
            f.write(question.question)
        create_tk_makefile(kernel_dir)

        return CapsuleGenerateDatasetElement(
            input_ids=input_ids,
            prompt=question.question,
            answer=question.answer,
            convo_id=question.kernelbench_id,
            metadata={"idx": index},
        )

    def __len__(self):
        return len(self.questions)
    

    def score(
        self, 
        pred: str, 
        answer: str, 
        convo_id: str,
    ):
        # 1. Parse results
        inference_result = pred 
        kernel_code = extract_code_blocks_of_type(inference_result, "cpp")
        new_model_code = extract_code_blocks_of_type(inference_result, "python")
        assert new_model_code is not None, "Custom Model code generation failed"

        # 2. Save the result
        kernel_dir = os.path.join(self.config.kernel_builds_dir, f"level_{self.config.level}_problem_{convo_id}_mode_{self.config.mode}")
        os.makedirs(kernel_dir, exist_ok=True)
        with open(os.path.join(kernel_dir, "inference.txt"), "w") as f:
            f.write(inference_result)
        with open(os.path.join(kernel_dir, "gen.py"), "w") as f:
            f.write(new_model_code)

        # 3. If kernel code exists, compile it
        if kernel_code:
            with open(os.path.join(kernel_dir, "custom_tk.cu"), "w") as f:
                f.write(kernel_code)
            try:
                make_process = subprocess.run(
                    ["make"], 
                    cwd=kernel_dir,
                    capture_output=True,
                    text=True,
                    check=True
                )
                print("Make output:", make_process.stdout)
            except subprocess.CalledProcessError as e:
                print("Make failed with error:", e.stderr)
                print("Failed to compile kernel")
                return 0, {
                    "compiled": False,
                    "correctness": False,
                    "runtime": -1.0,
                }
            
            sys.path.append(kernel_dir)
            try:
                import tk_kernels # we name all the thunderkitten kernel modules as tk_kernels now!
                print(f"Imported ThunderKittens Kernel modules at {kernel_dir}: {dir(tk_kernels)}")
            except ImportError as e:
                print(f"Failed to import tk_kernels: {e}")
                kernel_exec_result = create_compilation_file_kernel_exec_result(f"Failed to import tk_kernels: {e}")

        # 4. Try to dynamically execute and get ModelNew
        try:
            exec_globals = {
                "__builtins__": __builtins__,
                "torch": torch,
                "nn": torch.nn,
                "load_inline": load_inline,
            }
            exec(new_model_code, exec_globals)
            model_cls = exec_globals.get("ModelNew")
            assert model_cls is not None, "ModelNew class not defined in generated code"
        except Exception as e:
            print(f"Error executing new_model_code: {e}")
            kernel_exec_result = create_compilation_file_kernel_exec_result(str(e))
            return 0, {
                    "compiled": False,
                    "correctness": False,
                    "runtime": -1.0,
                }

        # 5. Evaluate the kernel (compiled inline via load_inline)
        try:
            kernel_exec_result = eval_kernel_against_ref(
                answer,
                model_cls=model_cls,
                measure_performance=True,
                num_correct_trials=5,
                num_perf_trials=100
            )
        except Exception as e:
            print(f"Error evaluating kernel: {e}")
            kernel_exec_result = create_compilation_file_kernel_exec_result(str(e))
            return 0, {
                    "compiled": False,
                    "correctness": False,
                    "runtime": -1.0,
                }

        # 6. Output
        print(f"Evaluation result for level {self.config.level} problem {convo_id}: {kernel_exec_result}")
        score = kernel_exec_result.correctness
        details = {
            "compiled": kernel_exec_result.compiled,
            "correctness": kernel_exec_result.correctness,
            "runtime": kernel_exec_result.runtime,
            # "metadata": kernel_exec_result.metadata,
            # "runtime_stats": kernel_exec_result.runtime_stats,
        }
        return score, details
        

