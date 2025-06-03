
import os
from capsules.tasks.thunderkittens.kernelbench_utils import ( 
    read_file, 
    KERNEL_BENCH_PATH as REPO_TOP_PATH,
)


############################################
# CUDA Prompt
############################################
PROBLEM_STATEMENT = """You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups. \n
    You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
"""


PROBLEM_INSTRUCTION = """
Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Match the specs of the Model module. Just output the new model code, no other text, and NO testing code! \n
"""


def prompt_generate_custom_cuda(
    arc_src: str, example_arch_src: str, example_new_arch_src: str
) -> str:
    prompt = PROBLEM_STATEMENT

    if example_arch_src != "" and example_new_arch_src != "":
        prompt += f"""
        Here's an example to show you the syntax of inline embedding custom CUDA operators in torch: The example given architecture is: \n
        ``` \n
        {example_arch_src}
        ``` \n
        The example new arch with custom CUDA kernels looks like this: 
        ```
        {example_new_arch_src}
        ``` \n
        """

    prompt += f"""
    You are given the following architecture: \n
    ```
    {arc_src}
    ```
    """
    prompt += PROBLEM_INSTRUCTION
    return prompt


def prompt_generate_custom_cuda_from_prompt_template(ref_arch_src: str) -> str:
    """
    Using prompt example (an element-wise addition) for prompt templates
    The most basic form of example just to show LLM the task and the expected output format
    """
    arch = ref_arch_src
    # path to prompt template, show an example of Model (torch specifications) and ModelNew (torch + custom CUDA kernels)
    example_arch_path = os.path.join(REPO_TOP_PATH, f"src/prompts/model_ex_add.py")
    example_new_arch_path = os.path.join(REPO_TOP_PATH, f"src/prompts/model_new_ex_add.py")
    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)

    return prompt_generate_custom_cuda(arch, example_arch, example_new_arch)



############################################
# TK Prompt
############################################


tk_description = """
The challenge of mapping AI architectures to GPU hardware is creating a critical bottleneck in AI progress. Despite substantial efforts, hand-written custom kernels fail to meet their theoretical performance thresholds, even on well-established operations like linear attention. The diverse hardware capabilities of GPUs might suggest that we need a wide variety of techniques to achieve high performance. However, our work explores whether a small number of key abstractions can drastically simplify the process. We present TK, a framework for writing performant AI kernels while remaining easy to use and maintain. Our abstractions map to the three levels of the GPU hierarchy: (1) at the warp-level, we provide 16x16 matrix tiles as basic data structures and PyTorch-like parallel compute operations over tiles, (2) at the thread-block level, we provide a template for overlapping asynchronous operations across parallel warps, and (3) at the grid-level, we provide support to help hide the block launch and tear-down, and memory costs. We show the value of TK by providing kernels that match or outperform prior kernels for a range of AI operations.

The TK library code is organized as follows:
- `kittens.cuh`: is the main header file for TK, and includes all of the functionality we'll need. 
- `common`: contains common utilities and functions used throughout the library (data types, math functions, etc.)
- `ops`: contains the implementations of basic pytorch-like operations (e.g., exp, mma_AB, mma_ABt, cumsum, etc.). There are sub-directories for warp and warpgroup scope operations (warp::exp, warpgroup::mma_AB, etc.).
- `types`: contains the definitions of the basic tile data structure used in TK (e.g., register tiles, shared tiles, global descriptors, etc.). 
- `pyutils`: contains python utilities for TK (e.g., pybind11 wrappers, etc.)


Here is a very simple starter kernel in TK:

"""


PROBLEM_TK_STATEMENT = """You write custom ThunderKitten kernels to replace the PyTorch operators in the given architecture to get speedups. \n
    ThunderKitten (TK) provides tile primitives to write CUDA Kernels for GPUs. You can make the decision to replace some operators in the given Torch architecture with custom ThunderKitten kernels and leave others unchanged.\n
"""


PROBLEM_TK_INSTRUCTION = """
Optimize the architecture named Model with custom ThunderKitten operators! Please output two piece of code wrapped in 2 codeblocks: 
1. ThunderKitten Kernel in .cu. Wrap this in ```cpp and ```
2. Optimized Torch Architecture in .py. It should have imports tk_kernels at the top and uses the replaced ThunderKitten kernels. Name your optimized output architecture ModelNew. Just output the new model code, no other text, and NO testing code! Wrap this in ```python and ```

Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Match the specs of the Model module.\n
"""


def prompt_generate_custom_thunderkitten_from_prompt_template(ref_arch_src: str) -> str:
    """
    Using prompt example (an element-wise addition) for prompt templates
    The most basic form of example just to show LLM the task and the expected output format
    """
    arch = ref_arch_src # this is the problem to
    # These are strictly defined for now

    # path to prompt template, show an example of Model (torch specifications) and ModelNew (torch + custom CUDA kernels)
    example_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/tk_prompts/model_ref_ex_mul.py"
    )
    example_new_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/tk_prompts/model_new_ex_mul.py"
    )
    example_new_kernel_path = os.path.join(
        REPO_TOP_PATH, f"src/tk_prompts/kernel_new_ex_mul.cu"
    )

    example_complex_kernel_path = os.path.join(
        REPO_TOP_PATH, f"src/tk_prompts/kernel_new_ex_attn.cu"
    )

    tk_knowledge_path = os.path.join(REPO_TOP_PATH, "src/tk_prompts/tk_knowledge.txt")

    example_arch_src = read_file(example_arch_path)
    example_new_arch_src = read_file(example_new_arch_path)
    example_new_kernel_src = read_file(example_new_kernel_path)
    example_complex_kernel_src = read_file(example_complex_kernel_path)
    tk_knowledge = read_file(tk_knowledge_path)

    prompt = PROBLEM_TK_STATEMENT

    if tk_knowledge:
        prompt += f"""
        Here is some information about ThunderKitten: \n
        ```
        {tk_knowledge}
        ``` \n
        """

    if example_arch_src != "" and example_new_arch_src != "":
        prompt += f"""
        Here's an example to show you how to write and use ThunderKitten kernel for an example problem: The example given PyTorch architecture to optimize is: \n
        ``` \n
        {example_arch_src}
        ``` \n
        The example new ThunderKitten kernel looks like this: 
        ```
        {example_new_kernel_src}
        ``` \n
        The example new PyTorch architecture calling custom ThunderKitten kernels looks like this: 
        ```
        {example_new_arch_src}
        ``` \n
        
        """

    # if example_complex_kernel_src:
    #     prompt += f"""
    #     Here's an example of TK Kernel that tiles the computation because the tensors are too big: \n
    #     ``` \n
    #     {example_complex_kernel_src}
    #     ``` \n
    #     """

    prompt += "\nNow it's your turn to write a new kernel. Focus on the following problem:\n"

    prompt += PROBLEM_TK_INSTRUCTION
    return prompt


def create_tk_makefile(kernel_dir: str):
    """
    Write a makefile for ThunderKitten (Pybind) to the kernel directory
    Assume kernel is in file custom_tk.cu, the kernel name is called custom_kernel
    """

    
    makefile_content = """# Compiler
NVCC=nvcc
GPU=A100

TARGET=tk_kernels
SRC=custom_tk.cu

NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -x cu -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static -lcublas
NVCCFLAGS+= -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype $(shell python3 -m pybind11 --includes) $(shell python3-config --ldflags) -shared -fPIC -lpython3.10

# Conditional setup based on the target GPU
ifeq ($(GPU),4090)
\tNVCCFLAGS+= -DKITTENS_4090 -arch=sm_89
else ifeq ($(GPU),A100)
\tNVCCFLAGS+= -DKITTENS_A100 -arch=sm_80
else
\tNVCCFLAGS+= -DKITTENS_HOPPER -arch=sm_90a
endif

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
\t$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)$(shell python3-config --extension-suffix)

# Clean target
clean:
\trm -f $(TARGET)"""

    with open(os.path.join(kernel_dir, "Makefile"), "w") as f:
        f.write(makefile_content)






