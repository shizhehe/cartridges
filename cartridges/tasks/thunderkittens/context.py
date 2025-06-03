

import os
from typing import List

from cartridges.context import StructuredContext
from cartridges.structs import Context, Section
from cartridges.tasks.thunderkittens.tk_strings import ( tk_description ) 
from cartridges.context import BaseContextConfig


class TKFile(StructuredContext):
    desc: str
    content: str


class TKRepo(StructuredContext):
    files: List[TKFile]


def load_tk_dataset() -> List[str]:

    data_path = "/home/simarora/code/ThunderKittens/"
    includes_path = f"{data_path}/include/"

    toc_walkthrough = []
    sections = []


    ##################################################################
    # 0. outline of the content
    ##################################################################
    title = "Tools for writing GPU kernels."
    toc_walkthrough.append(tk_description)


    ##################################################################
    # 1. load includes utility folder files
    ##################################################################
    for subfolder in ['common', 'pyutils']:
        title = f"ThunderKittens/include/{subfolder} folder"
        dir_path = f"{includes_path}/{subfolder}/"
        files = os.listdir(dir_path)
        for file in files:
            with open(os.path.join(dir_path, file), 'r') as f:
                content = f.read()
                context_str = f"File name: {file}\n\n{content}"
                fpath = f"{dir_path}/{file}"
                desc=f"TK library: {title} file"
                sections.append(TKFile(
                    content = context_str, desc = desc
                ))


    # ##################################################################
    # # 2. load includes types folder 
    # ##################################################################
    header_file  = f"{includes_path}/types/types.cuh"
    shared_file = f"{includes_path}/types/shared/st.cuh"
    with open(shared_file, 'r') as f:
        shared_content = f.read()
        desc = f"TK library: basic data structure for shared memory"
        sections.append(TKFile(
            content = shared_content, desc = desc
        ))
    register_file = f"{includes_path}/types/register/rt.cuh"
    with open(register_file, 'r') as f:
        register_content = f.read()
        desc = f"TK library: basic data structure for register memory"
        sections.append(TKFile(
            content = register_content, desc = desc
        ))
    globals_file = f"{includes_path}/types/global/gl.cuh"
    with open(globals_file, 'r') as f:
        globals_content = f.read()
        desc = f"TK library: basic data pointer to global memory"
        sections.append(TKFile(
            content = globals_content, desc = desc
        ))


    ##################################################################
    # 3. load includes ops folder
    ##################################################################
    header_file  = f"{includes_path}/ops/ops.cuh"

    # memory ops
    warp_memory_path = f"{includes_path}/ops/warp/memory/tile/"
    global_to_register_path = f"{warp_memory_path}/global_to_register.cuh"
    with open(global_to_register_path, 'r') as f:
        global_to_register_content = f.read()
        desc=f"TK library: global to register data loading file"
        sections.append(TKFile(
            content = global_to_register_content, desc=desc
        ))
    global_to_shared_path = f"{warp_memory_path}/global_to_shared.cuh"
    with open(global_to_shared_path, 'r') as f:
        global_to_shared_content = f.read()
        desc=f"TK library: global to shared data loading file"
        sections.append(TKFile(
            content = global_to_shared_content, desc=desc
        ))
    shared_to_register_path = f"{warp_memory_path}/shared_to_register.cuh"
    with open(shared_to_register_path, 'r') as f:
        shared_to_register_content = f.read()
        desc=f"TK library: shared to register data loading file"
        sections.append(TKFile(
            content = shared_to_register_content, desc=desc
        ))

    # compute ops
    warp_register_path = f"{includes_path}/ops/warp/register/tile/"
    for subfile in ['conversions.cuh', 'maps.cuh', 'mma.cuh', 'reductions.cuh']:
        with open(f"{warp_register_path}/{subfile}", 'r') as f:
            content = f.read()
            desc = f"TK library: warp register operations {subfile} file"
            sections.append(TKFile(
                content = content, desc=desc
            ))
    warp_shared_path = f"{includes_path}/ops/warp/shared/tile/"
    for subfile in ['maps.cuh', 'conversions.cuh', 'reductions.cuh']:
        with open(f"{warp_shared_path}/{subfile}", 'r') as f:
            content = f.read()
            desc = f"TK library: warp shared operations {subfile} file"
            sections.append(TKFile(
                content = content, desc=desc
            ))


    ##################################################################
    # 4. load some example kernels written in TK
    ##################################################################
    examples_path = f"{data_path}/kernels/"
    kernel_examples = ['micro_add', 'micro_warp_mul', 'matmul/H100/']       # TODO!!!!


    ##################################################################
    # 5. load extra helpful content
    ##################################################################
    extra_path = f"{data_path}/extra/"
    extra_files = os.listdir(extra_path)
    for file in extra_files:
        if 'pdf' in file: # temporarily
            continue
        with open(os.path.join(extra_path, file), 'r') as f:
            content = f.read()
            context_str = f"File name: {file}\n\n{content}"
            fpath = f"{extra_path}/{file}"
            desc = f"TK library: extra helpful content about {file}"
            sections.append(TKFile(
                content = context_str, desc = desc
            ))

    return sections


class TKContextConfig(BaseContextConfig):

    mode: str = "train"

    def instantiate(self) -> Context:
        print(f"Loading TK context with mode: {self.mode}")
        
        context = load_tk_dataset()

        if self.mode == "train":
            repo = TKRepo(files = context)
            return repo
        
        elif self.mode == "icl":
            context = load_tk_dataset()
            
            sections = []
            for i in range(len(context)):
                sections.append(context[i].content)

            section = Section(
                desc = "TK library: ThunderKittens knowledge",
                content = "\n\n".join(sections)
            )

            repo = Context(
                title = "TK library: ThunderKittens knowledge",
                sections = [section],
            )
            return repo



