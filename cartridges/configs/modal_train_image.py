import os
from pathlib import Path
from typing import Dict
import modal
from subprocess import check_output
import subprocess
import importlib
import sys

import torch

from cartridges.train import TrainConfig

secrets = modal.Secret.from_name("sabri-api-keys")
BRANCH = os.environ.get("BRANCH", "main")
BASE_PATH = os.path.join(os.environ["CARTRIDGES_DIR"], "cartridges/data/ruler/_data")
LOCAL_DIRS = [
    os.path.join(os.environ["CARTRIDGES_DIR"], "cartridges/data/ruler/_data")
]

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .run_commands(
        "git clone https://$GITHUB_TOKEN@github.com/chickadee-labs/cartridges-internal.git /root/cartridges-internal",
        secrets=[secrets],
    )
    .run_commands("cd /root/cartridges-internal && pip install -e .")
    .run_commands("cd /root/cartridges-internal && git fetch --all", force_build=True)
    .run_commands("cd /root/cartridges-internal && git pull", force_build=True)
    .env(
        {
            "CARTRIDGES_OUTPUT_DIR": "output",
            "CARTRIDGES_DIR": "cartridges-internal",
            "CARTRIDGES_WANDB_PROJECT": "cartridges",
            "CARTRIDGES_WANDB_ENTITY": "hazy-research",
        }
    )
)

if BRANCH != "main":
    image = image.run_commands(f"cd /root/cartridges-internal && git fetch --all && git checkout --track origin/{BRANCH}")

hf_cache_vol = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True
)

datasets_vol = modal.Volume.from_name(
    "cartridges-datasets", create_if_missing=True
)

triton_cache_vol = modal.Volume.from_name(
    "triton-cache", create_if_missing=True
)

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

NUM_GPUS = os.environ.get("NUM_GPUS", 8)
GPU_TYPE = os.environ.get("GPU_TYPE", "H100")

app = modal.App(f"train-{NUM_GPUS}x{GPU_TYPE}")

@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{NUM_GPUS}",
    secrets=[secrets],
    allow_concurrent_inputs=999,
    scaledown_window=5 * MINUTES,
    timeout=3 * HOURS,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/cartridges-datasets": datasets_vol,
        "/root/.triton/cache": triton_cache_vol,
    },
)
def train(
    script_str: str,
    script_path: str,
    data: Dict[str, bytes] = {},
    arglist: list[str] = [],
):
    num_gpus = torch.cuda.device_count()

    # we take any of the local datasets sent up to the remote,  write them to the 
    # output directory, and update the script to point to the new path
    # --- begin write local datasets ---
    for dataset_path, dataset_bytes in data.items():
        abspath = os.path.join(os.environ["CARTRIDGES_OUTPUT_DIR"], dataset_path)
        os.makedirs(os.path.dirname(abspath), exist_ok=True)

        with open(abspath, "wb") as f:
            f.write(dataset_bytes)
        script_str = script_str.replace(dataset_path, abspath)
    # --- end write local datasets ---

    # we overwrite the script at the given path with the version 
    # sent from the local entrypoint
    script_path = os.path.join("cartridges-internal", script_path)
    open(script_path, "w").write(script_str)
    
    print("Launching training script at:", script_path)
    print("Args:", arglist)
    subprocess.run(
        [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={num_gpus}",
            script_path,
            *arglist,
        ],
        env={
            **os.environ,
            # "CUDA_LAUNCH_BLOCKING": "1",
        }
    )


# Check if the git repository is dirty
def _is_git_repo_dirty(repo_path: str) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return bool(result.stdout.strip())

@app.local_entrypoint()
def main(
    *arglist
):
    path = arglist[0]
    assert "CARTRIDGES_DIR" in os.environ, "CARTRIDGES_DIR must be set to point to your local clone of the capsules repo."

    script_str = open(path).read()
    relative_path = os.path.relpath(path, os.environ["CARTRIDGES_DIR"])

    # we load any local datasets from the config and send them to the remote as bytes
    # then up on the remote, we write the bytes to the output directory and replace the 
    # local path with the relative path to the output directory
    # --- begin load local datasets ---
    PLACEHOLDER_MODULE_NAME = "config_module"
    spec = importlib.util.spec_from_file_location(PLACEHOLDER_MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[PLACEHOLDER_MODULE_NAME] = module
    spec.loader.exec_module(module)
    config: TrainConfig = module.config

    data = {}
    for dataset in config.dataset.data_sources:
        if dataset.type == "local":
            relpath = os.path.relpath(dataset.path, os.environ["CARTRIDGES_OUTPUT_DIR"])
            script_str = script_str.replace(dataset.path, relpath)
            print("Loading dataset from", dataset.path)
            data[relpath] = open(dataset.path, "rb").read()
            print("Loaded dataset from", dataset.path)
    # --- end load local datasets ---

    # Get the capsules directory from environment variable
    if _is_git_repo_dirty(os.environ["CARTRIDGES_DIR"]):
        print("The git repository at CARTRIDGES_DIR is dirty. Please commit or stash your changes before running training.")
    
    train.remote(
        script_str=script_str, 
        script_path=relative_path, 
        arglist=arglist[1:], 
        data=data
    )