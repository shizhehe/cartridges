import json
import time
from typing import List, Optional
import uuid
import modal
from pathlib import Path
import sys

from pydantic import BaseModel
import requests
from cartridges.clients.tokasaurus import TokasaurusClient


root = Path(__file__).parent.parent.parent

BRANCH = "geoff/cartridges"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    # capsules installation ensures we have tokasaurus client for manually creating per-container batches
    .run_commands(
        "git clone https://$GITHUB_TOKEN@github.com/hazyresearch/cartridges.git /root/cartridges",
        secrets=[modal.Secret.from_name("sabri-api-keys")],
    )
    .run_commands("cd /root/cartridges && pip install -e .")
    .run_commands(
        "git clone https://$GITHUB_TOKEN@github.com/ScalingIntelligence/tokasaurus.git /root/tokasaurus",
        secrets=[modal.Secret.from_name("sabri-api-keys")],
    )
    .run_commands("cd /root/tokasaurus && pip install -e . && pip install wandb")
)
if BRANCH != "main":
    image = image.run_commands(f"cd /root/tokasaurus && git fetch --all && git checkout --track origin/{BRANCH}")
image = image.run_commands("cd /root/tokasaurus && git pull", force_build=True)


hf_cache_vol = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True
)

MODEL_SIZE = "3b"
MINUTES = 60  # seconds
PORT = 8000
DP_SIZE = 1

GPU_TYPE = "A100-80GB"
# GPU_TYPE = "H100"
MIN_CONTAINERS = 0
MAX_CONTAINERS = 24
ALLOW_CONCURRENT_INPUTS = 2
generation = "3.2" if MODEL_SIZE in ["1b", "3b"] else "3.1"


app = modal.App(f"tksrs-batch-{BRANCH.replace('/', '-')}-llama-{MODEL_SIZE}-{DP_SIZE}x{GPU_TYPE}-min{MIN_CONTAINERS}")

class BatchRequest(BaseModel):
    chats: list[list[dict[str, str]]]
    cartridges: list[dict[str, str]] = []
    max_completion_tokens: int
    temperature: float = 0.6
    stop: Optional[List[str]] = None
    top_logprobs: Optional[int] = None


@app.function(
    image=image,
    gpu=f"{GPU_TYPE}:{DP_SIZE}",
    allow_concurrent_inputs=ALLOW_CONCURRENT_INPUTS,
    scaledown_window=5 * MINUTES,
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    secrets=[modal.Secret.from_name("sabri-api-keys")],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
    },
)
@modal.asgi_app()
def serve():
    import subprocess
    import os
    from fastapi import FastAPI, Request
    from openai import OpenAI

    web_app = FastAPI()

    os.system("nvidia-smi")
    os.system("which nvidia-smi")

    PING_TIMEOUT_SECONDS = 1.0
    WAIT_FOR_SERVER_BACKOFF_SECONDS = 1.0
    
    cmd = [
        "tksrs",
        f"model=meta-llama/Llama-{generation}-{MODEL_SIZE}-Instruct",
        f"kv_cache_num_tokens='({400_000})'",
        f"max_seqs_per_forward={1024}",
        # f"max_top_logprobs={20}" if BRANCH != "main" else "",
        f"port={PORT}",
        f"dp_size={DP_SIZE}",
    ]
    def ping() -> bool:
        """Check if the server is responsive.

        Returns:
            bool: True if the server responds with "pong", False otherwise.
        """
        try:
            response = requests.get(
                f"http://localhost:{PORT}/ping", timeout=PING_TIMEOUT_SECONDS
            )
            return response.json()["message"] == "pong"
        except requests.RequestException:
            return False

    TIMEOUT = 300
    
    subprocess.Popen(" ".join(cmd),shell=True)
    start_time = time.time()
    while time.time() - start_time < TIMEOUT:
        if ping():
            break
        time.sleep(WAIT_FOR_SERVER_BACKOFF_SECONDS)


    BATCH_INITIAL_WAIT = 0.5
    BATCH_CHECK_INTERVAL = 0.5

    @web_app.post("/batch")
    async def batch(req: BatchRequest):
        client = TokasaurusClient.Config(
                url=f"http://localhost:{PORT}/v1",
                use_modal_endpoint=False,
                model_name="meta-llama/Llama-3.2-3B-Instruct",
        ).instantiate()

        # Use the cartridge_chat method directly
        response = client.cartridge_chat(
            chats=req.chats,
            cartridges=req.cartridges,
            max_completion_tokens=req.max_completion_tokens,
            temperature=req.temperature,
            stop=req.stop,
            top_logprobs=req.top_logprobs,
        )
        
        print("Done with request!!!")
        return response
    
    return web_app
