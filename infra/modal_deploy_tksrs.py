import modal
from pathlib import Path

# Run command: modal deploy deploy_llama_modal_entry.py
# https://modal.com/apps/hazyresearch/main (view results of serving)

root = Path(__file__).parent.parent.parent

# BRANCH = "add-top-logprobs"
BRANCH = "capsules"

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .run_commands(
        "git clone https://$GITHUB_TOKEN@github.com/jordan-benjamin/tokasaurus.git /root/tokasaurus",
        secrets=[modal.Secret.from_name("sabri-api-keys")],
    )
    .run_commands("cd /root/tokasaurus && pip install -e .")
    .run_commands(f"cd /root/tokasaurus && git fetch --all && git checkout --track origin/{BRANCH}")
    .run_commands("cd /root/tokasaurus && git pull", force_build=True)
)


hf_cache_vol = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True
)
flashinfer_cache_vol = modal.Volume.from_name(
    "flashinfer-cache", create_if_missing=True
)

MODEL_SIZE = "3b"
MINUTES = 60  # seconds
PORT = 8000
DP_SIZE = 1
GPU_TYPE = "A100-80GB"
# GPU_TYPE = "L40S"
# GPU_TYPE = "H100"
MIN_CONTAINERS = 0
MAX_CONTAINERS = 64
ALLOW_CONCURRENT_INPUTS = 4
generation = "3.2" if MODEL_SIZE in ["1b", "3b"] else "3.1"

app = modal.App(f"tksrs-entry-{BRANCH}-{MODEL_SIZE}-{DP_SIZE}x{GPU_TYPE.split('-')[0]}-min{MIN_CONTAINERS}-max{MAX_CONTAINERS}")

@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{DP_SIZE}",
    allow_concurrent_inputs=ALLOW_CONCURRENT_INPUTS,
    scaledown_window=1 * MINUTES,
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    secrets=[modal.Secret.from_name("sabri-api-keys")],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/flashinfer": flashinfer_cache_vol,
    },
)
@modal.web_server(
    port=PORT, 
    startup_timeout=5 * MINUTES,     
)
def serve():
    import subprocess
    import os

    os.system("nvidia-smi")
    os.system("which nvidia-smi")
    
    cmd = [
        "FLASHINFER_WORKSPACE_BASE=/root/.cache/huggingface/flashinfer",
        "python",
        "tokasaurus/entry.py",
        f"model=meta-llama/Llama-{generation}-{MODEL_SIZE}-Instruct",
        f"kv_cache_num_tokens={400_000}",
        f"port={PORT}",
        f"local_proc_name='model_worker'",
    ]

    subprocess.Popen(
        " ".join(cmd),
        shell=True,
        cwd="/root/tokasaurus",
    )

