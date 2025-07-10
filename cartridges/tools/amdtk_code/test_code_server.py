"""
demo_execute.py

A tiny helper that sends a dummy Python snippet to the /execute endpoint
exposed by `code_server.py` and prints the results.

Prerequisites:
    pip install requests

Usage (with the server already running locally on the default port 9000):

    python demo_execute.py

You can override the host/port, e.g.:

    python demo_execute.py --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import requests


AMDTK_ROOT = Path("/shared/amdgpu/home/tech_ops_amd_xqh/sabri/code-memory/AMDTK")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a simple code snippet to the code execution server."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server hostname")
    parser.add_argument("--port", default=9000, type=int, help="Server port")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/execute"

    files = [
        # {
        #     "relpath": "ThunderKittens-HIP/include/ops/warp/memory/tile/global_to_shared_load_naive.cuh",
        #     "content": open(AMDTK_ROOT / "ThunderKittens-HIP/include/ops/warp/memory/tile/global_to_shared_load.cuh").read()
        # },
        # {
        #     "relpath": "ThunderKittens-HIP/include/ops/warp/memory/tile/shared_to_register_load_naive.cuh",
        #     "content": open(AMDTK_ROOT / "ThunderKittens-HIP/include/ops/warp/memory/tile/shared_to_register_load.cuh").read()
        # },
        # {
        #     "relpath": "ThunderKittens-HIP/include/types/shared/st_naive.cuh",
        #     "content": open(AMDTK_ROOT / "ThunderKittens-HIP/include/types/shared/st.cuh").read()
        # }
    ]
    
    response = requests.post(url, json={"files": files}, timeout=30)
    print(response.content)

    data = response.json()
    print(data)


    # print(f"Job ID : {data.get('job_id')}")
    # print(f"GPU ID : {data.get('gpu_id')}")
    # print("----- STDOUT -----")
    # print(data.get("output") or "(no stdout)")
    # print("----- STDERR -----")
    # print(data.get("error") or "(no stderr)")


if __name__ == "__main__":
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(main) for _ in range(8)]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"main() generated an exception: {exc}")
