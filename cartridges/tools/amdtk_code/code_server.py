"""
A very small asyncio/FastAPI based server that can execute LLM-generated Python
snippets on a pool of GPUs.  Each GPU is represented by a single long–lived
worker which guarantees that only one snippet is ever executed on that GPU at
a time.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from typing import Dict, List, Optional, Tuple
from contextlib import asynccontextmanager
import shutil
import tempfile
from pathlib import Path

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Body


ROOT_DIR = Path(__file__).parent.parent

class FileUpdate(BaseModel):
    relpath: str
    content: str

class CommandResult(BaseModel):
    cwd: str
    command: str
    env: Dict[str, str]

    stdout: str
    stderr: str
    return_code: Optional[int]


async def run_command(cwd: Path, command: str, env: Dict[str, str]) -> CommandResult:
    process = await asyncio.create_subprocess_shell(
        command,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env
    )
    stdout, stderr = await process.communicate()
    return CommandResult(
        cwd=str(cwd),
        command=command,
        env=env,
        return_code=process.returncode,
        stdout=stdout.decode(),
        stderr=stderr.decode()
    )

class GPUWorker:
    """A single-GPU worker that serialises execution of code snippets."""

    def __init__(self, gpu_id: int) -> None:
        self.gpu_id: int = gpu_id
        self._queue: asyncio.Queue[
            Tuple[Path, asyncio.Future[CommandResult]]
        ] = asyncio.Queue()
        # Don't create the task here - we'll do it in startup
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the background runner coroutine."""
        if self._task is None:
            self._task = asyncio.create_task(self._runner())

    async def stop(self) -> None:
        """Stop the background runner coroutine."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def is_idle(self) -> bool:
        """Whether the worker currently has no pending jobs."""
        return self._queue.empty()

    async def submit(self, source_dir: Path) -> CommandResult:
        """
        Schedule a snippet for execution and await its completion.
        Returns a dict with 'output' and 'error' keys.
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[CommandResult] = loop.create_future()
        await self._queue.put((source_dir, fut))
        return await fut

    async def _runner(self) -> None:
        """Continuously process items in the queue."""
        while True:
            source_dir, fut = await self._queue.get()
            try:
                result = await self._execute(source_dir)
                fut.set_result(result)
            except Exception as exc:  # pragma: no cover
                fut.set_exception(exc)
            finally:
                self._queue.task_done()

    async def _execute(self, source_dir: Path) -> CommandResult:
        """
        Run the given snippet in a background thread *safely* so the
        event-loop stays responsive.  Stdout / stderr are captured.
        """
        kernel_dir = source_dir / "kernels" / "memory_matmul"
        command = await run_command(kernel_dir, "python test_python.py", {
            **os.environ.copy(),
            "THUNDERKITTENS_ROOT": str(source_dir / "ThunderKittens-HIP"),
            "HIP_VISIBLE_DEVICES": str(self.gpu_id)
        })
        return command



###############################################################################
# Server setup
###############################################################################

NUM_GPUS: int = int(os.getenv("NUM_GPUS", "8"))
if NUM_GPUS < 1:
    raise RuntimeError("NUM_GPUS must be at least 1")

# Global workers list - will be populated in startup
_workers: List[GPUWorker] = []

# Simple load-balancing: pick first idle worker, else the least-busy queue.
def _choose_worker() -> GPUWorker:
    for w in _workers:
        if w.is_idle():
            return w
    # All busy – choose the one with the smallest backlog.
    return min(_workers, key=lambda w: w._queue.qsize())  # type: ignore[attr-defined]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifecycle of GPU workers."""
    # Startup: create and start workers
    global _workers
    _workers = [GPUWorker(i) for i in range(NUM_GPUS)]
    for worker in _workers:
        await worker.start()
    
    yield
    
    # Shutdown: stop workers
    for worker in _workers:
        await worker.stop()

app = FastAPI(
    title="LLM Code Execution Server",
    version="0.1.0",
    lifespan=lifespan
)

class ExecuteRequest(BaseModel):
    files: List[FileUpdate]
    # prompt: str
    # model_name: str

@app.post("/execute")
async def execute_code(request: ExecuteRequest) -> List[CommandResult]:
    files = request.files
    prompt = request.prompt
    model_name = request.model_name

    job_id = str(uuid.uuid4())

    # 1. copy the AMDTK directory into a temporary directory
    src_dir = ROOT_DIR / "AMDTK"
    temp_dir = tempfile.mkdtemp()
    dst_dir = Path(temp_dir) / "AMDTK"
    shutil.copytree(src_dir, dst_dir)

    # 2. make edits to the files
    for file in files:
        print(f"Updating file: {file.relpath} with content: {file.content}")
        file_path = dst_dir / file.relpath
        file_path.write_text(file.content)
        print(f"File updated: {file_path} to {file_path.read_text()}")

    # 3. Build the code
    kernel_dir = dst_dir / "kernels" / "memory_matmul"
    build_command = await run_command(
        cwd=kernel_dir, 
        command="make clean && make", 
        env={
            **os.environ.copy(),
            "THUNDERKITTENS_ROOT": str(dst_dir / "ThunderKittens-HIP")
        }
    )

    # 4. Run the code on the GPU using the GPU workers
    worker = _choose_worker()
    try:
        python_command = await worker.submit(dst_dir)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


    return [build_command, python_command]


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "9002"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
