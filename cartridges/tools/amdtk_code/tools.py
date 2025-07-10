from typing import List, Literal, Optional
import aiohttp

from pydantic import BaseModel

from codemem.tools.amdtk_code.code_server import FileUpdate, CommandResult
from codemem.tools.base import Tool


class AMDTKCodeTool(Tool):
    class Config(Tool.Config):
        code_server_host: str = "127.0.0.1"
        code_server_port: int = 9002

    class ToolInput(BaseModel):
        files: List[FileUpdate]


    def __init__(self, config: Config):
        super().__init__(config)
        self.code_server_host = config.code_server_host
        self.code_server_port = config.code_server_port

    async def run_tool(self, input: ToolInput) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"http://{self.code_server_host}:{self.code_server_port}/execute", json=input.model_dump()) as response:
                data = await response.json()
        breakpoint()

        results = [CommandResult.model_validate(r) for r in data["detail"]]
        breakpoint()
    
    @property
    def description(self) -> str:
        return (
            "Test and benchmark a matmul kernel written with the ThunderKittens framework for an AMD GPU. "
            "Pass a list of files to change in the ThunderKittens framework."
            "The tool will apply the changes before running the kernel."
        )
    
    @property
    def name(self) -> str:
        return "test_matmul_kernel"
