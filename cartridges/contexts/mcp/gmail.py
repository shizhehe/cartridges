import os
from typing import List
from cartridges.contexts.mcp.base import MCPContext
from cartridges.synthesizers.self_study_mcp import PromptSampler

from pydantic import BaseModel

class Message(BaseModel):
    id: str
    subject: str
    from_address: str
    to_addresses: List[str]
    date: str
    snippet: str
    content: str
    raw: dict

class Thread(BaseModel):
    id: str
    messages: List[Message]
    

class GmailMCPContext(MCPContext):

    class Config(MCPContext.Config):
        email: str
        
    def __init__(self, config: Config):
        command = "python"
        args = [
            "-m",
            "cartridges.contexts.mcp.gmail_server"
        ]
        env = {
            "CARTRIDGES_DIR": os.environ["CARTRIDGES_DIR"],
            "CARTRIDGES_OUTPUT_DIR": os.environ["CARTRIDGES_OUTPUT_DIR"],
        }
        super().__init__(config, command=command, args=args, env=env)
    
    async def sample_subcontext(self):
        if self.session is None:
            raise RuntimeError("Not connected to an MCP server. You must call `connect_to_server` first.")
    
        out = await self.session.call_tool(
            name="fetch_threads",
            arguments={
                "num_threads": 4,
                "label_names": ["Categories (stanford)/Primary (stanford)"],
            }
        )
        out.content
        return out

