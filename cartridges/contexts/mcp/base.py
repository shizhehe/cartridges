import abc
import asyncio
from contextlib import AsyncExitStack
import random
from typing import Dict, List, Optional


from mcp import ClientSession, StdioServerParameters, stdio_client
from cartridges.context import Context, BaseContextConfig
from pydrantic import ObjectConfig


class MCPContext(Context):

    class Config(ObjectConfig, BaseContextConfig):
        _pass_as_config = True
    

    def __init__(self, config: Config, command: str, args: List[str], env: Dict[str, str]):
        self.config = config
        self.session: Optional[ClientSession] = None
        self.command = command
        self.args = args
        self.env = env
        self.exit_stack = AsyncExitStack()


    async def connect_to_server(self):
        """Connect to an MCP server
        """

        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env
        )
         
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    @abc.abstractmethod
    async def sample_subcontext(self):
        if self.session is None:
            raise RuntimeError("Not connected to an MCP server. You must call `connect_to_server` first.")
        raise NotImplementedError


class CompoundMCPContext(MCPContext):

    class Config(BaseContextConfig):
        contexts: List[MCPContext.Config]
    
    def __init__(self, config: Config):
        self.contexts = [ctx.instantiate() for ctx in config.contexts]
        
    async def connect_to_server(self):
        await asyncio.gather(*(ctx.connect_to_server() for ctx in self.contexts))

    async def sample_subcontext(self):
        return random.choice(self.contexts).sample_subcontext()
        