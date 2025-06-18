import asyncio
from typing import Optional
from contextlib import AsyncExitStack, ExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    async def connect_to_server(self):
        """Connect to an MCP server
        """
        server_params = StdioServerParameters(
            # **{
            #     "command": "docker",
            #     "args": [
            #     "run",
            #     "--rm",
            #     "-i",
            #     "--mount", "type=bind,src=/Users/sabrieyuboglu/code/cartridges,dst=/workspace",
            #     "mcp/git"
            #     ]
            # }
            **{
                "command": "docker",
                "args": [
                    "run",
                    "-i",
                    "--rm",
                    "-e",
                    "SLACK_BOT_TOKEN",
                    "-e",
                    "SLACK_TEAM_ID",
                    "-e",
                    "SLACK_CHANNEL_IDS",
                    "mcp/slack"
                ],
                "env": {
                    "SLACK_BOT_TOKEN": "",
                    "SLACK_TEAM_ID": "",
                }
            }
        )
         
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{ 
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]
        print(available_tools)

        out = await self.session.call_tool(
            name="slack_list_channels",
            arguments={
                "cursor": 0,
                "limit": 100,
                
            }
        )
        print(out)

        return str(out)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
        
    client = MCPClient()
    try:
        await client.connect_to_server()
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())