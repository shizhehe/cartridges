import asyncio
from cartridges.contexts.mcp.gmail import GmailMCPContext



async def main():
    context = GmailMCPContext.Config(
        email="eyuboglu@stanford.edu",
    )
    context = context.instantiate()
    await context.connect_to_server()
    out = await context.sample_subcontext()
    await context.exit_stack.aclose()
    return out

if __name__ == "__main__":
    out = asyncio.run(main())
    breakpoint()
