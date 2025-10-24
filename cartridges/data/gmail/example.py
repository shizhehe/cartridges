import asyncio
from cartridges.data.gmail.resources import GmailResource


resource = GmailResource.Config(
    labels=[
        "categories--stanford--primary--stanford-", 
        "categories--stanford--updates--stanford-"
    ],
    date_start="2025/09/01",
    date_end="2025/09/14",
).instantiate()


async def main():
    await resource.setup()

    print(await resource.sample_prompt(1))

if __name__ == "__main__":
    asyncio.run(main())

