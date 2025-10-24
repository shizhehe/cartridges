import asyncio
from cartridges.data.gmail.resources import GmailResource, LabelConfig

# this is for a personal email 
resource = GmailResource.Config(
    labels=None,
    date_start="2025/06/01",
    date_end="2025/09/12",
    date_days_in_bucket=30,
    search_query="from:Alun Roberts",
).instantiate()

async def main():
    await resource.setup()
    thread_content, prompts = await resource.sample_prompt(3)
    print(thread_content)
    print(prompts)


if __name__ == "__main__":
    asyncio.run(main())

