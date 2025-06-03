from pathlib import Path
from capsules.context import BaseContextConfig, TexDocument
from capsules.tasks.mtob.context import SimpleStructuredContext


class HydragenStructuredContext(BaseContextConfig):

    def instantiate(self):
        return SimpleStructuredContext(
            content=(Path(__file__).parent / "hydragen_content.txt").read_text()
        )


class MonkeysStructuredContext(BaseContextConfig):

    def instantiate(self):
        return SimpleStructuredContext(
            content=(Path(__file__).parent / "monkeys_content.txt").read_text()
        )

