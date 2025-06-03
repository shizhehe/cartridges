from __future__ import annotations
from pydantic import BaseModel
from typing import Literal, Optional, Union

from cartridges.baselines.mtob import prompt_generic
from cartridges.context import TexDocument, TexChapter, TexSection
from cartridges.context import BaseContextConfig

from cartridges.context import StructuredContext, HTMLDocument, HTMLElement


class SimpleStructuredContext(StructuredContext):
    content: str

    @property
    def text(self) -> str:
        return self.content


class MTOBNoStructuredContext(BaseContextConfig):
    setup: Literal["latex_and_sentences", "medium_and_sentences"]

    def instantiate(self) -> StructuredContext:
        return SimpleStructuredContext(
            content=prompt_generic(
                grammar_book=(
                    "medium" if self.setup == "medium_and_sentences" else "latex"
                ),
                include_wordlist=False,
                include_sentences=True,
            )
        )


class MTOBStructuredContextConfig(BaseContextConfig):

    book_type: Literal["long", "medium", "full", "full_tex"] = "full_tex"

    def instantiate(self) -> StructuredContext:
        assert self.book_type == "full_tex", "TODO: Implement other book types"
        return KalamangContext.load()


TEMPLATE = """Textbook
{textbook}

---
Wordlist
{word_list}

---
Sentences
{sentences}
"""


class KalamangContext(StructuredContext):
    textbook: Union[TexDocument, str]
    word_list: list[KalamangWord]
    sentences: list[KalamangSentence]

    @property
    def text(self) -> str:
        return TEMPLATE.format(
            textbook=(
                self.textbook if isinstance(self.textbook, str) else self.textbook.text
            ),
            word_list="\n".join([word.text for word in self.word_list]),
            sentences="\n".join([sentence.text for sentence in self.sentences]),
        )

    @classmethod
    def load(cls) -> KalamangContext:
        from cartridges.tasks.mtob.load import load_mtob, MTOBData

        data: MTOBData = load_mtob()

        ctx = cls(
            textbook=TexDocument.from_string(data.grammar_book_tex),
            word_list=[
                KalamangWord(
                    kalamang_word=k,
                    english_word=e,
                    part_of_speech=pos,
                )
                for k, (pos, e) in data.wordlist_ke.items()
            ],
            sentences=[
                KalamangSentence(
                    english_sentence=example.original,
                    kalamang_sentence=example.translation,
                )
                for example in data.train_examples
            ],
        )
        return ctx


class KalamangWord(StructuredContext):
    kalamang_word: str
    english_word: str
    part_of_speech: Literal[
        "adj",
        "adv",
        "clf",
        "cnj",
        "dem",
        "dv",
        "gram",
        "gramm",
        "int",
        "n",
        "phrs",
        "pro",
        "q",
        "qnt",
        "v",
        "vi",
        "vt",
    ] = None


class KalamangSentence(StructuredContext):
    kalamang_sentence: str
    english_sentence: str
