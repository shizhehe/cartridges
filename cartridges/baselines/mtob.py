import tqdm
from functools import cache
from typing import Literal
import pydrantic
import evaluate
import statistics

from capsules.clients.base import ClientConfig
from capsules.clients.tokasaurus_batch import TokasaurusBatchClient
from capsules.data.mtob import (
    load_test_ek,
    load_test_ke,
    load_train_examples,
    load_wordlist,
    load_book_long,
    load_book_medium,
    load_book_full,
    load_book_full_tex,
)

load_train_examples = cache(load_train_examples)
load_wordlist = cache(load_wordlist)
load_book_long = cache(load_book_long)
load_book_medium = cache(load_book_medium)
load_book_full = cache(load_book_full)
load_book_full_tex = cache(load_book_full_tex)


@cache
def parallel_sentences_str():
    return f"""Here is the collection of parallel sentences:

START OF PARALLEL SENTENCES
{"\n".join([f"{pair['original']}:{pair['translation']}" for pair in load_train_examples()])}
END OF PARALLEL SENTENCES

The collection of parallel sentences is now over."""


@cache
def wordlist_str():
    wordlist = load_wordlist()
    kalamang_to_english_wordlist = [
        f"{source}: {target}"
        for (source, (_part_of_speech, target)) in wordlist["ke"].items()
    ]
    english_to_kalamang_wordlist = [
        f"{source}: {target}" for (source, target) in wordlist["ek"].items()
    ]

    return f"""Now here is the bilingual word list:

START OF WORD LIST
{"\n".join(kalamang_to_english_wordlist + english_to_kalamang_wordlist)}
END OF WORD LIST

The bilingual word list is now over."""


def book_text(
    grammar_book: Literal["full", "medium", "long", "latex"],
):
    if grammar_book == "full":
        book_text = load_book_full_tex()
    elif grammar_book == "medium":
        book_text = load_book_medium()
    elif grammar_book == "long":
        book_text = load_book_long()
    elif grammar_book == "latex":
        book_text = load_book_full_tex()
    else:
        raise ValueError(
            f"Invalid grammar book type: {grammar_book}. Must be one of 'full', 'medium', 'long', or 'latex'."
        )

    return f"""{'Here is the book' if grammar_book in ('full', 'latex') else 'Here is a subset of the book'}, "A grammar of Kalamang":
START OF GRAMMAR BOOK
{book_text}
END OF GRAMMAR BOOK

The grammar book is now over."""


def user_prompt(
    source_language: str,
    target_language: str,
    input_sentence: str,
) -> str:
    return f"""Please translate the following sentence from {source_language} to {target_language}.

Translate this sentence: "{input_sentence}".

I understand that you may not be familiar enough with Kalamang to make a confident translation, but please give your best guess.
Respond with only the translation and no other text."""


def what_is_given(
    include_wordlist: bool,
    include_sentences: bool,
) -> str:
    if include_sentences and include_wordlist:
        return "You will be given a field linguistics grammar book, a bilingual word list, and a collection of parallel English/Kalamang sentences."
    elif include_sentences:
        return "You will be given a field linguistics grammar book and a collection of parallel sentences"
    elif include_wordlist:
        return "You will be given a field linguistics grammar book and a bilingual word list"
    else:
        return "You will be given a field linguistics grammar book."


# def prompt_sentence_first(
#     grammar_book: Literal["full", "medium", "long", "latex"],
#     source_language: str,
#     target_language: str,
#     input_sentence: str,
#     include_wordlist: bool,
#     include_sentences: bool,
# ) -> str:

#     return f"""You are tasked with translating the following sentence from {source_language} to {target_language}: "{input_sentence}".
# {what_is_given(include_wordlist, include_sentences)}

# {book_text(grammar_book)}

# {f"""Remember that you are tasked with translating the following sentence from {source_language} to {target_language}: "{input_sentence}".
# {wordlist_str()}

# """ if include_wordlist else ''}
# {f"""Remember that you are tasked with translating the following sentence from {source_language} to {target_language}: "{input_sentence}".
# {parallel_sentences_str()}

# """ if include_sentences else ''}."""


def prompt_sentence_first(
    grammar_book: Literal["full", "medium", "long", "latex"],
    source_language: str,
    target_language: str,
    input_sentence: str,
    include_wordlist: bool,
    include_sentences: bool,
) -> str:
    breakpoint()
    return f"""You are tasked with translating between English and Kalamang.
{f"""You are tasked with translating the following sentence from {source_language} to {target_language}: "{input_sentence}".""" if True else ''}
{what_is_given(include_wordlist, include_sentences)}

{book_text(grammar_book)}

{f"""Remember that you are tasked with translating between English and Kalamang.
{wordlist_str()}

""" if include_wordlist else ''}
{f"""Remember that you are tasked with translating between English and Kalamang.
{parallel_sentences_str()}

""" if include_sentences else ''}."""


def prompt_tell_task(
    grammar_book: Literal["full", "medium", "long", "latex"],
    source_language: str,
    target_language: str,
    include_wordlist: bool,
    include_sentences: bool,
) -> str:

    return f"""You are tasked with translating between English and Kalamang.
{what_is_given(include_wordlist, include_sentences)}

{book_text(grammar_book)}

{f"""Remember that you are tasked with translating between English and Kalamang.
{wordlist_str()}

""" if include_wordlist else ''}
{f"""Remember that you are tasked with translating between English and Kalamang.
{parallel_sentences_str()}

""" if include_sentences else ''}."""


def prompt_generic(
    grammar_book: Literal["full", "medium", "long", "latex"],
    include_wordlist: bool,
    include_sentences: bool,
) -> str:
    return f"""{what_is_given(include_wordlist, include_sentences)}

{book_text(grammar_book)}

{f"""{wordlist_str()}

""" if include_wordlist else ''}
{f"""{parallel_sentences_str()}

""" if include_sentences else ''}."""


class MtobBaselineConfig(pydrantic.RunConfig):
    client: ClientConfig

    direction: Literal["ke", "ek"]
    grammar_book: Literal["full", "medium", "long", "latex"]
    include_wordlist: bool
    include_sentences: bool

    prompt_type: Literal["sentence_first", "tell_task", "generic"]

    temperature: float
    num_samples: int
    max_completion_tokens: int

    def run(self):

        if self.direction == "ke":
            source_language = "Kalamang"
            target_language = "English"

            load_fn = load_test_ke

        elif self.direction == "ek":
            source_language = "English"
            target_language = "Kalamang"

            load_fn = load_test_ek
        else:
            raise ValueError(
                f"Invalid direction: {self.direction}. Must be 'ke' or 'ek'."
            )

        sources = []
        ground_truths = []

        data = load_fn()
        for item in data:
            sources.append(item["original"])
            ground_truths.append(item["ground_truth"])

        chats = []
        for source in sources:
            if self.prompt_type == "sentence_first":
                prompt = prompt_sentence_first(
                    self.grammar_book,
                    source_language,
                    target_language,
                    source,
                    self.include_wordlist,
                    self.include_sentences,
                )
            elif self.prompt_type == "tell_task":
                prompt = prompt_tell_task(
                    self.grammar_book,
                    source_language,
                    target_language,
                    self.include_wordlist,
                    self.include_sentences,
                )
            elif self.prompt_type == "generic":
                prompt = prompt_generic(
                    self.grammar_book,
                    self.include_wordlist,
                    self.include_sentences,
                )
            else:
                raise ValueError(f"Invalid prompt type: {self.prompt_type}.")

            chats.append(
                [
                    dict(role="system", content=prompt),
                    dict(
                        role="user",
                        content=user_prompt(source_language, target_language, source),
                    ),
                ]
            )

        client = self.client.instantiate()

        results = []
        for _ in tqdm.tqdm(range(self.num_samples)):
            if isinstance(client, TokasaurusBatchClient):
                response = client.chat_with_logprobs(
                    chats=chats,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_completion_tokens,
                )
                data = [i.assistant_text for i in response]
            else:
                response = client.chat(
                    chats=chats,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_completion_tokens,
                )
                data = [i.text for i in response.samples]
            chrf_metric = evaluate.load("chrf")

            result = chrf_metric.compute(predictions=data, references=ground_truths)
            print()
            print("iter done, score is", result['score'], flush=True)

            results.append(result['score'])


        print(f"{len(results)} results", results)
        print("mean", sum(results) / len(results))
        print("variance", statistics.variance(results) if len(results) > 1 else 0)
