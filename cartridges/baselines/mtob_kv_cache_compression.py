from typing import Literal
from kvpress import DuoAttentionPress, ExpectedAttentionPress
import pydrantic
import torch
import tqdm

from capsules.baselines.kv_press import ModelWithCompressedCache
from capsules.baselines.mtob import (
    prompt_generic,
    prompt_sentence_first,
    prompt_tell_task,
    user_prompt,
)
from capsules.data.mtob import load_test_ek, load_test_ke

from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate


class MtobKVCacheCompressionBaseline(pydrantic.RunConfig):
    model: str
    kv_compression: Literal["duo", "expected_attention"]
    kv_compression_ratio: float

    direction: Literal["ke", "ek"]
    grammar_book: Literal["full", "medium", "long", "latex"]
    include_wordlist: bool
    include_sentences: bool

    prompt_type: Literal["sentence_first", "tell_task", "generic"]

    temperature: float
    num_samples: int
    max_completion_tokens: int

    def run(self):
        with torch.no_grad():
            model = AutoModelForCausalLM.from_pretrained(
                self.model, device_map="auto", torch_dtype=torch.bfloat16
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model)

            press = (
                DuoAttentionPress(head_compression_ratio=self.kv_compression_ratio)
                if self.kv_compression == "duo"
                else ExpectedAttentionPress(compression_ratio=self.kv_compression_ratio)
            )

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

            assert (
                self.prompt_type != "sentence_first"
            ), "sentence_first is not supported for KV cache compression"

            if self.prompt_type == "tell_task":
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

            system_message = dict(role="system", content=prompt)

            prefix_tokens = tokenizer.apply_chat_template(
                [system_message], add_generation_prompt=False
            )

            compressed_model = ModelWithCompressedCache(
                press=press, model=model, tokenizer=tokenizer, prefix_tokens=prefix_tokens
            )

            results = []
            for source in tqdm.tqdm(sources, desc="Processing samples"):
                content = user_prompt(source_language, target_language, source)
                message_tokens = tokenizer.apply_chat_template(
                    [system_message, dict(role="user", content=content)],
                    add_generation_prompt=True,
                )
                question_tokens_only = message_tokens[len(prefix_tokens) :]
                answer = compressed_model.generate(
                    input_ids=question_tokens_only,
                    max_new_tokens=self.max_completion_tokens,
                )
                results.append(answer)

            chrf_metric = evaluate.load("chrf")

            result = chrf_metric.compute(predictions=results, references=ground_truths)
            breakpoint()
            print()
            print("iter done, score is", result["score"], flush=True)

            # results.append(result["score"])

            # print(f"{len(results)} results", results)
            # print("mean", sum(results) / len(results))
            # print("variance", statistics.variance(results) if len(results) > 1 else 0)
