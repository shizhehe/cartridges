from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
import asyncio
import itertools
import math
import time
from typing import Callable, Dict, List, Literal, Optional, Union

# from kvpress import DuoAttentionPress, ExpectedAttentionPress
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
import wandb
import pydrantic

from cartridges.data.resources import Resource
from cartridges.datasets import CartridgeGenerateDatasetElement, CartridgeGenerateDataset
from cartridges.clients.base import Client, ClientConfig, ClientResponse

from cartridges.train import GenerationEvalConfig
from cartridges.utils import WandBConfig, prepare_wandb, seed_everything, get_logger
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


logger = get_logger(__name__)


class EvaluateConfig(pydrantic.RunConfig):
    name: str = "default"  # A name for the run for wandb
    generator: BaselineGenerator.Config
    wandb: Optional[WandBConfig] = None

    # dataset for actually producing generations
    eval: GenerationEvalConfig

    batch_size: int
    max_num_batches_in_parallel: int = 1
    parallelism_strategy: Literal["thread", "process"] = "thread"

    tokenizer: str = "meta-llama/Llama-3.2-1B-Instruct"
    device: str = "cuda"

    seed: int = 42

    def run(self):
        return asyncio.run(evaluate_generation(self))


async def evaluate_generation(config: EvaluateConfig):
    seed_everything(config.seed)

    logger.info(f"ICL will be saved to {config.run_dir}")
    logger.info("Initializing tokenizer and dataset")
    # download_wandb_artifacts(config)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    
    dataset=config.eval.dataset.instantiate(tokenizer=tokenizer, seed=config.seed)
    generator = config.generator.instantiate()

    if config.wandb is not None:
        config.wandb.name = config.name
        prepare_wandb(config.wandb, config.to_dict())

    dataset_batch_size = config.batch_size // config.eval.num_samples
    total_batches = math.ceil(len(dataset) / dataset_batch_size)
    all_rows = []

    # TODO: the generate dataset actually determines the temperature for training
    # so to ensure fair comparison we assert that the temperature is the same
    print("config.eval.temperature: ", config.eval.temperature)
    print("config.generator.temperature: ", config.generator.temperature)
    assert config.eval.temperature == config.generator.temperature

    # Use asyncio for concurrent execution
    tasks = [
        _process_batch(
            batch_start=batch_idx * dataset_batch_size,
            batch_end=min(
                (batch_idx + 1) * dataset_batch_size, len(dataset)
            ),
            generator=generator,
            dataset=dataset,
            eval_config=config.eval,
        )
        for batch_idx in range(total_batches)
    ]
    
    # Process in chunks to limit concurrency
    chunk_size = config.max_num_batches_in_parallel
    for i in tqdm(range(0, len(tasks), chunk_size), desc="Processing batch chunks"):
        chunk_tasks = tasks[i:i + chunk_size]
        batch_results = await asyncio.gather(*chunk_tasks)
        for batch_rows in batch_results:
            all_rows += batch_rows

    prefix = f"generate_{config.eval.name_for_wandb}"
    df = pd.DataFrame(all_rows)
    score_cols = [col for col in df.columns if col.endswith("score")]
    avg_scores = {f"{prefix}/{col}": df[col].mean() for col in score_cols}

    if hasattr(dataset, "batch_score_with_answers"):
        batch_score = dataset.batch_score_with_answers(
            df["pred"].tolist(), df["answer"].tolist()
        )
        if isinstance(batch_score, dict):
            batch_score = {f"{prefix}/{k}": v for k, v in batch_score.items()}
        else:
            batch_score = {f"{prefix}/batch_score": batch_score}
        avg_scores.update(batch_score)

    if hasattr(generator, "kv_cache_size_bytes"):
        avg_scores[f"{prefix}/kv_cache_size_bytes"] = generator.kv_cache_size_bytes

    if config.wandb is not None:
        wandb.log(
            {
                **avg_scores,
                f"{prefix}/num_system_and_user_tokens": df[
                    "num_system_and_user_tokens"
                ].mean(),
                f"{prefix}/num_assistant_tokens": df["num_assistant_tokens"].mean(),
                f"{prefix}/table": df,
                f"{prefix}/num_samples": len(df),
            },
            step=0,
        )
        wandb.finish()


async def _process_batch(
    batch_start: int,
    batch_end: int,
    generator: BaselineGenerator,
    dataset: CartridgeGenerateDataset,
    eval_config: GenerationEvalConfig,
):
    num_samples = eval_config.num_samples
    has_score = hasattr(dataset, "score")
    elems = [dataset[elem_idx] for elem_idx in range(batch_start, batch_end)]
    sample_idxs = sum([[i] * len(elems) for i in range(num_samples)], [])
    elems = elems * num_samples
    responses: List[GenerateBaselineResponse] = await generator.generate(elems)

    results = []
    for response, element, sample_idx in zip(responses, elems, sample_idxs):
        if has_score:
            metrics, extras = dataset.score(
                pred=response.text, answer=element.answer, convo_id=element.convo_id
            )
        else:
            metrics, extras = None, {}

        if not isinstance(metrics, dict):
            # TODO: Support for older datasets that return a single bool or float as metrics
            metrics = {"score": metrics}
        else:
            metrics = {f"{k}_score": v for k, v in metrics.items()}

        results.append(
            {
                "prompt": element.prompt,
                "answer": element.answer,
                "pred": response.text,
                "num_system_and_user_tokens": response.num_system_and_user_tokens,
                "num_assistant_tokens": response.num_assistant_tokens,
                "prompt_messages": response.prompt_messages,
                "convo_id": element.convo_id,
                "sample_idx": sample_idx,
                **metrics,
                **element.metadata,
                **extras,
            }
        )
    return results


@dataclass
class GenerateBaselineResponse:
    prompt_messages: List[Dict[str, str]]
    num_system_and_user_tokens: int
    num_assistant_tokens: int
    text: str


class BaselineGenerator(ABC):

    class Config(pydrantic.ObjectConfig):
        _pass_as_config: bool = True

    def __init__(self, config: Config):
        self.config = config

    async def generate(
        self, elements: List[CartridgeGenerateDatasetElement]
    ) -> List[GenerateBaselineResponse]:
        raise NotImplementedError()


class ICLBaseline(BaselineGenerator):

    class Config(BaselineGenerator.Config):
        client: ClientConfig
        temperature: float = 0.0
        # used to count number of tokens in the prompt
        tokenizer: str = "meta-llama/Llama-3.2-3B-Instruct"

        # The user prompt template which should contain {content}
        user_prompt_template: str = "{content}"

        context: Union[str, Resource.Config]

        # The system prompt template which should contain {title} and {content}
        # variables.
        system_prompt_template: str = "{title}\n\n{content}"
        max_completion_tokens: int = 384
        max_context_tokens: Optional[int] = None  # will truncate if longer
        enable_thinking: Optional[bool] = None

        log_system_prompt: bool = False

    def __init__(self, config: Config):
        self.config = config
        self.client = config.client.instantiate()

        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)


        if isinstance(self.config.context, str):
            ctx_text = self.config.context
        else:
            resource = self.config.context.instantiate()
            # TODO (SE): Need to properly call the resource setup!
            ctx_text = resource.to_string()


        if self.config.max_context_tokens is not None:
            ctx_text = self.tokenizer.decode(
                self.tokenizer.encode(ctx_text)[: self.config.max_context_tokens],
                add_special_tokens=False,

                # suppresses the truncation error 
                max_length=999_999_999, 
                truncation=True
                
            )

        if config.system_prompt_template is not None:
            system_prompt = config.system_prompt_template.format(
                content=ctx_text,
            )

            self.system_prompt = self.post_process_system_prompt(
                system_prompt,
            )
        else:
            self.system_prompt = None

       
        self.metadata = {}

    def post_process_system_prompt(self, system_prompt: str) -> str:
        return system_prompt

    async def generate(
        self, elements: List[CartridgeGenerateDatasetElement]
    ) -> List[GenerateBaselineResponse]:

        chats = []
        for element in elements:
            messages = []
            if self.system_prompt is not None:
                messages.append(
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                )
            messages.append(
                {
                    "role": "user",
                    "content": self.config.user_prompt_template.format(
                        content=element.prompt
                    ),
                },
            )
            chats.append(messages)


        response: ClientResponse = await self.client.chat(
            chats=chats,
            max_completion_tokens=self.config.max_completion_tokens,
            temperature=self.config.temperature,
            enable_thinking=self.config.enable_thinking,
        )
        assert len(response.samples) == len(chats)

        results = []
        for sample, messages, element in zip(response.samples, chats, elements):
            num_prompt_tokens = len(
                self.tokenizer.apply_chat_template(
                    messages,
                )
            )
            num_assistant_tokens = len(self.tokenizer.encode(sample.text))
            log_messages = [
                msg
                for msg in messages
                if (msg["role"] != "system" or self.config.log_system_prompt)
            ]

            results.append(
                GenerateBaselineResponse(
                    prompt_messages=log_messages,
                    num_system_and_user_tokens=num_prompt_tokens,
                    num_assistant_tokens=num_assistant_tokens,
                    text=sample.text,
                )
            )
        return results

class ICLBaselineFirstKTokens(ICLBaseline):
    class Config(ICLBaseline.Config):
        first_k_tokens: int = 1024

    def post_process_system_prompt(self, system_prompt: str) -> str:
        # only keep the first k tokens
        system_prompt_tokens = self.tokenizer.encode(system_prompt)
        target_tokens = math.floor(
            self.config.frac_of_tokens * len(system_prompt_tokens)
        )
        system_prompt_tokens = system_prompt_tokens[: target_tokens - 1] + [
            system_prompt_tokens[-1]
        ]
        return self.tokenizer.decode(system_prompt_tokens)
class ICLBaselineFracTokens(ICLBaseline):
    class Config(ICLBaseline.Config):
        frac_of_tokens: float = 0.5

    def post_process_system_prompt(self, system_prompt: str) -> str:
        # only keep the first k tokens
        system_prompt_tokens = self.tokenizer.encode(system_prompt)
        target_tokens = math.floor(
            self.config.frac_of_tokens * len(system_prompt_tokens)
        )
        system_prompt_tokens = system_prompt_tokens[: target_tokens - 1] + [
            system_prompt_tokens[-1]
        ]
        return self.tokenizer.decode(system_prompt_tokens)


SUMMARY_SYSTEM_PROMPT = """You are a helpful assistant that is highly skilled at writing accurate and clear summaries of long documents."""
SUMMARY_USER_PROMPT = """Please summarize the following text in roughly {word_num} words:\n\n{content}"""
MAX_CONTEXT_LENGTH = 110000

class ICLBaselineSummaryFromLargeModel(ICLBaseline):
    class Config(ICLBaseline.Config):
        summary_tokens: int = 1024
        summary_client: ClientConfig

    def __init__(self, config: Config, context: Context):
        self.summary_client = config.summary_client.instantiate()
        self.context = context
        # Don't call super().__init__ yet, we'll initialize system prompt separately
        self.config = config
        self.client = config.client.instantiate()
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        self.metadata = {}
        # System prompt will be set in async_init

    async def async_init(self):
        """Initialize the system prompt asynchronously"""
        if isinstance(self.config.context, str):
            ctx_text = self.config.context
        else:
            ctx_text = self.config.context()

        if self.config.max_context_tokens is not None:
            ctx_text = self.tokenizer.decode(
                self.tokenizer.encode(ctx_text)[: self.config.max_context_tokens],
                add_special_tokens=False,
                max_length=999_999_999, 
                truncation=True
            )

        if self.config.system_prompt_template is not None:
            system_prompt = self.config.system_prompt_template.format(
                content=ctx_text,
            )
            self.system_prompt = await self.post_process_system_prompt(system_prompt)
        else:
            self.system_prompt = None
    
    async def post_process_system_prompt(self, system_prompt):
        # decide if we need to chunk the context
        context_tokens = self.tokenizer.encode(self.context.text)
        if len(context_tokens) > MAX_CONTEXT_LENGTH:
            # chunk the context
            num_chunks = math.ceil(len(context_tokens) / MAX_CONTEXT_LENGTH)
            chunk_size = math.ceil(len(context_tokens) / num_chunks)
            print("chunk size: ", chunk_size)
            chunks = [
                self.tokenizer.decode(
                    context_tokens[i : i + chunk_size], skip_special_tokens=True
                )
                for i in range(0, len(context_tokens), chunk_size)
            ]
        else: 
            chunks = [self.context.text]
        # format chats
       
        chats = []
        for chunk in chunks:
            chats.append(
                [
                    {
                        "role": "system",
                        "content": SUMMARY_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": SUMMARY_USER_PROMPT.format(
                            word_num=int(self.config.summary_tokens * 0.4),
                            content=chunk,
                        ),
                    },
                ]
            )

        # ping client to get summary
        response: ClientResponse = await self.summary_client.chat(
            chats=chats,
            max_completion_tokens=self.config.summary_tokens,
            temperature=self.config.temperature,
        )

        summary = "\n".join([sample.text for sample in response.samples])
        print("summary: ", summary)
        return self.config.system_prompt_template.format(
            content=summary,
        )
        
    async def generate(
        self, elements: List[CartridgeGenerateDatasetElement]
    ) -> List[GenerateBaselineResponse]:
        # Ensure async initialization is complete
        await self.async_init()
        return await super().generate(elements)



class KVCacheCompressionBaseline(BaselineGenerator):
    class Config(BaselineGenerator.Config):
        kv_compression: str
        kv_compression_ratio: float

        # used to count number of tokens in the prompt
        model: str

        # The system prompt template which should contain {title} and {content}
        # variables.
        max_completion_tokens: int = 64

        log_system_prompt: bool = False
        temperature: float = 0.0

    def __init__(self, config: Config, context: Context):
        self.config = config
        assert (
            config.temperature == 0.0
        ), "Temperature must be 0.0 for KVCacheCompressionBaseline"

        with torch.no_grad():

            self.model = AutoModelForCausalLM.from_pretrained(
                config.model, device_map="cuda", torch_dtype=torch.bfloat16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(config.model)

            press = make_press(config.kv_compression, config.kv_compression_ratio)

            self.system_message = dict(role="system", content=context.to_string())
            prefix_tokens = self.tokenizer.apply_chat_template(
                [self.system_message], add_generation_prompt=False
            )

            self.compressed_model = ModelWithCompressedCache(
                press=press,
                model=self.model,
                tokenizer=self.tokenizer,
                prefix_tokens=prefix_tokens,
            )

            self.kv_cache_size_bytes = self.compressed_model.kv_size_bytes()

    def generate(
        self, elements: List[CapsuleGenerateDatasetElement]
    ) -> List[GenerateBaselineResponse]:
        with torch.no_grad():
            results = []
            for source in tqdm(elements, desc="Processing samples"):
                (input_ids,) = source.input_ids.tolist()

                # message_tokens = self.tokenizer.apply_chat_template(
                #     [self.system_message, dict(role="user", content=content)],
                #     add_generation_prompt=True,
                # )
                # question_tokens_only = message_tokens[len(prefix_tokens) :]
                answer = self.compressed_model.generate(
                    input_ids=input_ids,
                    max_new_tokens=self.config.max_completion_tokens,
                )
                results.append(
                    GenerateBaselineResponse(
                        text=answer,
                        # TODO: fix this to report correct token counts 
                        num_system_and_user_tokens=0,
                        num_assistant_tokens=0,
                        prompt_messages=[],
                    )
                )
            return results

class CartridgeConfig(pydrantic.BaseConfig):
    id: str
    source: str
    force_redownload: bool = False

class CartridgeBaseline(BaselineGenerator):

    class Config(BaselineGenerator.Config):
        client: ClientConfig
        temperature: float = 0.0
        # used to count number of tokens in the prompt
        tokenizer: str = "meta-llama/Llama-3.2-3B-Instruct"
        
        cartridges: Optional[List[CartridgeConfig]] = None


        # The user prompt template which should contain {content}
        user_prompt_template: str = "{content}"

        context: Union[str, Resource.Config]

        # The system prompt template which should contain {title} and {content}
        # variables.
        max_completion_tokens: int = 384
        max_context_tokens: Optional[int] = None  # will truncate if longer
        enable_thinking: Optional[bool] = None

        log_system_prompt: bool = False

    def __init__(self, config: Config):
        self.config = config
        self.client = config.client.instantiate()

        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)


        if isinstance(self.config.context, str):
            ctx_text = self.config.context
        else:
            resource = self.config.context.instantiate()
            # TODO (SE): Need to properly call the resource setup!
            ctx_text = resource.to_string()


        if self.config.max_context_tokens is not None:
            ctx_text = self.tokenizer.decode(
                self.tokenizer.encode(ctx_text)[: self.config.max_context_tokens],
                add_special_tokens=False,

                # suppresses the truncation error 
                max_length=999_999_999, 
                truncation=True
                
            )

       
        self.metadata = {}

    async def generate(
        self, elements: List[CartridgeGenerateDatasetElement]
    ) -> List[GenerateBaselineResponse]:
        print("generating with cartridges: ", self.config.cartridges)

        chats = []
        for element in elements:
            messages = []

            messages.append(
                {
                    "role": "user",
                    "content": self.config.user_prompt_template.format(
                        content=element.prompt
                    ),
                },
            )
            chats.append(messages)

        kwargs = {}
        if self.config.cartridges is not None:
            kwargs["cartridges"] = [cartridge.model_dump() for cartridge in self.config.cartridges]
        print("kwargs: ", kwargs)


        response: ClientResponse = await self.client.chat(
            chats=chats,
            max_completion_tokens=self.config.max_completion_tokens,
            temperature=self.config.temperature,
            enable_thinking=self.config.enable_thinking,
            **kwargs,
        )
        assert len(response.samples) == len(chats)

        t0 = time.time()
        results = []
        for sample, messages, element in zip(response.samples, chats, elements):
            num_prompt_tokens = len(
                self.tokenizer.apply_chat_template(
                    messages,
                )
            )
            num_assistant_tokens = len(self.tokenizer.encode(sample.text))
            log_messages = [
                msg
                for msg in messages
                if (msg["role"] != "system" or self.config.log_system_prompt)
            ]

            results.append(
                GenerateBaselineResponse(
                    prompt_messages=log_messages,
                    num_system_and_user_tokens=num_prompt_tokens,
                    num_assistant_tokens=num_assistant_tokens,
                    text=sample.text,
                )
            )
        print(f"Time taken: {time.time() - t0} seconds")
        return results