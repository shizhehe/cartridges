import abc
from collections import defaultdict
from dataclasses import asdict
from inspect import getsource
from itertools import tee
import json
import os
from platform import release
import re
import random
import concurrent.futures
from textwrap import dedent
import time
from typing import Callable, List, Literal, Optional, Set, Tuple

import numpy as np
from pydrantic import ObjectConfig
from transformers import AutoTokenizer

from cartridges.context import StructuredContext, list_nested_contexts
from cartridges.structs import TrainingExample
from cartridges.clients.base import Client, ClientConfig, ClientSample
from cartridges.synthesizers.base import ConvoSynthesizer
from cartridges.synthesizers.outline import get_outline
from cartridges.tools.base import Tool
from cartridges.utils import disk_cache, get_logger

logger = get_logger(__name__)


TOOL_PROMPT_TEMPLATE = """You need to respond to the following message:

--- begin message ---
{message}
--- end message ---

You can take one of the following actions to study the corpus:

--- begin tools ---
{tools}
--- end tools ---

Please respond with a JSON object containing the following fields:
```json
{{
    "tool": "tool-name",
    "kwargs": {{
        "arg-name": "arg-value",
    }}
}}
```
"""

SYSTEM_PROMPT_TEMPLATE = """You are in a conversation about the corpus of information below. Note that some of the corpus has been omitted to fit within the context window.
When you talk about the passage, do not start your responses with "According to the passage", "In the passage", or any other similar phrase. Just start with the content of your response.

<corpus>
{subcorpus}
</corpus>
"""


class PromptSampler(abc.ABC):

    class Config(ObjectConfig):
        _pass_as_config = True

    @abc.abstractmethod
    def __call__(
        self,
        batch_idx: int,
        num_convos: int,
    ) -> List[tuple[str, str]]:
        """
        Returns a system_prompt and a list of seed prompts
        """
        raise NotImplementedError()


class SelfStudySynthesizer(ConvoSynthesizer):

    class Config(ConvoSynthesizer.Config):
        client: ClientConfig

        tools: List[Tool.Config]
        use_tools_a: bool = False
        use_tools_b: bool = False
        max_tool_tokens: int = 128

        # `prompt_sampler` is responsible for sampling the initial system prompt and the seed prompts
        # We combined them together since some strategies might involve dependencies
        # between the two
        prompt_sampler: PromptSampler.Config
        system_prompt_template: str = SYSTEM_PROMPT_TEMPLATE

        max_rounds: int = 1

        tool_prompt_template_a: str = TOOL_PROMPT_TEMPLATE
        temperature_a: float = 0.6
        max_completion_tokens_a: int = 512
        prob_cot_a: float = 0.0

        tool_prompt_template_b: str = TOOL_PROMPT_TEMPLATE
        temperature_b: float = 0.0
        max_completion_tokens_b: int = 1024

        tokenizer: str

        num_top_logprobs: int = 20

        use_tools: Optional[bool] = None  # DEPRECATED: Here for backwards compatibility

    def __init__(self, config: Config, context: StructuredContext):
        self.config = config

        if self.config.use_tools is not None:
            logger.warning("use_tools is deprecated. Use use_tools_a and use_tools_b instead. Setting use_tools_a = use_tools and use_tools_b = use_tools")
            self.config.use_tools_a = self.config.use_tools
            self.config.use_tools_b = self.config.use_tools

        self.context = context
        self.client = self.config.client.instantiate()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
        self.tools: List[Tool] = [
            tool.instantiate(
                context=self.context,
                tokenizer=self.tokenizer,
            )
            for tool in self.config.tools
        ]
        self.tools = {tool.name: tool for tool in self.tools}

        self.prompt_sampler = self.config.prompt_sampler.instantiate(
            context=self.context,
            tokenizer=self.tokenizer,
            client=self.client,
        )
        random.seed(82)

    def sample_convos(
        self, batch_idx: int, num_convos: int, total_batches: int
    ) -> list[TrainingExample]:
        # (1) Get initial system prompt and seed prompts
        # --- begin prompt sampling ---
        t0 = time.time()
        subcorpus, seed_prompts = self.prompt_sampler(batch_idx, num_convos)
        initial_system_prompt = self.config.system_prompt_template.format(
            subcorpus=subcorpus
        )
        assert len(seed_prompts) == num_convos
        logger.info(f"Prompt sampling took {time.time() - t0} seconds")
        # --- end prompt sampling ---

        # (2) Initialize convos
        # --- begin initialization of convos ---
        t0 = time.time()
        convos: List[List[dict]] = [[] for _ in range(num_convos)]
        contexts: List[str] = [initial_system_prompt] * num_convos
        metas: List[dict] = [
            {
                "tool_calls": [],
                "seed_prompt": seed_prompt,
                "initial_system_prompt": initial_system_prompt,
            }
            for seed_prompt in seed_prompts
        ]
        logger.info(f"Initialization of convos took {time.time() - t0} seconds")
        # --- end initialization of convos ---
        # (3) Generate convos
        for round_idx in range(self.config.max_rounds):

            # (3.1) bot_a requests new content to be added to the context
            # --- begin bot A tool usage ---
            if self.config.use_tools_b:
                t0 = time.time()

                tool_resps: List[str] = self._get_content_via_tool(
                    prompt_template=self.config.tool_prompt_template_a,
                    convos=[
                        [system(ctx), user(seed), *flip_roles(convo)]
                        for ctx, seed, convo in zip(contexts, seed_prompts, convos)
                    ],
                    metas=metas,
                    contexts=contexts,
                )
                contexts = [ctx + resp for ctx, resp in zip(contexts, tool_resps)]
                logger.info(
                    f"Round {round_idx}: Bot A tool usage (select + apply) took {time.time() - t0} seconds"
                )
            # --- end bot A tool usage ---

            # (3.2) With new information in context, generate user message
            # --- begin bot A response generation ---
            t0 = time.time()
            resps = self.client.chat(
                [
                    [system(ctx), user(seed), *flip_roles(convo)]
                    for ctx, seed, convo in zip(contexts, seed_prompts, convos)
                ],
                temperature=self.config.temperature_a,
                max_completion_tokens=self.config.max_completion_tokens_a,
            ).samples
            convos = [
                convo
                + [
                    user(
                        resp.output_text,
                        cot=random.random() < self.config.prob_cot_a,
                    )
                ]
                for convo, resp in zip(convos, resps)
            ]
            logger.info(
                f"Round {round_idx}: Bot A response generation took {time.time() - t0} seconds"
            )
            # --- end bot A response generation ---

            # (3.3) bot_b requests new content to be added to the context
            # --- begin bot B tool usage ---
            if self.config.use_tools_b:
                t0 = time.time()
                tool_resps: List[str] = self._get_content_via_tool(
                    prompt_template=self.config.tool_prompt_template_b,
                    convos=[
                        [system(ctx), *convo] for ctx, convo in zip(contexts, convos)
                    ],
                    metas=metas,
                    contexts=contexts,
                )
                contexts = [ctx + resp for ctx, resp in zip(contexts, tool_resps)]
                logger.info(
                    f"Round {round_idx}: Bot B tool usage (select + apply) took {time.time() - t0} seconds"
                )
            # --- end bot B tool usage ---

            # (3.4) bot_b generates a response
            # --- begin bot B response generation ---
            t0 = time.time()
            resps = self.client.chat(
                [[system(ctx), *convo] for ctx, convo in zip(contexts, convos)],
                temperature=self.config.temperature_b,
                top_logprobs=self.config.num_top_logprobs,
                logprobs_start_message=1,  # do not include logprobs for the system prompt
                max_completion_tokens=self.config.max_completion_tokens_b,
            ).samples
            convos = [
                convo + [assistant(resp.output_text)]
                for convo, resp in zip(convos, resps)
            ]
            logger.info(
                f"Round {round_idx}: Bot B response generation took {time.time() - t0} seconds"
            )
            # --- end bot B response generation ---

        # (4) Convert responses and chats to training examples
        # --- begin conversion to training examples ---
        t0 = time.time()
        examples = self._responses_and_chats_to_training_examples(
            samples=resps,
            convos=convos,
            metas=metas,
            contexts=contexts,
        )
        logger.info(f"Conversion to training examples took {time.time() - t0} seconds")
        # --- end conversion to training examples ---
        return examples

    def _get_content_via_tool(
        self,
        convos: list[list[dict]],
        prompt_template: str,
        metas: list[dict],
        contexts: list[str],
    ):
        # (1) Build a string describing all of the available tools and their arguments
        # --- begin tool string ---
        tool_str = ""
        for name, tool in self.tools.items():
            tool_str += (
                f"{name}: {tool.description()}\n"
                f"Arguments:\n"
                f"{getsource(tool.ToolInput).split('\n')[1:]}"
            )
        assert convos[0][-1]["role"] == "user"
        # --- end tool string ---

        # (2) Query the model to pick a tool and set its arguments
        # --- begin tool selection ---
        t0 = time.time()
        resps = self.client.chat(
            [
                # we funk with the last user message to add the tool prompt
                convo[:-1]
                + [user(prompt_template.format(tools=tool_str, message=convo[-1]))]
                for convo in convos
            ],
            temperature=self.config.temperature_a,
            max_completion_tokens=self.config.max_tool_tokens,
        ).samples
        reqs = [resp.output_text for resp in resps]
        logger.info(f"Tool selection took {time.time() - t0} seconds")
        # --- end tool selection ---

        # (3) Parse the tool responses and apply the tool. If it fails, just return empty string
        # --- begin tool application ---
        t0 = time.time()
        tool_responses = []

        # (3.1) Group requests by tool 
        # --- begin tool grouping ---
        tool_to_reqs = defaultdict(list)
        for idx, (req, meta) in enumerate(zip(reqs, metas, strict=True)):
            try:
                json_str = re.search(r"\{.*\}", req, re.DOTALL).group(0)
                spec = json.loads(json_str)
                tool_obj = self.tools[spec["tool"]]
                tool_to_reqs[spec["tool"]].append({
                    "idx": idx,
                    "spec": spec["kwargs"],
                    "tool_obj": tool_obj,
                    "input": tool_obj.ToolInput(**spec["kwargs"]),
                    "raw_request": req,
                })

            except Exception as e:
                logger.info(f"Error parsing tool request: {type(e).__name__}: {e}")
                tool_to_reqs["__failure__"].append(
                    {
                        "idx": idx,
                        "raw_request": req,
                        "error": str(e),
                    }
                )
        # --- end tool grouping ---

        # (3.2) Apply the tool in batch
        # --- begin applying tool in groups ---
        results = [None] * len(reqs)
        for tool, curr_reqs in tool_to_reqs.items():
            if tool == "__failure__":
                for req in curr_reqs:
                    results[req["idx"]] = {
                        "success": False,
                        "raw_request": req["raw_request"],
                        "error": req["error"],
                        "tool": tool,
                        "input": None,
                        "tool_response": None,
                    }
            else:
                inputs = [req["input"] for req in curr_reqs]
                current_subctxs = [contexts[req["idx"]] for req in curr_reqs]
                
                # NOTE: the batch call should handle errors internally
                outputs = tool_obj.batch_call(inputs, current_subctxs=current_subctxs)

                for req, output in zip(curr_reqs, outputs):
                    output["input"] = output["input"].dict()  # convert to dict for serialization
                    # batch returns "success", "input", "tool_response", "error"
                    results[req["idx"]] = {
                        **output,
                        "raw_request": req["raw_request"],
                        "tool": tool,
                    }
        # --- end applying tool in groups ---

        tool_responses = []
        for result, meta in zip(results, metas):
            meta["tool_calls"].append(result)
            resp = result["tool_response"]
            tool_responses.append("" if resp is None else resp)

        logger.info(f"Tool application took {time.time() - t0} seconds")
        # --- end tool application ---

        return tool_responses

    def _responses_and_chats_to_training_examples(
        self,
        samples: list[ClientSample],
        convos: list[list[dict]],
        metas: list[dict],
        contexts: list[str] | None,
    ) -> list[TrainingExample]:
        examples = []
        for sample, chat, meta, context in zip(
            samples,
            convos,
            metas,
            contexts,
            strict=True,
        ):
            if sample.top_logprobs is None:
                continue
            
            examples.append(
                TrainingExample(
                    messages=[TrainingExample.Message(**message) for message in chat],
                    top_logprob_ids=sample.top_logprobs.top_ids,
                    top_logprob_logprobs=sample.top_logprobs.top_logprobs.astype(
                        np.float32
                    ),  # We can convert to float32 to save space in the file
                    token_ids=sample.top_logprobs.token_ids,
                    num_output_tokens=sample.num_output_tokens,
                    type="todo",
                    metadata=meta,
                    system_prompt=context,
                )
            )
        return examples


# --- begin chat helper functions ---


def system(content: str) -> dict:
    return dict(role="system", content=content)


def user(content: str, cot: bool = False) -> dict:
    if cot:
        instruction = random.choice(COT_INSTRUCTIONS)
        content = f"{content}\n\n{instruction}"
    return dict(role="user", content=content)


def assistant(content: str) -> dict:
    return dict(role="assistant", content=content)


def flip_roles(convo: list[dict]) -> list[dict]:
    def flip_role(role: str) -> str:
        if role == "user":
            return "assistant"
        elif role == "assistant":
            return "user"
        return role

    return [dict(role=flip_role(d["role"]), content=d["content"]) for d in convo]


# --- end chat helper functions ---


# --- begin generators for seed prompt slices  ---


def structuring_seed_prompt(**kwargs):
    DATA_FORMATS = [
        "JSON",
        "YAML",
        "TOML",
        "INI",
        "XML",
        "plain text",
    ]

    data_format = random.choice(DATA_FORMATS)

    EXAMPLES = [
        (
            "Can you structure the information in {{subsection}} of {{document}} related to {{something specific}} "
            f"in the following format: {data_format}? "
            "Be sure to include precise information like any dates, times, names, and numerical values.'"
        ),
        (
            "Can you structure the information in {{subsection}} of {{document}} "
            f"in the following format: {data_format}? "
            "Be sure to include precise information like any dates, times, names, and numerical values.'"
        ),
    ]

    example = random.choice(EXAMPLES)

    return (
        f"Please generate a single chat message instructing an LLM to structure the information in {data_format}. "
        "Output only the chat message itself and absolutely nothing else. "
        "Make sure it is clear what section and document you are asking about. "
        f"The message can follow the following template, filling in details from the corpus: \n\n'{example}'"
    )


def summarization_seed_prompt(**kwargs):
    prompts = [
        (
            "Please generate a single chat message instructing an LLM to summarize part of the corpus. "
            "Make sure the instruction is very explicit about the section of the corpus that you want to summarize. "
            "Include details (ids, names, titles, dates, etc.) that make it clear what you are asking about. "
        ),
        (
            "Please generate a single chat message instructing an LLM to summarize a section. "
            "Make sure the instruction is explicit about the section that should be summarized and the document it is from."
        ),
    ]
    prompt = random.choice(prompts)
    return prompt


def question_seed_prompt(**kwargs):
    prompts = [
        (
            "Generate a question for an LLM that will test its knowledge of the information in the corpus above. "
            "In your question be sure to include details (ids, names, titles, dates, etc.) that make it clear what you are asking about. "
            "Output only a single question. Do NOT include any other text or explanation other than the question."
        ),
        (
            "Generate a message for an LLM that will test its knowledge of the information in the corpus above."
            "Be sure to include details (ids, names, titles, dates, etc.) in the question so that it can be answered without access to the corpus (i.e. closed-book setting). "
            "Output only a single question. Do NOT include any other text or explanation other than the question."
        ),
        (
            "You are helping to quiz a user about the information in the corpus. "
            "Please generate a question about the subsection of the corpus above. "
            "Be sure to include details (ids, names, titles, dates, etc.) in the question to make it clear what you are asking about. "
            "Answer only with the question, do not include any other text."
        ),
    ]
    prompt = random.choice(prompts)
    return prompt


def use_case_seed_prompt(**kwargs):
    prompt = (
        "You are working to train a language model on the information in the following corpus. "
        "Your primary goal is to think about practical, real-world tasks or applications that someone could achieve using the knowledge contained within this corpus. "
        "Consider how a user might want to apply this information, not just recall it. "
        "After considering potential use cases, your task will be to generate a sample question that reflects one of these downstream applications. "
        "This question/instruction/task should be something a user, who has access to this corpus, might ask when trying to accomplish their specific goal. "
        "Output only a single question. Do NOT include any other text or explanation other than the question."
    )
    return prompt


def creative_seed_prompt(**kwargs):
    prompt = [
        (
            "You are having a creative conversation inspired by the information in the corpus. "
            "Please generate a question for your conversation partner to start off the discussion. "
            "Answer only with the question, do not include any other text."
        ),
    ]
    return random.choice(prompt)

def generic_seed_prompt(**kwargs):
    return (
        f"Please generate a single chat message to begin a conversation about the information in the corpus. Ask a question about the corpus or make a request."
    )


SLICE_TYPES = Literal[
    "structuring", "summarization", "aggregation", "question", "use_case", "creative", 'mtob_ke', 'mtob_ek', 'generic'
]
SEED_PROMPT_REGISTRY: dict[SLICE_TYPES, Callable] = {
    "structuring": structuring_seed_prompt,
    "summarization": summarization_seed_prompt,
    "question": question_seed_prompt,
    "use_case": use_case_seed_prompt,
    "creative": creative_seed_prompt,
    "generic": generic_seed_prompt,
}
# --- end generators for seed prompt slices  ---


# --- begin definition of prompt sampler  ---

class SlicePromptSampler(PromptSampler):

    class Config(PromptSampler.Config):
        slices: List[SLICE_TYPES]
        max_tokens_initial_context: int = 4096
        
        include_outline: bool = True
        leaves_only: bool = False

    def __init__(
        self,
        config: Config,
        context: StructuredContext,
        tokenizer: AutoTokenizer,
    ):
        self.config = config
        self.context = context
        self.tokenizer = tokenizer

        self.ctxs = list_nested_contexts(self.context, leaves_only=self.config.leaves_only)
        self.outline = get_outline(self.context)

    def __call__(
        self,
        batch_idx: int,
        num_convos: int,
    ) -> List[tuple[str, str]]:
        seed_prompts = [
            SEED_PROMPT_REGISTRY[random.choice(self.config.slices)]()
            for _ in range(num_convos)
        ]
        return self._sample_initial_subcontext(), seed_prompts

    def _sample_initial_subcontext(self) -> str:
        # TODO: This can be improved
        path, ctx = random.choice(self.ctxs)
        tokens = self.tokenizer.encode(ctx.text)
        if len(tokens) > self.config.max_tokens_initial_context:
            start_idx = random.randint(
                0, len(tokens) - self.config.max_tokens_initial_context
            )
            end_idx = start_idx + self.config.max_tokens_initial_context
            text = self.tokenizer.decode(tokens[start_idx:end_idx])
        else:
            text = ctx.text
        if self.config.include_outline:
            return dedent(
                f"""\
                --- begin outline ---
                {self.outline}
                --- end outline ---

                Below is a subsection from the corpus (located at `{path}`).

                {text}
            """
            )
        else:
            return  f"Below is a subsection from the corpus (located at `{path}`). {text}"
            



class SlicePromptSamplerWithChunks(PromptSampler):

    class Config(PromptSampler.Config):
        slices: List[SLICE_TYPES]
        # min and max chunk size in tokens
        min_chunk_size: int = 512
        max_chunk_size: int = 2048

        desc: str = ""

    def __init__(
        self,
        config: Config,
        context: StructuredContext,
        tokenizer: AutoTokenizer,
        **kwargs,
    ):
        self.config = config
        self.context = context
        self.tokenizer = tokenizer

        self.tokens = self.tokenizer.encode(self.context.text)

    def __call__(self, batch_idx: int, num_convos: int,) -> List[tuple[str, str]]:
        seed_prompts = [
            SEED_PROMPT_REGISTRY[random.choice(self.config.slices)]()
            for _ in range(num_convos)
        ]
        return self._sample_initial_subcontext(), seed_prompts

    def _sample_initial_subcontext(self) -> str:
        chunk_size = random.randint(self.config.min_chunk_size, self.config.max_chunk_size)
        chunk_start = random.randint(0, len(self.tokens) - chunk_size)
        chunk_end = chunk_start + chunk_size
        chunk = self.tokenizer.decode(self.tokens[chunk_start:chunk_end])

        if self.config.desc:
            chunk = f"{self.config.desc}\n\n{chunk}"
        
        return chunk
    

CONTEXTUALIZED_CHUNK_TEMPLATE = """\
<chunk>
{summary}
It is located at the following path in the corpus: `{path}`

<text>
{text}
</text>
</chunk>\
"""
    
class SlicePromptSamplerWithContextualizedChunks(SlicePromptSamplerWithChunks):

    class Config(PromptSampler.Config):
        slices: List[SLICE_TYPES]
        # min and max chunk size in tokens
        min_chunk_size: int = 512
        max_chunk_size: int = 2048

        single_leaf: bool = False
        

        num_summaries: int = 1
        summary_temperature: float = 0.6

        force_cache: bool = False
        
    def __init__(
        self,
        config: Config,
        context: StructuredContext,
        tokenizer: AutoTokenizer,
        client: Client,
    ):
        from cartridges.generate.tree_sampler import summarize_context_tree

        self.config = config
        self.context = context
        self.tokenizer = tokenizer

        self.leaves, self.weights, self.summaries = disk_cache(
            self._summarize_context_tree,
            cache_dir=os.path.join(os.environ["CARTRIDGES_OUTPUT_DIR"], "ctxual_summaries"),
            force=self.config.force_cache,
        )(
            context=context,
            tokenizer=tokenizer,
            client=client,
            max_tokens_per_section=self.config.max_chunk_size,
            num_summaries=self.config.num_summaries,
            summary_temperature=self.config.summary_temperature,
        )

        self.leaf_id_to_summaries = {
            id(leaf): [s[i] for s in self.summaries] 
            for i, leaf in enumerate(self.leaves)
        }

    @staticmethod
    def _summarize_context_tree(
        context: StructuredContext,
        tokenizer: AutoTokenizer,
        client: Client,
        max_tokens_per_section: int,
        num_summaries: int,
        summary_temperature: float,
    ):
        tree = structured_context_to_context_tree(
            context, 
            tokenizer, 
            max_tokens_per_section=max_tokens_per_section
        )
        
        def summarize_once():
            return summarize_context_tree(
                tokenizer=tokenizer,
                client=client,
                tree=tree,
                min_tokens_to_summarize=128,
                max_tokens_to_summarize=16384,
                max_tokens_in_summary=128,
                downward_pass=True,
                temperature=summary_temperature,
            )

        leaves: List[ContextTreeLeaf] = [
            leaf for leaf in tree.leaves()
        ]
        summaries: List[List[str]] = []
        weights: List[float] = [leaf.num_tokens for leaf in leaves]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(summarize_once) for _ in range(num_summaries)]

            for future in concurrent.futures.as_completed(futures):
                summ_tree: ContextTree = future.result()
                summaries.append([leaf.summary for leaf in summ_tree.leaves()])

        return leaves, weights, summaries
    
    def _sample_initial_subcontext(self) -> str:
        if self.config.single_leaf:
            return self._sample_initial_subcontext_single()
        else:
            return self._sample_initial_subcontext_multiple()
    
    def _sample_initial_subcontext_multiple(self) -> str:
        chunk_size = random.randint(self.config.min_chunk_size, self.config.max_chunk_size)

        seed: ContextTreeLeaf = random.choices(self.leaves)[0]
        ids: Set[int] = {id(seed)}

        context = [
            (seed.path(), seed.summary, seed.value)
        ]

        current_root: ContextTree = seed.parent_data.parent
        token_count = seed.num_tokens
        while token_count < chunk_size:
            for leaf in current_root.leaves():
                leaf: ContextTreeLeaf
                if id(leaf) not in ids:
                    if leaf.num_tokens + token_count <= chunk_size:
                        text = leaf.value 
                    else:
                        tokens = self.tokenizer.encode(leaf.value)[:chunk_size - token_count]
                        token_count += len(tokens)
                        text = self.tokenizer.decode(tokens)
                    summary = random.choice(self.leaf_id_to_summaries[id(leaf)])
                    context.append((leaf.path(), summary, text))
                    ids.add(id(leaf))
                    break
            else:
                if current_root.parent_data is None:
                    break
                current_root = current_root.parent_data.parent

        chunk_strs = []
        for path, summary, text in sorted(context, key=lambda x: x[0]):
            chunk_strs.append(
                CONTEXTUALIZED_CHUNK_TEMPLATE.format(
                    path=path,
                    summary=summary,
                    text=text,
                )
            )
        chunk_str = "\n-----------\n".join(chunk_strs)
        return chunk_str

    
    def _sample_initial_subcontext_single(self) -> str:
        leaf_idx = random.choices(range(len(self.leaves)), weights=self.weights, k=1)[0]
        summary = random.choice(self.summaries)[leaf_idx]
        leaf = self.leaves[leaf_idx]

        chunk_size = random.randint(self.config.min_chunk_size, self.config.max_chunk_size)
        if leaf.num_tokens > chunk_size:
            tokens = self.tokenizer.encode(leaf.value)
            chunk_start = random.randint(0, len(tokens) - chunk_size)
            chunk_end = chunk_start + chunk_size
            chunk = self.tokenizer.decode(tokens[chunk_start:chunk_end])
        else:
            chunk = leaf.value

        return CONTEXTUALIZED_CHUNK_TEMPLATE.format(
            summary=summary,
            path=leaf.path(),
            text=chunk,
        )

# The COT instructions below are randomly sampled from the following list.

COT_INSTRUCTIONS = [
    "If helpful, you can think before responding. Put your thinking between <thinking> and </thinking> tags. Then, provide your final response between <response> and </response> tags.",
    "Respond in the following format: <thinking>...</thinking> <response>...</response>",
    "Explain your reasoning before providing your final response.",
    "Explain your reasonining between <reasoning> and </reasoning> tags.",
    "Provide your final answer within <answer>...</answer> tags. Optionally, you can explain your reasoning between <reasoning> and </reasoning> tags.",
    "You may include your reasoning before answering. Use <reasoning>...</reasoning> to enclose your thoughts, and <final>...</final> for your answer.",
    "First, think through the problem and enclose your thoughts in <thought>...</thought>. Then, present your answer clearly in <output>...</output>.",
    "Use <step-by-step>...</step-by-step> for your intermediate reasoning, followed by <answer>...</answer> for the final result.",
    "Start with your analysis in <deliberation>...</deliberation>. Conclude with a clear answer in <response>...</response>.",
    "You may show your chain of thought in <chain>...</chain> and state your final decision in <decision>...</decision>.",
    "Wrap your reasoning process in <logic>...</logic>, and your ultimate conclusion in <conclusion>...</conclusion>.",
    "Think carefully before answering. Put your process in <process>...</process> and your solution in <solution>...</solution>.",
    "Please provide your thought process first, enclosed in <rationale>...</rationale>, and then give your final answer in <final_answer>...</final_answer>.",
    "Include your reasoning in <analysis>...</analysis> and your conclusion in <result>...</result>.",
    "Begin with <thinking_process>...</thinking_process> to show how you reasoned through the problem. Finish with <response>...</response>.",
    "Use <explanation>...</explanation> to walk through the logic. Then state your answer in <output>...</output>.",
    "Present your logical steps in <reasoning_chain>...</reasoning_chain> and conclude with <final_response>...</final_response>.",
    "Start with <evaluation>...</evaluation> to explain how you analyzed the question. Then, give your answer in <decision>...</decision>.",
    "Outline your reasoning in <deduction>...</deduction> and present the final answer in <resolution>...</resolution>.",
    "First explain in <justification>...</justification>, then give your definitive answer in <answer>...</answer>.",
    "Place your step-by-step logic in <path>...</path> and your outcome in <solution>...</solution>.",
    "Break down the problem in <walkthrough>...</walkthrough> before stating your answer in <conclusion>...</conclusion>.",
    "Reason through the problem in <examine>...</examine> and finalize with <respond>...</respond>.",
    "Give your thought process inside <trace>...</trace> and the answer inside <reply>...</reply>.",
    "Write your full reasoning under <work>...</work>, then state the answer clearly in <end>...</end>.",
    "Use <rundown>...</rundown> to explain your steps, and <final>...</final> to share your answer.",
    "First, walk through your reasoning process step by step. Then, clearly state your final answer.",
    "Begin by explaining how you approach the problem. Afterward, give your final response.",
    "Start with a detailed breakdown of your thought process. Conclude with a concise answer.",
    "Explain your logic as you work through the problem. When you're done, provide your conclusion.",
    "Think out loud as you reason through the question. End with a definitive answer.",
    "Work through the problem in detail, reasoning carefully. Then summarize your final decision.",
    "Describe each step you take to solve the problem. Finish by stating the final result.",
    "Provide a thorough explanation of how you arrive at your answer. Then state the answer clearly.",
    "Show your reasoning process from start to finish. Make sure to give your final answer at the end.",
    "Break the problem into logical steps and explain each one. Then give your final response.",
    "Write out your reasoning clearly and methodically. Conclude with your final conclusion.",
    "Reflect on the problem and describe your full reasoning. Then, say what your answer is.",
    "Think critically about the question and narrate your process. Then provide your final decision.",
    "Show your internal reasoning inside `{{Rationale}}...{{/Rationale}}`. Give the final conclusion inside `{{Conclusion}}...{{/Conclusion}}`.",
    "Detail your problem-solving steps between `[Steps]...[/Steps]`. Provide the answer between `[Result]...[/Result]`.",
    "Explain the derivation process using `Derivation: ... End Derivation`. State the final output using `Output: ... End Output`.",
    "Map out your thought path within `<Path>...</Path>`. State the final destination within `<Destination>...</Destination>`.",
    "Provide your analysis enclosed in `<<Analysis>>...<</Analysis>>`. Present the determined answer enclosed in `<<Answer>>...<</Answer>>`.",
    "Elaborate on your thinking process with `// Elaboration Start` and `// Elaboration End`. Provide the final response with `// Response Start` and `// Response End`.",
    "Use `(Thought Process: ... )` to show your thinking. Use `(Final Answer: ... )` for the result.",
    "Lay out the groundwork in `<Foundation>...</Foundation>`. Build the final answer in `<Structure>...</Structure>`.",
    "Document the logical flow in `{* Logic Flow *} ... {* /Logic Flow *}`. Deliver the outcome in `{* Outcome *} ... {* /Outcome *}`.",
    "Begin with your deliberation, marked by `Deliberation: ...`. Conclude with your final decision, marked by `Decision: ...`.",
    "Record your internal monologue in `<Monologue>...</Monologue>`. State the external response in `<Statement>...</Statement>`.",
    "Chart the sequence of reasoning within `[[Sequence]]...[[/Sequence]]`. Present the end point within `[[Endpoint]]...[[/Endpoint]]`.",
    "Dissect the problem in `<Dissection>...</Dissection>`. Synthesize the answer in `<Synthesis>...</Synthesis>`.",
    "Narrate your thought process using `Narrative: ...`. Provide the concluding answer using `Conclusion: ...`.",
    "Outline your strategy in `<Strategy>...</Strategy>`. Execute the final answer in `<Execution>...</Execution>`.",
    "Feel free to think it through out loud first, then just drop your answer at the end.",
    "Walk yourself through the problem—no rush. Once you're set, say what you'd go with.",
    "You can talk it out step by step. Just wrap up with whatever you think the answer is.",
    "Think it through however you like, and then let me know your final call.",
    "Start by working through the logic in your own way. When you're done, give the answer.",
    "Lay out your thinking as it comes to you. At the end, just say what you'd choose.",
    "Break it down how you want, no pressure. Then tell me what your final answer would be.",
    "Talk yourself through the reasoning part. When it feels right, give your answer.",
    "Explain it like you're figuring it out in real time, then land on your pick.",
    "Take a moment to think it through, and when you're ready, just say your answer.",
    "Work it out however makes sense to you. Then drop your answer when you're good.",
    "Go step by step, like you're thinking out loud. End with whatever answer you'd settle on.",
    "Walk through your reasoning step by step, and once it all clicks, share your answer.",
    "Take me through your thought process from start to finish, then tell me your conclusion.",
    "Think out loud as you piece it together. When you’re ready, just state your answer.",
    "Break down the problem in your own words, then wrap up with your choice.",
    "Talk through each part as you solve it, and then give your final answer.",
    "Map out your logic as you go, and once you’re confident, give your answer.",
    "Reason it out at your own pace, and when you’re set, let me know your answer.",
    "Feel free to puzzle it out stepwise. At the end, just say your final pick.",
    "Process the question however you like. Once you’ve worked it out, share your answer.",
    "Unpack the problem in detail, and when you’re ready, provide your answer.",
    "Work through the details in your own style, and then land on your answer.",
    "Think through the scenario, explaining as you go, and finish with your answer.",
    "Go through your reasoning openly, and then make your final call.",
    "Lay out your analysis clearly, and when you reach a conclusion, share it.",
    "Step through your logic in real time, and end with your decision.",
    "Explain your approach as you solve it, then just give your answer at the end.",
    "Go through the motions of solving it, and when you’re done, state your answer.",
    "Work the problem out in your own way, and when you’re ready, say your answer.",
    "Detail your thought process as it unfolds, then close with your answer.",
    "Take your time to reason through it, and when you’ve got it, give your answer.",
    "Unpack your reasoning in <logic>...</logic>. Conclude with your solution in <result>...</result>.",
    "Share your thought process inside [Reasoning]...[/Reasoning]. State your answer in [Answer]...[/Answer].",
    "Walk through your approach in <<<Thinking>>>...<<<End Thinking>>>. Provide your answer in <<<Answer>>>...<<<End Answer>>>.",
    "Place your analysis in <Breakdown>...</Breakdown>. Offer the answer in <Solution>...</Solution>.",
    "Use [[Rationale]]...[[/Rationale]] for your reasoning. Use [[Conclusion]]...[[/Conclusion]] for your answer.",
    "Think through the problem in # Reasoning: ... # End Reasoning. Give your answer in # Answer: ... # End Answer.",
    "Lay out your process in <Process>...</Process>. Deliver your conclusion in <Conclusion>...</Conclusion>.",
    "Step through your logic in {Reasoning: ... }. Present your answer in {Answer: ... }.",
    "Present your thought process in --- Reasoning --- ... --- End Reasoning ---. State your answer in --- Answer --- ... --- End Answer ---.",
    "Outline your thinking in (Reasoning Start)...(Reasoning End). Wrap up with (Answer Start)...(Answer End).",
    "Describe your approach using <Approach>...</Approach>. Summarize your answer using <Summary>...</Summary>.",
    "Go through your logic in [Logic Path]...[/Logic Path]. Finalize with [Final Response]...[/Final Response].",
    "Map out your steps in <<Steps>>...<</Steps>>. Place your answer in <<Result>>...<</Result>>.",
    "Narrate your reasoning in <Explanation>...</Explanation>. State your answer in <Reply>...</Reply>.",
    "Explain your process inside {Process: ...}. Conclude with {Result: ...}.",
    "Break down your logic in [Analysis]...[/Analysis]. Conclude with your answer in [Conclusion]...[/Conclusion].",
    "Think out your process in <Thoughts>...</Thoughts>. Share your answer in <Answer>...</Answer>.",
    "Detail your reasoning in [[Analysis]]...[[/Analysis]]. Give the answer in [[Final]]...[[/Final]].",
    "Work through your thoughts in --- Process --- ... --- /Process ---. Finish with --- Solution --- ... --- /Solution ---.",
    "Talk through your logic in {Deliberation}...{/Deliberation}. Give your answer in {Decision}...{/Decision}.",
]
