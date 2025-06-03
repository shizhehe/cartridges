import abc
from collections import defaultdict
from inspect import getsource
from itertools import tee
import json
from platform import release
import re
import random
from textwrap import dedent
import time
from typing import Callable, List, Literal, Optional, Tuple

import numpy as np
from pydrantic import ObjectConfig
from transformers import AutoTokenizer

from capsules.context import StructuredContext, list_nested_contexts
from capsules.generate.structs import TrainingExample
from capsules.clients.base import CapsulesConvoWithLogprobs, ClientConfig
from capsules.generate.generators.base import ContextConvoGenerator
from capsules.generate.outline import get_outline
from capsules.generate.tree_sampler import (
    ContextTreeLeaf,
    flood_fill_from_leafs_tokens,
    serialize_with_elide,
    structured_context_to_context_tree,
    summarize_context_tree,
)
from capsules.tools.base import Tool
from capsules.utils import get_logger

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

SYSTEM_PROMPT_TEMPLATE_WITH_DEP = """You are in a conversation about the corpus of information below. Note that some of the corpus has been omitted to fit within the context window.
When you talk about the passage, do not start your responses with "According to the passage", "In the passage", or any other similar phrase. Just start with the content of your response.

<corpus>
{subcorpus}
</corpus>

Remember to indicate which conversation you're talking about in your response, if possible. 
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


class AutoConvoGenerator(ContextConvoGenerator):

    class Config(ContextConvoGenerator.Config):
        client: ClientConfig

        tools: List[Tool.Config]
        use_tools_a: bool = False
        use_tools_b: bool = False

        # `prompt_sampler` is responsible for sampling the initial system prompt and the seed prompts
        # We combined them together since some strategies might involve dependencies
        # between the two
        prompt_sampler: PromptSampler.Config
        system_prompt_template: str = SYSTEM_PROMPT_TEMPLATE
        system_prompt_template_doc: str = SYSTEM_PROMPT_TEMPLATE_WITH_DEP

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
        )
        random.seed(82)

    def sample_convos(
        self, batch_idx: int, num_convos: int, total_batches: int
    ) -> list[TrainingExample]:
        # (1) Get initial system prompt and seed prompts
        subcorpus, seed_prompts = self.prompt_sampler(batch_idx, num_convos)
                
        if hasattr(self.prompt_sampler.config, "document_name") and self.prompt_sampler.config.document_name:
            initial_system_prompt = self.config.system_prompt_template_doc.format(
                subcorpus=subcorpus, 
                # document_name = self.prompt_sampler.config.document_name,
            )
        else:
            initial_system_prompt = self.config.system_prompt_template.format(
                subcorpus=subcorpus,
            )
        assert len(seed_prompts) == num_convos

        # (2) Initialize convos
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
                )
                contexts = [ctx + resp for ctx, resp in zip(contexts, tool_resps)]
                logger.info(
                    f"Round {round_idx}: Bot A tool usage (select + apply) took {time.time() - t0} seconds"
                )
            # --- end bot A tool usage ---

            # (3.2) With new information in context, generate user message
            # --- begin bot A response generation ---
            t0 = time.time()
            resps = self.client.chat_with_logprobs(
                [
                    [system(ctx), user(seed), *flip_roles(convo)]
                    for ctx, seed, convo in zip(contexts, seed_prompts, convos)
                ],
                temperature=self.config.temperature_a,
                max_completion_tokens=self.config.max_completion_tokens_a,
            )

            convos = [
                convo
                + [
                    user(
                        f"{resp.assistant_text}", 
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
                )
                contexts = [ctx + resp for ctx, resp in zip(contexts, tool_resps)]
                logger.info(
                    f"Round {round_idx}: Bot B tool usage (select + apply) took {time.time() - t0} seconds"
                )
            # --- end bot B tool usage ---

            # (3.4) bot_b generates a response
            # --- begin bot B response generation ---
            resps = self.client.chat_with_logprobs(
                [[system(ctx), *convo] for ctx, convo in zip(contexts, convos)],
                temperature=self.config.temperature_b,
                top_logprobs=self.config.num_top_logprobs,
                max_completion_tokens=self.config.max_completion_tokens_b,
            )
            convos = [
                convo + [assistant(resp.assistant_text)]
                for convo, resp in zip(convos, resps)
            ]
            logger.info(
                f"Round {round_idx}: Bot B response generation took {time.time() - t0} seconds"
            )
            # --- end bot B response generation ---

        examples = self._responses_and_chats_to_training_examples(
            convos_with_logprobs=resps,
            convos=convos,
            metas=metas,
            contexts=contexts,
        )
        return examples

    def _get_content_via_tool(
        self,
        convos: list[list[dict]],
        prompt_template: str,
        metas: list[dict],
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
        resps = self.client.chat_with_logprobs(
            [
                # we funk with the last user message to add the tool prompt
                convo[:-1]
                + [user(prompt_template.format(tools=tool_str, message=convo[-1]))]
                for convo in convos
            ],
            temperature=self.config.temperature_a,
            max_completion_tokens=self.config.max_completion_tokens_a,
        )
        reqs = [resp.assistant_text for resp in resps]
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
                outputs = tool_obj.batch_call(inputs)

                for req, output in zip(curr_reqs, outputs):
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
        convos_with_logprobs: list[CapsulesConvoWithLogprobs],
        convos: list[list[dict]],
        metas: list[dict],
        contexts: list[str] | None,
    ) -> list[TrainingExample]:
        examples = []
        for convo_with_logprobs, chat, meta, context in zip(
            convos_with_logprobs,
            convos,
            metas,
            contexts,
            strict=True,
        ):
            # (1) Strip the system prompt from the returned token_ids
            header_locations = np.where(convo_with_logprobs.token_ids == 128006)[
                0
            ].tolist()
            try:
                assert len(header_locations) == len(chat) + 1
                assert header_locations[0] == 1
            except:
                return []

            prefix_end_idx = header_locations[1]
            token_ids = convo_with_logprobs.token_ids[prefix_end_idx:]

            assert len(token_ids) > convo_with_logprobs.num_output_tokens
            assert (
                convo_with_logprobs.top_logprob_logprobs.shape
                == convo_with_logprobs.top_logprob_ids.shape
            )
            assert (
                convo_with_logprobs.top_logprob_logprobs.shape[0] == len(token_ids) - 1
            ), "You probably need to pull down on tokasaurus or your first message is not a system message"

            # (2) Create the training example
            examples.append(
                TrainingExample(
                    messages=[TrainingExample.Message(**message) for message in chat],
                    top_logprob_ids=convo_with_logprobs.top_logprob_ids,
                    top_logprob_logprobs=convo_with_logprobs.top_logprob_logprobs.astype(
                        np.float32
                    ),  # We can convert to float32 to save space in the file
                    token_ids=token_ids,
                    num_output_tokens=convo_with_logprobs.num_output_tokens,
                    type="todo",
                    metadata=meta,
                    context=context,
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



def regurgitate_seed_prompt(**kwargs):
    prompt = (
        "Output the exact same text as the text in the corpus above, but put it in json tags.",
        "Output the exact same text as the text in the corpus above, but put it in xml tags.",
        "Our goal is to remember the information in the corpus above. Output it verbatim!", 
        "If there are multiple sections in the corpus, split them up and put them in separate json objects.",
        "If there are multiple sections in the corpus, split them up and put them in separate xml objects.",
        "Repeat the text verbatim in your answer",
    )
    prompt = random.choice(prompt)

    return (
        f"Please output a single chat message that states '{prompt}'. "
        "Output only the chat message itself and absolutely nothing else. "
    )


def creative_seed_prompt(**kwargs):
    prompt = [
        (
            "You are having a creative conversation inspired by the information in the corpus. "
            "Please generate a question for your conversation partner to start off the discussion. "
            "Answer only with the question, do not include any other text."
        ),
    ]
    return random.choice(prompt)


def kalamang_ke(**kwargs):
    prompt = [
        (
            "Please generate a question asking the user to translate a Kalamang sentence of your choosing to English. "
            "Answer only with the question, do not include any other text."
        )
    ]
    return prompt[0]
def kalamang_ek(**kwargs):
    prompt = [
        (
            "Please generate a question asking the user to translate a Kalamang sentence of your choosing to English. "
            "Answer only with the question, do not include any other text."
        )
    ]
    return prompt[0]


SLICE_TYPES = Literal[
    "structuring", "summarization", "aggregation", "question", "use_case", "creative", 'mtob_ke', 'mtob_ek', 
    "regurgitate"
]
SEED_PROMPT_REGISTRY: dict[SLICE_TYPES, Callable] = {
    "structuring": structuring_seed_prompt,
    "summarization": summarization_seed_prompt,
    "question": question_seed_prompt,
    "use_case": use_case_seed_prompt,
    "creative": creative_seed_prompt,

    "regurgitate": regurgitate_seed_prompt,
    
    
    "mtob_ke": kalamang_ke,
    "mtob_ek": kalamang_ek,
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
        document_name: str = ""

    def __init__(
        self,
        config: Config,
        context: StructuredContext,
        tokenizer: AutoTokenizer,
    ):
        self.config = config
        self.context = context
        self.tokenizer = tokenizer

        self.tokens = self.tokenizer.encode(self.context.text)

        # assert self.config.document_name != "", "document_name must be set in the config"

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
      


class SlicePromptSamplerOnSections(PromptSampler):

    class Config(PromptSampler.Config):
        slices: List[SLICE_TYPES]
        # min and max chunk size in tokens
        min_chunk_size: int = 1
        max_chunk_size: int = 3

        desc: str = ""
        document_name: str = ""

    def __init__(
        self,
        config: Config,
        context: StructuredContext,
        tokenizer: AutoTokenizer,
    ):
        self.config = config
        self.context = context
        self.tokenizer = tokenizer

        tokens_by_section = []
        for convo in self.context.convos:
            text = convo.text
            tokens = self.tokenizer.encode(text)
            tokens_by_section.append(tokens)

        self.tokens_by_section = tokens_by_section

        # assert self.config.document_name != "", "document_name must be set in the config"

    def __call__(self, batch_idx: int, num_convos: int,) -> List[tuple[str, str]]:
        seed_prompts = [
            SEED_PROMPT_REGISTRY[random.choice(self.config.slices)]()
            for _ in range(num_convos)
        ]
        return self._sample_initial_subcontext(), seed_prompts

    def _sample_initial_subcontext(self) -> str:        
        is_contiguous = random.choice([True, False])

        chunk = []
        if is_contiguous:
            # sample random sections as the chunk
            num_sections = random.randint(self.config.min_chunk_size, self.config.max_chunk_size)
            start_section = random.randint(0, len(self.tokens_by_section) - num_sections)
            end_section = start_section + num_sections
            tokens = []
            for i in range(start_section, end_section):
                chunk.append(self.tokenizer.decode(
                    self.tokens_by_section[i]
                    , skip_special_tokens=True
                ))
        else:
            # sample random tokens from random sections
            num_sections = random.randint(self.config.min_chunk_size, self.config.max_chunk_size)
            tokens = []
            for i in range(num_sections):
                section_idx = random.randint(0, len(self.tokens_by_section) - 1)
                chunk.append(self.tokenizer.decode(
                    self.tokens_by_section[section_idx], 
                    skip_special_tokens=True
                ))

        chunk = "\n\n".join(chunk)
        tokens = self.tokenizer.encode(chunk)
        chunk = self.tokenizer.decode(tokens)

        if self.config.desc:
            chunk = f"{self.config.desc}\n\n{chunk}"
        
        return chunk



class SlicePromptSamplerWithSummarization(PromptSampler):

    class Config(PromptSampler.Config):
        slices: List[SLICE_TYPES]

        max_tokens_in_context: Tuple[int, int] = (1024, 8192)

        # tree and sampling params        
        max_tokens_per_page: int = 4096
        num_focus_leaves_per_context: int = 1
        sibling_bias: int = 1
        

        # summary params
        min_tokens_to_summarize: int = 512
        max_tokens_to_summarize: int = 8192
        max_tokens_in_summary: int = 128


        client: ClientConfig

    def __init__(
        self,
        config: Config,
        context: StructuredContext,
        tokenizer: AutoTokenizer,
    ):
        self.config = config
        self.context = context
        self.tokenizer = tokenizer
        self.client = config.client.instantiate()

        tree = structured_context_to_context_tree(
            self.context, self.tokenizer, self.config.max_tokens_per_page
        )

        logger.info(f"Summarizing context tree with {len(tree.leaves())} leaves")
        self.tree = summarize_context_tree(
            tokenizer=tokenizer,
            client=self.client,
            tree=tree,
            min_tokens_to_summarize=self.config.min_tokens_to_summarize,
            max_tokens_to_summarize=self.config.max_tokens_to_summarize,
            max_tokens_in_summary=self.config.max_tokens_in_summary,
        )
        logger.info(f"Summarized context tree with {len(self.tree.leaves())} leaves")
        

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

    def _sample_initial_subcontext(
        self,
    ) -> tuple[str, list[ContextTreeLeaf]]:
        
        global_summary = self.tree.summary 
 
        while cur.parent_data is not None:
            cur = cur.parent_data.parent
            if not isinstance(cur, ContextTreePagedText):
                local_summary = cur.summary
                break

        out = """You are in a conversation about a large text. The following is a summary of the full text: "{summary}"

        The section you are currently discussing (path: {path}) is located within the section: "{local_summary}"

        Here is the text of the section you are currently discussing:
        <text>
        {text}
        </text>

        When you talk about the passage, do not start your responses with "According to the passage", "In the passage", or any other similar phrase. Just start with the content of your response.
        """.format(
            summary=global_summary,
            local_summary=local_summary,
            path=leaf.path(),
            text=leaf.value,
        )


        all_leaves = self.tree.leaves()

        leaves = random.choices(
            all_leaves,
            k=self.config.num_focus_leaves_per_context,
            weights=[leaf.num_tokens for leaf in all_leaves],
        )
        
        nodes = flood_fill_from_leafs_tokens(
            leaves,
            sibling_bias=self.config.sibling_bias,
            max_tokens=self.config.max_tokens_in_context,
        )

        context_str = serialize_with_elide(self.tree, nodes)

        return context_str




class TreeSlicePromptSampler(SlicePromptSampler):

    class Config(PromptSampler.Config):
        slices: List[SLICE_TYPES]

        max_tokens_per_page: int
        max_tokens_in_context: Tuple[int, int]
        num_focus_leaves_per_context: int
        sibling_bias: int

        desc: str = ""

    def __init__(
        self,
        config: Config,
        context: StructuredContext,
        tokenizer: AutoTokenizer,
    ):
        self.config = config
        self.context = context
        self.tokenizer = tokenizer

        self.tree = structured_context_to_context_tree(
            self.context, self.tokenizer, self.config.max_tokens_per_page
        )

    def _sample_initial_subcontext(
        self,
    ) -> tuple[str, list[ContextTreeLeaf]]:
        all_leaves = self.tree.leaves()

        leaves = random.choices(
            all_leaves,
            k=self.config.num_focus_leaves_per_context,
            weights=[leaf.num_tokens for leaf in all_leaves],
        )

        nodes = flood_fill_from_leafs_tokens(
            leaves,
            sibling_bias=self.config.sibling_bias,
            max_tokens=self.config.max_tokens_in_context,
        )

        context_str = serialize_with_elide(self.tree, nodes)

        if self.config.desc:
            context_str = f"{self.config.desc}\n\n{context_str}"

        return context_str


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
