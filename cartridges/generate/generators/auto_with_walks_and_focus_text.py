import abc
from inspect import getsource
import json
import re
import random
import time
from typing import List, Optional

import numpy as np
from pydrantic import ObjectConfig
from transformers import AutoTokenizer

from capsules.context import StructuredContext, list_nested_contexts
from capsules.generate.structs import TrainingExample
from capsules.clients.base import CapsulesConvoWithLogprobs, ClientConfig
from capsules.generate.generators.base import ContextConvoGenerator
from capsules.generate.outline import get_outline
from capsules.tools.base import Tool
from capsules.utils import get_logger
from capsules.generate.tree_sampler import (
    structured_context_to_context_tree,
    flood_fill_from_leafs,
    serialize_with_elide,
    ContextTreeLeaf,
)

logger = get_logger(__name__)


SYSTEM_PROMPT_TEMPLATE = """You are in a conversation about a corpus of information.
{prompt_specific_instructions}

The outline of the information is given below.
--- begin outline ---
{outline}
--- end outline ---
"""

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

SIMPLE_QA_SYSTEM = """You are working to train a language model on the information in the following corpus.

You will be given a corpus with some sections elided and your job is to generate a training question that is grounded in the provided content. 
After, we will train a langauge model to answer the question without access to the document (i.e. closed book).
The question should be challenging and can require the model to remember details.

Here is the corpus:
{corpus}
"""

MEMORIZATION_SYSTEM = """Here is a corpus of information that you should understand deeply.
{corpus}
"""

SIMPLE_QA_SEED = """Please generate a challenging question that focuses on the following excerpt from the corpus.

{focus}


The question should test one model knowledge of the information in the document when asked in a closed-book setting.
Output only a single question. Do NOT include any other text or explanation other than the question."""

MEMORIZATION_SEED = """Please generate a question that asks one to recall all the information in the following excerpt from the corpus.

{focus}

In your question, you should not give away the answer.
You should write your question in an unambigious manner that asks for the content verbatim (if it makes sense) and asks to connect it to the rest of the document.

The question should make sense in a closed-book setting (without the context of the corpus).
Please generation your question now. Output no other text or explanation other than the question.
"""



USAGE_SYSTEM = """I am going to provide you with a corpus of information.
Your primary goal is to think about practical, real-world tasks or applications that someone could achieve using the knowledge contained within this corpus. Consider how a user might want to apply this information, not just recall it.

After considering potential use cases, your task will be to generate a sample question that reflects one of these downstream applications. This question should be something a user, who has access to this corpus, might ask when trying to accomplish their specific goal.

Here is the corpus:
{corpus}
"""

USAGE_SEED = """Please generate a single, realistic question based on the provided corpus context.

This question should represent a user trying to use the information in the corpus to achieve a specific practical goal or complete a task.

If it makes sense, please incorporate information from the following excerpt into your question:
--- begin focus excerpt ---
{focus}
--- end focus excerpt ---

Imagine someone trying to put the information *from this excerpt* into action to achieve their goal: what question would they ask?

Output only the question itself. Do not include any preamble, explanation, or description of the use case. Just the question.
"""




# SEED_PROMPT = """You are helping to quiz a user about the information in the corpus.

# Please generate a question about the corpus above.

# I am going to give you a section of the corpus to focus on. The xml-style tags are simply to help you understand the structure of the text.
# Please focus your your question on the following section:
# {focus_text}

# Make sure your answer is standalone and makes sense in a closed book setting (without the context of the corpus).
# Do not use phrases like "as mentioned above" or "as stated in the text".

# Ask your question in plain text English.

# Answer only with the question, do not include any other text.
# """


class PromptSampler(abc.ABC):

    class Config(ObjectConfig):
        _pass_as_config = True

        x: int = 0

    @abc.abstractmethod
    def __call__(self, num_prompts: int) -> List[str]:
        raise NotImplementedError()


class AutoConvoGenerator(ContextConvoGenerator):

    class Config(ContextConvoGenerator.Config):
        client: ClientConfig
        tools: List[Tool.Config]

        max_rounds: int = 1

        initial_system_prompt_template: str = SYSTEM_PROMPT_TEMPLATE

        seed_prompts: list[tuple[str, str]] = [
            # (SIMPLE_QA_SYSTEM, SIMPLE_QA_SEED),
            # (MEMORIZATION_SYSTEM, MEMORIZATION_SEED),
            (USAGE_SYSTEM, USAGE_SEED),
        ]
        tool_prompt_template_a: str = TOOL_PROMPT_TEMPLATE
        temperature_a: float = 0.6
        max_completion_tokens_a: int = 512

        tool_prompt_template_b: str = TOOL_PROMPT_TEMPLATE
        temperature_b: float = 0.0
        max_completion_tokens_b: int = 1024

        tokenizer: str

        num_top_logprobs: int = 20

        max_tokens_per_page: int = 384
        num_nodes_in_content: int = 350  # TODO

        num_focus_leaves_per_context: int = 8
        sibling_bias: int = 3
        seed: int = 58

    def __init__(self, config: Config, context: StructuredContext):
        random.seed(config.seed)

        self.config = config
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

        self.tree = structured_context_to_context_tree(
            self.context, self.tokenizer, self.config.max_tokens_per_page
        )

        if isinstance(self.config.seed_prompts, str):
            self.config.seed_prompts = [self.config.seed_prompts]


    def _sample_initial_subcontext_and_focus_leaves(
        self,
    ) -> tuple[str, list[ContextTreeLeaf]]:
        all_leaves = self.tree.leaves()

        leaves = random.choices(
            all_leaves,
            k=self.config.num_focus_leaves_per_context,
            weights=[leaf.num_tokens for leaf in all_leaves],
        )

        nodes = flood_fill_from_leafs(
            leaves,
            num_nodes_to_include=self.config.num_nodes_in_content,
            sibling_bias=self.config.sibling_bias,
        )

        context_str = serialize_with_elide(
            self.tree,
            nodes,
        )

        return context_str, leaves

    def sample_convos(
        self, batch_idx: int, num_convos: int, total_batches: int
    ) -> list[TrainingExample]:

        # (2) Get initial subsection
        context, focus_leaves = self._sample_initial_subcontext_and_focus_leaves()

        assert not isinstance(self.config.seed_prompts, PromptSampler)

        assert num_convos >= self.config.num_focus_leaves_per_context
        seed_prompt_templates = random.choices(self.config.seed_prompts, k=num_convos)

        contexts = [
            system_template.format(corpus=context)
            for (system_template, _) in seed_prompt_templates
        ]

        seed_prompts = [
            seed_prompt.format(
                focus=f"Path: {focus_leaves[index % len(focus_leaves)].path()}, content:\n{focus_leaves[index % len(focus_leaves)].value}"
            )
            for index, (_, seed_prompt) in enumerate(seed_prompt_templates)
        ]

        convos: List[List[dict]] = [[] for _ in range(num_convos)]

        metas: List[dict] = [{"tool_calls": []} for _ in range(num_convos)]
        for round_idx in range(self.config.max_rounds):

            # (3.1) bot_a requests new content to be added to the context
            # --- begin bot A tool usage ---
            t0 = time.time()
            # tool_resps: List[str] = self._get_content_via_tool(
            #     prompt_template=self.config.tool_prompt_template_a,
            #     convos=[
            #         [system(ctx), user(seed), *flip_roles(convo)]
            #         for ctx, seed, convo in zip(contexts, seed_prompts, convos)
            #     ],
            #     metas=metas,
            # )
            # contexts = [ctx + resp for ctx, resp in zip(contexts, tool_resps)]
            logger.info(
                f"Round {round_idx}: Bot A tool usage took {time.time() - t0} seconds"
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
                convo + [user(resp.assistant_text)]
                for convo, resp in zip(convos, resps)
            ]
            logger.info(
                f"Round {round_idx}: Bot A response generation took {time.time() - t0} seconds"
            )
            # --- end bot A response generation ---

            # (3.3) bot_b requests new content to be added to the context
            # --- begin bot B tool usage ---
            t0 = time.time()
            # tool_resps: List[str] = self._get_content_via_tool(
            #     prompt_template=self.config.tool_prompt_template_b,
            #     convos=[[system(ctx), *convo] for ctx, convo in zip(contexts, convos)],
            #     metas=metas,
            # )
            # contexts = [ctx + resp for ctx, resp in zip(contexts, tool_resps)]
            logger.info(
                f"Round {round_idx}: Bot B tool usage took {time.time() - t0} seconds"
            )
            # --- end bot B tool usage ---

            # (3.4) bot_b generates a response
            # --- begin bot B response generation ---
            # breakpoint()
            resps = self.client.chat_with_logprobs(
                # [[system(ctx), *convo] for ctx, convo in zip(contexts, convos)],
                [[system(context), *convo] for convo in convos],
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
        resps = [resp.assistant_text for resp in resps]
        # --- end tool selection ---

        # (3) Parse the tool responses and apply the tool. If it fails, just return empty string
        # --- begin tool application ---
        t0 = time.time()
        tool_responses = []
        for resp, meta in zip(resps, metas):
            try:
                json_str = re.search(r"\{.*\}", resp, re.DOTALL).group(0)
                spec = json.loads(json_str)
                tool = self.tools[spec["tool"]]
                tool_responses.append(tool(tool.ToolInput(**spec["kwargs"])))

                # We log metadata about the
                meta["tool_calls"].append(
                    {
                        "success": True,
                        "raw_request": resp,
                        "tool": spec["tool"],
                        "input": spec["kwargs"],
                        "tool_response": tool_responses[-1],
                        "error": None,
                    }
                )
            except Exception as e:
                # if it fails, just return nothing
                logger.error(f"Error parsing tool response: {e}")
                tool_responses.append("")
                meta["tool_calls"].append(
                    {
                        "success": False,
                        "raw_request": resp,
                        "error": str(e),
                        "tool": None,
                        "input": None,
                        "tool_response": None,
                    }
                )
        logger.info(f"Tool application took {time.time() - t0} seconds")
        # --- end tool application ---

        return tool_responses

    def _responses_and_chats_to_training_examples(
        self,
        convos_with_logprobs: list[CapsulesConvoWithLogprobs],
        convos: list[list[dict]],
        metas: list[dict],
    ) -> list[TrainingExample]:
        examples = []
        for convo_with_logprobs, chat, meta in zip(
            convos_with_logprobs,
            convos,
            metas,
            strict=True,
        ):
            # (1) Strip the system prompt from the returned token_ids
            header_locations = np.where(convo_with_logprobs.token_ids == 128006)[
                0
            ].tolist()
            assert len(header_locations) == len(chat) + 1
            assert header_locations[0] == 1
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
                )
            )
        return examples


# --- begin chat helper functions ---


def system(content: str) -> dict:
    return dict(role="system", content=content)


def user(content: str) -> dict:
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


# MAX_ROUNDS = 3
# SEED_PROMPT = "Please ask a question about the passage of text."
# TOOL_PROMPT = "Please use one of the tools below to pull in relevant context."

# def generate_convo(context: List[str]):
#     convo = []
#     context = random.choice(context)
#     for _ in range(MAX_ROUNDS):
#         tool_resp = llm.chat(context + SEED_PROMPT + convo + TOOL_PROMPT)
#         context += use_tool(tool_resp)

#         response = llm.chat(context + SEED_PROMPT + convo)
#         convo += [user(response)]

#         tool_choice = llm.chat(context + convo + TOOL_PROMPT)
#         context += use_tool(tool_choice)

#         response = llm.chat(context + convo)
#         convo += [assistant(response)]
# for ctx, convo in zip(contexts, convos):
#     print(ctx)
#     for message in convo:
#         print(f"{message['role']}: {message['content']}")
#         print("\n")
#     print("-"*100)


# use cases prompt
USER_META_USES_PROMPT = """I am going to give you a corpus of information about {corpus_description}.
Your task is to identify a list of reasons why someone would want to use this corpus. 

<summary>
{corpus_summary}
</summary>

Here is a random section of the corpus:
<random-section>
{initial_section}
</random-section>

List out some of the use cases you forsee for this corpus.
"""
