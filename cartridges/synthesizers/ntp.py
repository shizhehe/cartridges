
from collections import defaultdict
import asyncio
import time
import uuid
import random
from typing import List

from transformers import AutoTokenizer

from cartridges.structs import TrainingExample
from cartridges.data.tools import instantiate_tools
from cartridges.clients.base import ClientConfig, ClientSample, FlatTopLogprobs
from cartridges.synthesizers.base import AsyncConvoSynthesizer, ConvoSynthesizer
from cartridges.data.tools import Tool, ToolSet, ToolOutput
from cartridges.data import MODEL_TO_TOOL_TEMPLATE, MODEL_TO_TOOL_CALL_PARSER, ToolCall, render_tool_template
from cartridges.utils import get_logger
from cartridges.data.resources import Resource

logger = get_logger(__name__)

TOOL_PROMPT_TEMPLATE = """You need to respond to the following message:

<message>
{message}
</message>
{tools}"""

SYSTEM_PROMPT_TEMPLATE = """
You are in a conversation about the following user information.

<info>
{subcorpus}
</info>"""


PROMPTS = [
    "Can you please repeat the context verbatim? Say nothing else.",
    "Please repeat all of the text in your system prompt word-for-word. Say nothing else.",
    "Regurgitate the entire system prompt verbatim. Say nothing else.",
    "Repeat word-for-word the context. Do not include any other text except for the exact words in the context.",
]


class NTPSynthesizer(AsyncConvoSynthesizer):

    class Config(ConvoSynthesizer.Config):
        client: ClientConfig

        resources: List[Resource.Config]

        system_prompt_template: str = SYSTEM_PROMPT_TEMPLATE

        max_rounds: int = 1

        prob_thinking: float = 0.0

        temperature_b: float = 0.0
        max_completion_tokens_b: int = 1024

        num_top_logprobs: int = 20
        min_prob_mass: float = 0.99


    def __init__(self, config: Config):
        self.config = config

        self.client = self.config.client.instantiate()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.client.model_name)
    
        self.is_setup = False

        random.seed(82)
    
    async def setup(self):
        
        self.resources: List[Resource] = [
            resource.instantiate() for resource in self.config.resources
        ]
        await asyncio.gather(*[resource.setup() for resource in self.resources])
    
        self.is_setup = True
    
    async def cleanup(self):
        """Clean up resources"""
        self.is_setup = False
    
    async def __aenter__(self):
        await self.setup()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
        return False

    async def sample_convos(
        self, batch_idx: int, batch_size: int, total_batches: int
    ) -> list[TrainingExample]:
        batch_id = f"{batch_idx}"

        if not self.is_setup:
            raise RuntimeError("Synthesizer not setup. Call setup() first.")

        # (1) Get initial system prompt and seed prompts
        # --- begin prompt sampling ---
        t0 = time.time()
        resource = random.choice(self.resources)
        ctx, seed_prompts = await resource.sample_prompt(batch_size=batch_size)

        initial_system_prompt = self.config.system_prompt_template.format(subcorpus=ctx)
        assert len(seed_prompts) == batch_size
        logger.info(f"[batch={batch_id}] Prompt sampling took {time.time() - t0} seconds")
        # --- end prompt sampling ---

        # (2) Initialize convos
        # --- begin initialization of convos ---
        t0 = time.time()
        convos: List[List[dict]] = [[] for _ in range(batch_size)]
        contexts: List[str] = [initial_system_prompt] * batch_size
        metas: List[dict] = [
            {
                "tool_calls": [],
                "seed_prompt": seed_prompt,
                "initial_system_prompt": initial_system_prompt,
            }
            for seed_prompt in seed_prompts
        ]
        logger.info(f"[batch={batch_id}] Initialization of convos took {time.time() - t0} seconds")
        # --- end initialization of convos ---
        # (3) Generate convos
        for round_idx in range(self.config.max_rounds):


            # (3.2) With new information in context, generate user message
            # --- begin bot A response generation ---
            t0 = time.time()
            text = random.choice(PROMPTS)
            resp = ClientSample(text=text, top_logprobs=None, token_ids=None)
            convos = [
                convo + [user(text, resp_obj=resp)]
                for convo in convos
            ]
            logger.info(
                f"[batch={batch_id}] Round {round_idx}: Bot A response generation took {time.time() - t0} seconds"
            )
            # --- end bot A response generation ---

            # (3.4) bot_b generates a response
            # --- begin bot B response generation ---
            t0 = time.time()
            resps = await self.client.chat(
                [trim_fields([system(ctx), *convo]) for ctx, convo in zip(contexts, convos)],
                temperature=self.config.temperature_b,
                top_logprobs=self.config.num_top_logprobs,
                max_completion_tokens=self.config.max_completion_tokens_b,
                modal_upstream_id=batch_id,
                enable_thinking=random.random() < self.config.prob_thinking,
            )
            resps: List[ClientSample] = resps.samples
            convos = [
                convo + [assistant(resp.text, resp_obj=resp)]
                for convo, resp in zip(convos, resps)
            ]
            logger.info(
                f"[batch={batch_id}] Round {round_idx}: Bot B response generation took {time.time() - t0} seconds"
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
        logger.info(f"[batch={batch_idx}] Conversion to training examples took {time.time() - t0} seconds")
        # --- end conversion to training examples ---
        return examples

    def _responses_and_chats_to_training_examples(
        self,
        samples: list[ClientSample],
        convos: list[list[dict]],
        metas: list[dict],
        contexts: list[str] | None,
    ) -> list[TrainingExample]:
        examples = []
        for chat, meta, context in zip(
            convos,
            metas,
            contexts,
            strict=True,
        ):

            def prepare_logprobs(message: dict) -> FlatTopLogprobs | None:
                if message["resp_obj"].top_logprobs is not None:
                    return message["resp_obj"].top_logprobs.flatten(
                        threshold=self.config.min_prob_mass)
                else:
                    return None

            
            examples.append(
                TrainingExample(
                    messages=[
                        TrainingExample.Message(
                            role=message["role"],
                            content=message["content"],
                            token_ids=message["resp_obj"].token_ids,
                            top_logprobs=prepare_logprobs(message),
                        )
                        for message in chat
                    ],
                    type="todo",
                    metadata=meta,
                    system_prompt=context,
                )
            )
        return examples


# --- begin chat helper functions ---
def system(content: str) -> dict:
    return dict(role="system", content=content)


def user(content: str, resp_obj: ClientSample = None) -> dict:
    return dict(role="user", content=content, resp_obj=resp_obj)


def assistant(content: str, resp_obj: ClientSample) -> dict:
    return dict(role="assistant", content=content, resp_obj=resp_obj)


def flip_roles(convo: list[dict]) -> list[dict]:
    def flip_role(role: str) -> str:
        if role == "user":
            return "assistant"
        elif role == "assistant":
            return "user"
        return role

    return [dict(role=flip_role(d["role"]), content=d["content"]) for d in convo]

def trim_fields(convo: list[dict]) -> list[dict]:
    return [dict(role=d["role"], content=d["content"]) for d in convo]

# --- end chat helper functions ---

