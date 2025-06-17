from __future__ import annotations

from collections import defaultdict
import random
import time
from tqdm import tqdm
from typing import List, Literal
import uuid
import math
import concurrent.futures


from cartridges.clients.base import Client, ClientConfig
from cartridges.clients.tokasaurus_batch import CartridgesConvoWithLogprobs
from cartridges.generate.chunk import Chunker
from cartridges.generate.generators.base import (
    ContextConvoGenerator,
    get_subcontext,
    responses_and_chats_to_training_examples,
)
from cartridges.structs import (
    Context,
    Document,
    Section,
    TrainingExample,
)
from cartridges.generate.subcontext import SubcontextGenerator
from cartridges.utils import get_logger

logger = get_logger(__name__)


DATA_FORMATS = [
    "JSON",
    "JSONL", 
    "CSV",
    "YAML",
    "TOML",
    "INI",
    "XML",
    # "MessagePack",
    # "CSV",
]


SYSTEM_PROMPT_TEMPLATE = """You are a lawyer with expertise in the following state statutes.
{context}""" 



INTRA_STATE_INSTRUCTION_TEMPLATES = [
    "In Alabama, for what reasons can a landlord evict a tenant?",
    "Explain these California statutes like I am a 5 year old.",
    "List the types of retaliatory evictions that are prohibited by Alabama law.",
    "How many different types of evictions are there in Alaska?",
    "What are the tenant's rights under the Arizona rental statutes?",
    "Describe the eviction process in Texas according to state law.",
    "What legal protections do tenants have against eviction in New York?",
    "Summarize the key points of the Florida landlord-tenant statutes.",
    "What are the consequences for landlords violating tenant rights in Illinois?",
    "What are the legal requirements for a landlord to enter a rental property in Ohio?",
    "How does tenant screening work under Massachusetts law?",
    "What are the tenant's responsibilities under the Nevada rental statutes?",
    "Explain the process for disputing a security deposit deduction in Washington.",
    "What are the penalties for late rent payments in Michigan?",
    "What are the rights of tenants facing eviction in Pennsylvania?",
    "Describe the legal process for breaking a lease in Virginia.",
    "What are the landlord's obligations for property maintenance in Oregon?",
    "How do rent control laws affect tenants in New Jersey?",
] + [ 
    (
        f"Can you structure all of the information in the statutes in {data_format} format?"
    )
    for data_format in DATA_FORMATS
] + [
    (
        f"Write a {data_format} file that summarizes the information in the statutes?"
    )
    for data_format in DATA_FORMATS
] + [
    (
        f"Create a list in {data_format} format containing the information in the statutes."
    )
    for data_format in DATA_FORMATS
]

INTER_STATE_INSTRUCTION_TEMPLATES = [
    "How does the eviction notice period differ in Georgia compared to Alaska?",
    "How does the eviction process in Colorado compare to that in New Mexico?",

]

HARD_INSTRUCTION_TEMPLATES = [
    "In Florida, if a mobile home park owner wants to evict a mobile home owner under Chapter 723 F.S., what procedural differences exist compared to evicting a regular tenant under Chapter 83?",
    "Under California Civil Code § 1954, when can a landlord enter a tenant's dwelling without written notice, and how does that interact with the tenant’s right to quiet enjoyment under Civil Code § 1927? If a landlord repeatedly uses lawful reasons to enter but does so excessively, can the tenant bring a claim, and under what standard?",
    "Under the Texas Property Code, if a tenant abandons a rental unit without notice, under what circumstances can a landlord keep the security deposit, and when must they still provide an accounting? How does 'abandonment' differ from 'surrender'' under Texas law?",
]

ADDITIONAL_INSTRUCTIONS = [
    (
        "If helpful, you can think before responding. Put your thinking between <thinking> and </thinking> tags."
        "Then, provide your final response between <response> and </response> tags."
    ), 
    (
        "If helpful, you can think before responding. Put your thinking between <thinking> and </thinking> tags."
        "Then, provide your final response between <response> and </response> tags."
    ), 
    (
        "Respond in the following format: <thinking>...</thinking> <response>...</response>"
    ),
    (
        "Explain your reasoning before providing your final response."
    ),
    (
        "Explain your reasonining between <reasoning> and </reasoning> tags."
    ),
    (
        "Provide your final answer within <answer>...</answer> tags. Optionally, you can explain your reasoning between <reasoning> and </reasoning> tags."
    ),
    (
        ""
    )
]

QUESTION_PROMPT_TEMPLATE = """Please generate an instruction or question that is grounded in the information provided in the specific statutes you've been provided.

Here are some examples of instructions and questions:

--- begin instruction examples ---

{instruction_examples}

--- end instruction examples ---

Do not include any other text than the instruction. Do not simply copy the examples."""

HARD_Q_PROMPT_TEMPLATE = """Please generate a question that is grounded in the information provided in the specific statutes you've been provided.
The question should:
1. Require a specific output
2. Focus primarily on information from this specific excerpt
3. Test understanding of key concepts and details from this excerpt
3. Have multiple levels of reasoning, components, or variables
4. Directly involve the AI to demonstrate active problem-solving skills
5. Involve a level of creativity in approaching the problem?

Limit the length of your question to 3-4 sentences. Do not include any other text than the instruction. Do not simply copy the examples.
"""

HARD_Q_PROMPT_TEMPLATE_EXAMPLES = """Please generate a question that is grounded in the information provided in the specific statutes you've been provided.
The question should:
1. Require a specific output
2. Focus primarily on information from this specific excerpt
3. Test understanding of key concepts and details from this excerpt
3. Have multiple levels of reasoning, components, or variables
4. Directly involve the AI to demonstrate active problem-solving skills
5. Involve a level of creativity in approaching the problem?

Here are some examples of instructions and questions:

--- begin instruction examples ---

{instruction_examples}

--- end instruction examples ---

Do not include any other text than the instruction. Do not simply copy the examples."""

CREATIVE_Q_PROMPT_TEMPLATE = """Please generate a question that is grounded in the information provided in the specific statutes you've been provided.

The question should: 
1. Focus primarily on information from these specific legal statutes.
2. Test understanding of key details or concepts from these legal statutes.
3. Encourage a playful, creative response—such as explaining the concept through a short science fiction story, composing a limerick, or using an analogy from an unexpected domain (e.g., cooking, space travel, or wizardry).

Limit the length of your question to 3-4 sentences. Do not include any other text than the instruction. Do not simply copy the examples."""


ANSWER_PROMPT_TEMPLATE = """{instruction}{additional_instructions}"""


class LegalInstructionResponseGenerator(ContextConvoGenerator):
    class Config(ContextConvoGenerator.Config):

        question_client: ClientConfig
        question_temperature: float = 0.0
        question_max_completion_tokens: int = 512
        question_prompt_template: str = QUESTION_PROMPT_TEMPLATE
        instruction_templates: list[str] = None

        answer_client: ClientConfig
        answer_temperature: float = 0.0
        answer_max_completion_tokens: int = 512
        # The system prompt for the answer generator. It can contain
        # variables for `{instruction}` and `{additional_instructions}`.
        answer_prompt_template: str = ANSWER_PROMPT_TEMPLATE

        # The system prompt for the answer generator. It can contain
        # variables for `{context}`.
        system_prompt_template: str = SYSTEM_PROMPT_TEMPLATE

        max_statutes_per_subcontext: int = 4
        subcontext_type: Literal["inter_state", "intra_state", "both"] = "both"
        subcontext_frac_inter: float = 0.1 # Fraction of convos that use the note as the subcontext if subcontext_type is "both"

        num_top_logprobs: int = 20

        toc_num_samples: int = 16
        toc_max_num_batches_in_parallel: int = 1
        toc_batch_size: int = 16
        toc_max_completion_tokens: int = 256

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.config = config

        self.question_client = config.question_client.instantiate()
        self.answer_client = config.answer_client.instantiate()

        self.state_to_statutes = defaultdict(list)
        for section in self.context.sections:
            # the path format is f"reglab/housing_qa/{state}/{idx}", see context config
            state = section.path.split("/")[-2]
            self.state_to_statutes[state].append(section)

    def sample_convos(
        self,
        batch_idx: int,
        num_convos: int,
        total_batches: int,
    ) -> list[TrainingExample]:
        routing_tag = str(uuid.uuid4())
        
        # (1) sample a subcontext  provide to the question generators
        # --- begin get subcontext ---
        t0 = time.time()
        subcontext_type = self.config.subcontext_type
        if self.config.subcontext_type == "both":
            subcontext_type = "inter_state" if random.random() < self.config.subcontext_frac_inter else "intra_state"

        if subcontext_type == "intra_state":
            state = random.choice(list(self.state_to_statutes.keys()))
            all_statutes = self.state_to_statutes[state]
            selected_statutes = random.sample(all_statutes, min(len(all_statutes), self.config.max_statutes_per_subcontext))

            statutes_str = "\n\n".join([statute.to_string() for statute in selected_statutes])
            subcontext = f"Below are select statutes from {state}'s legal code related to housing law. \n {statutes_str}"
            instruction_templates = INTRA_STATE_INSTRUCTION_TEMPLATES
        else:
            # inter_state
            raise NotImplementedError("Inter-state subcontexts not implemented yet")
        
        if self.config.instruction_templates is not None:
            instruction_templates = self.config.instruction_templates

        system_prompt = self.config.system_prompt_template.format(context=subcontext)
        logger.info(f"Time taken to get subcontext: {time.time() - t0} seconds")
        # --- end get subcontext ---

        # (2) Create question chats
        # --- begin create question chats ---
        t0 = time.time()
        question_chats = []
        for _ in range(num_convos):
            num_examples = random.randint(1, 3)
            instruction_templates = random.choices(instruction_templates, k=num_examples)
            additional_instructions = random.choices(ADDITIONAL_INSTRUCTIONS, k=num_examples)

            instruction_examples = "Instruction Examples:\n\n"
            for instruction_template, additional_instruction in zip(instruction_templates, additional_instructions):
                example_instruction = instruction_template
                instruction_examples += f"\n{example_instruction}\n{additional_instruction}\n---\n"
            instruction_examples += "\n\n"

            question_chats.append(
                [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": self.config.question_prompt_template.format(
                            instruction_examples=instruction_examples,
                        )
                    }
                ]
            )
        logger.info(f"Time taken to create question chats: {time.time() - t0} seconds")
        # --- end create question chats ---

        # (3) Sample question responses
        # --- begin sample question responses ---
        t0 = time.time()
        question_responses = self.question_client.chat_with_logprobs(
            chats=question_chats,
            temperature=self.config.question_temperature,
            max_completion_tokens=self.config.question_max_completion_tokens,
            routing_tag=routing_tag,
        )  
        questions = [chat.assistant_text for chat in question_responses]
        logger.info(f"Time taken to sample question responses: {time.time() - t0} seconds")
        # --- end sample question responses ---

        # (4) Create answer chats
        # --- begin create answer chats ---
        t0 = time.time()
        answer_chats = [
            [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": question,
                },
            ]
            for question in questions
        ]

        answer_responses = self.answer_client.chat_with_logprobs(
            chats=answer_chats,
            temperature=self.config.answer_temperature,
            top_logprobs=self.config.num_top_logprobs,
            max_completion_tokens=self.config.answer_max_completion_tokens,
            routing_tag=routing_tag,
        )
        logger.info(f"Time taken to sample answers: {time.time() - t0} seconds")
        # --- end sample answers ---

        # (4) Construct convos
        # --- begin construct convos ---
        t0 = time.time()
        examples = responses_and_chats_to_training_examples(answer_responses, answer_chats)
        logger.info(
            f"Time taken to construct {len(answer_responses)} convos: {time.time() - t0} seconds"
        )
        # --- end construct convos ---
        return examples


class LegalInstructionResponseGeneratorNoExamples(ContextConvoGenerator):
    class Config(ContextConvoGenerator.Config):

        question_client: ClientConfig
        question_temperature: float = 0.0
        question_max_completion_tokens: int = 512
        question_prompt_template: str = CREATIVE_Q_PROMPT_TEMPLATE

        answer_client: ClientConfig
        answer_temperature: float = 0.0
        answer_max_completion_tokens: int = 512
        # The system prompt for the answer generator. It can contain
        # variables for `{instruction}` and `{additional_instructions}`.
        answer_prompt_template: str = ANSWER_PROMPT_TEMPLATE

        # The system prompt for the answer generator. It can contain
        # variables for `{context}`.
        system_prompt_template: str = SYSTEM_PROMPT_TEMPLATE

        max_statutes_per_subcontext: int = 4
        subcontext_type: Literal["inter_state", "intra_state", "both"] = "both"
        subcontext_frac_inter: float = 0.1 # Fraction of convos that use the note as the subcontext if subcontext_type is "both"

        num_top_logprobs: int = 20

        toc_num_samples: int = 16
        toc_max_num_batches_in_parallel: int = 1
        toc_batch_size: int = 16
        toc_max_completion_tokens: int = 256

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.config = config

        self.question_client = config.question_client.instantiate()
        self.answer_client = config.answer_client.instantiate()

        self.state_to_statutes = defaultdict(list)
        for section in self.context.sections:
            # the path format is f"reglab/housing_qa/{state}/{idx}", see context config
            state = section.path.split("/")[-2]
            self.state_to_statutes[state].append(section)

    def sample_convos(
        self,
        batch_idx: int,
        num_convos: int,
        total_batches: int,
    ) -> list[TrainingExample]:
        routing_tag = str(uuid.uuid4())
        
        # (1) sample a subcontext  provide to the question generators
        # --- begin get subcontext ---
        t0 = time.time()
        subcontext_type = self.config.subcontext_type
        if self.config.subcontext_type == "both":
            subcontext_type = "inter_state" if random.random() < self.config.subcontext_frac_inter else "intra_state"

        if subcontext_type == "intra_state":
            state = random.choice(list(self.state_to_statutes.keys()))
            all_statutes = self.state_to_statutes[state]
            selected_statutes = random.sample(all_statutes, min(len(all_statutes), self.config.max_statutes_per_subcontext))

            statutes_str = "\n\n".join([statute.to_string() for statute in selected_statutes])
            subcontext = f"Below are select statutes from {state}'s legal code related to housing law. \n {statutes_str}"
            instruction_templates = INTRA_STATE_INSTRUCTION_TEMPLATES
        else:
            # inter_state
            raise NotImplementedError("Inter-state subcontexts not implemented yet")

        system_prompt = self.config.system_prompt_template.format(context=subcontext)
        logger.info(f"Time taken to get subcontext: {time.time() - t0} seconds")
        # --- end get subcontext ---

        # (2) Create question chats
        # --- begin create question chats ---
        t0 = time.time()
        question_chats = []
        for _ in range(num_convos):
            # num_examples = random.randint(1, 3)
            # instruction_templates = random.choices(instruction_templates, k=num_examples)
            # additional_instructions = random.choices(ADDITIONAL_INSTRUCTIONS, k=num_examples)

            # instruction_examples = "Instruction Examples:\n\n"
            # for instruction_template, additional_instruction in zip(instruction_templates, additional_instructions):
            #     example_instruction = instruction_template
            #     instruction_examples += f"\n{example_instruction}\n{additional_instruction}\n---\n"
            # instruction_examples += "\n\n"

            question_chats.append(
                [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": self.config.question_prompt_template
                    }
                ]
            )
        logger.info(f"Time taken to create question chats: {time.time() - t0} seconds")
        # --- end create question chats ---

        # (3) Sample question responses
        # --- begin sample question responses ---
        t0 = time.time()
        question_responses = self.question_client.chat_with_logprobs(
            chats=question_chats,
            temperature=self.config.question_temperature,
            max_completion_tokens=self.config.question_max_completion_tokens,
            routing_tag=routing_tag,
        )  
        questions = [chat.assistant_text for chat in question_responses]
        logger.info(f"Time taken to sample question responses: {time.time() - t0} seconds")
        # --- end sample question responses ---

        # (4) Create answer chats
        # --- begin create answer chats ---
        t0 = time.time()
        answer_chats = [
            [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": question,
                },
            ]
            for question in questions
        ]

        answer_responses = self.answer_client.chat_with_logprobs(
            chats=answer_chats,
            temperature=self.config.answer_temperature,
            top_logprobs=self.config.num_top_logprobs,
            max_completion_tokens=self.config.answer_max_completion_tokens,
            routing_tag=routing_tag,
        )
        logger.info(f"Time taken to sample answers: {time.time() - t0} seconds")
        # --- end sample answers ---

        # (4) Construct convos
        # --- begin construct convos ---
        t0 = time.time()
        examples = responses_and_chats_to_training_examples(answer_responses, answer_chats)
        logger.info(
            f"Time taken to construct {len(answer_responses)} convos: {time.time() - t0} seconds"
        )
        # --- end construct convos ---
        return examples

