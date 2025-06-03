from __future__ import annotations

import random
import time
from tqdm import tqdm
from typing import List, Literal, Tuple
import uuid
import math
import concurrent.futures
from transformers import AutoTokenizer


from capsules.clients.base import Client, ClientConfig
from capsules.clients.tokasaurus_batch import CapsulesConvoWithLogprobs
from capsules.context import StructuredContext, list_nested_contexts
from capsules.generate.chunk import Chunker
from capsules.generate.generators.auto import PromptSampler, SYSTEM_PROMPT_TEMPLATE
from capsules.generate.generators.base import (
    ContextConvoGenerator,
    get_subcontext,
    responses_and_chats_to_training_examples,
)
from capsules.generate.outline import get_outline
from capsules.generate.structs import (
    Context,
    Document,
    Section,
    TrainingExample,
)
from capsules.generate.tree_sampler import ContextTreeLeaf, flood_fill_from_leafs, flood_fill_from_leafs_tokens, serialize_with_elide, structured_context_to_context_tree
from capsules.utils import get_logger

logger = get_logger(__name__)

DATA_FORMATS = [
    "JSON",
    "YAML",
    "TOML",
    "INI",
    "XML",
    "plain text",
]


INFO_CATEGORIES = [
  "Patient Identification & Administrative Info",
  "Primary Diagnosis",
  "Comorbid Diagnoses",
  "Presenting Symptoms & Clinical History",
  "Physical Examination Findings",
  "Imaging and Radiology Reports",
  "Laboratory Findings",
  "Histopathology & Molecular Pathology",
  "Cerebrospinal Fluid Analysis",
  "Oncology Treatment Plan",
  "Surgery & Procedures",
  "Medication Lists",
  "Neurology/Neurosurgery Assessments",
  "Psychological Evaluation",
  "Functional and Respiratory Diagnostics",
  "Tumor Board Decisions",
  "Follow-up Recommendations",
  "Discharge Instructions"
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


NOTE_INSTRUCTION_TEMPLATES = [
    (
        "Can you write a summary of all of the information related to {categories} in note {{note_id}} for patient {{patient_id}}? "
        "Be sure to include all information related to dates, times, and numerical values. "
    ),
    (
        "Please extract and organize all information about {categories} from note {{note_id}} for patient {{patient_id}}. "
        "Include specific measurements, test results, and any changes over time."
    ),
    (
        "What are the key findings related to {categories} in note {{note_id}} for patient {{patient_id}}? "
        "Please provide a comprehensive analysis with all relevant details."
    ),
    (
        "In the {{note_id}} dated {{note_date}}, what treatment plans were implemented for {{patient_name}}'s {{condition}}?"
        "Detail the medications, procedures, and their outcomes."
    ),
    (
        "What are the key findings related to {categories} in note {{note_id}} for patient {{patient_name}} ({{patient_id}})? "
        "Please provide a comprehensive analysis with all relevant details."
    ),
    (
        "How do the details in {{note_id}} for {{patient_name}} ({{patient_id}}) fit into the broader context of {{patient_name}}'s medical record?"
    ),
    (
        "What medications were prescribed for {{patient_name}} ({{patient_id}}) in {{note_id}}?"
    ),
    (
        "What were the recommended treatments for {{patient_name}} ({{patient_id}}) in {{note_id}}?"
    ),
    (
        "What changed for {{patient_name}} ({{patient_id}}) in {{note_id}}?"
    ),
    (
        "What was the dose of {{medication}} for {{patient_name}} ({{patient_id}}) in {{note_id}}?"
    )
]

FORMAT_INSTRUCTION_TEMPLATES = [ 
    (
        "Can you structure all of the information in note {{note_id}} for patient {{patient_name}} ({{patient_id}}) related to {categories} "
        f"in the following format: {data_format}? "
        "Be sure to include all information related to the patient and any dates, times, and numerical values."
    )
    for data_format in DATA_FORMATS
]

RECORD_INSTRUCTION_TEMPLATES = [
    (
        "Summarize {{patient_name}}'s medical history focusing specifically on {categories}. "
    ),
    (
        "What are the key findings related to {categories} in note {{note_id}} for patient {{patient_name}} ({{patient_id}})? "
        "Please provide a comprehensive analysis with all relevant details."
    ),
    (
        "Create a timeline of events related to {categories} based on the medical record of {{patient_name_and_id}}. "
        "Include all dates, procedures, and notable changes in condition."
    ),
    (
        "Compare the {{patient_name}}'s condition ({{patient_id}}, DoB: {{patient_dob}}) before and after the treatment for {{condition}}. "
        "What improvements or deteriorations were observed?"
    ),

    (
        "Identify all abnormal findings related to {categories} in note {{note_id}} for patient {{patient_name}} ({{patient_id}}). "
        "Explain their significance and any follow-up actions taken."
    ),
    (
        "Extract all numerical values and measurements related to {categories} from note {{note_id}} for patient {{patient_name}} ({{patient_id}}). "
        "Organize them chronologically and indicate normal/abnormal ranges where available."
    ),
]

INSTRUCTION_TEMPLATES = NOTE_INSTRUCTION_TEMPLATES + FORMAT_INSTRUCTION_TEMPLATES + RECORD_INSTRUCTION_TEMPLATES


QUESTION_PROMPT_TEMPLATE = """Please generate an instruction or question that is grounded in the information provided in the patient panel and the specific note you've been provided.

Here are some examples of instructions and questions:
{instruction_examples}

In your response, fill in the placeholders for {{note_id}}, {{patient_name}}, {{patient_id}} etc. with the appropriate values for the notes you've been provided.

Do not include any text other than the instruction. Do not simply copy one of the examples."""


class LongHealthPromptSampler(PromptSampler):
    class Config(PromptSampler.Config):
        prompt_template: str = QUESTION_PROMPT_TEMPLATE
        initial_system_prompt_template: str = SYSTEM_PROMPT_TEMPLATE
        max_tokens_initial_context: int = 16384

    
    def __init__(self, config: Config, context: StructuredContext, tokenizer: AutoTokenizer):
        self.config = config
        self.context = context
        self.tokenizer = tokenizer
        self.ctxs = list_nested_contexts(self.context)
        self.outline = get_outline(self.context)
    
    def _sample_initial_subcontext(self) -> str:
        # TODO: This can be improved
        path, ctx = random.choice(self.ctxs)
        tokens = self.tokenizer.encode(ctx.text)
        if len(tokens) > self.config.max_tokens_initial_context:
            start_idx = random.randint(0, len(tokens) - self.config.max_tokens_initial_context)
            end_idx = start_idx + self.config.max_tokens_initial_context
            text = self.tokenizer.decode(tokens[start_idx:end_idx])
        else:
            text = ctx.text
        
        return self.config.initial_system_prompt_template.format(
            outline=self.outline, 
            context=text,
            path=path
        )


    def __call__(self, batch_idx: int, num_convos: int) -> List[tuple[str, str]]:

        initial_system_prompt = self._sample_initial_subcontext()

        seed_prompts = []
        for _ in range(num_convos):
            num_examples = random.randint(1, 4)
            instruction_templates = random.choices(INSTRUCTION_TEMPLATES, k=num_examples)
            categories = random.choices(INFO_CATEGORIES, k=num_examples)
            additional_instructions = random.choices(ADDITIONAL_INSTRUCTIONS, k=num_examples)

            instruction_examples = ""
            for instruction_template, category, additional_instruction in zip(instruction_templates, categories, additional_instructions):
                example_instruction = instruction_template.format(
                    categories=category, 
                )
                instruction_examples += f"\n{example_instruction}\n{additional_instruction}\n---\n"
            instruction_examples += "\n\n"

            seed_prompts.append(
                self.config.prompt_template.format(
                    instruction_examples=instruction_examples,
                )
            )
        return initial_system_prompt, seed_prompts




SYSTEM_PROMPT_TEMPLATE = """You are in a conversation about a corpus of information. 
The corpus, with some of the content elided, is given below.
--- begin corpus ---
{corpus}
--- end corpus ---

"""

class LongHealthTreePromptSampler(PromptSampler):
    class Config(PromptSampler.Config):
        prompt_template: str = QUESTION_PROMPT_TEMPLATE
        initial_system_prompt_template: str = SYSTEM_PROMPT_TEMPLATE
        num_focus_leaves_per_context: int = 1
        max_tokens_per_page: int = 256
        max_tokens_in_context: int | Tuple[int, int] = 8192
        sibling_bias: int = 3

    
    def __init__(self, config: Config, context: StructuredContext, tokenizer: AutoTokenizer):
        self.config = config
        self.context = context
        self.tokenizer = tokenizer
        self.ctxs = list_nested_contexts(self.context)
        
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

        return self.config.initial_system_prompt_template.format(corpus=context_str)

 
    def __call__(self, batch_idx: int, num_convos: int) -> List[tuple[str, str]]:

        initial_system_prompt = self._sample_initial_subcontext()

        seed_prompts = []
        for _ in range(num_convos):
            num_examples = random.randint(1, 4)
            instruction_templates = random.choices(INSTRUCTION_TEMPLATES, k=num_examples)
            categories = random.choices(INFO_CATEGORIES, k=num_examples)
            additional_instructions = random.choices(ADDITIONAL_INSTRUCTIONS, k=num_examples)

            instruction_examples = ""
            for instruction_template, category, additional_instruction in zip(instruction_templates, categories, additional_instructions):
                example_instruction = instruction_template.format(
                    categories=category, 
                )
                instruction_examples += f"\n{example_instruction}\n{additional_instruction}\n---\n"
            instruction_examples += "\n\n"

            seed_prompts.append(
                self.config.prompt_template.format(
                    instruction_examples=instruction_examples,
                )
            )
        return initial_system_prompt, seed_prompts
        
