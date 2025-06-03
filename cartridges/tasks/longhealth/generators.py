from __future__ import annotations

import random
import time
from tqdm import tqdm
from typing import List, Literal
import uuid
import math
import concurrent.futures


from capsules.clients.base import Client, ClientConfig
from capsules.clients.tokasaurus_batch import CapsulesConvoWithLogprobs
from capsules.generate.chunk import Chunker
from capsules.generate.generators.base import (
    ContextConvoGenerator,
    get_subcontext,
    responses_and_chats_to_training_examples,
)
from capsules.generate.structs import (
    Context,
    Document,
    Section,
    TrainingExample,
)
from capsules.generate.subcontext import SubcontextGenerator
from capsules.utils import get_logger

logger = get_logger(__name__)


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

DATA_FORMATS = [
    "JSON",
    "YAML",
    "TOML",
    "INI",
    "XML",
    # "MessagePack",
    # "CSV",
]


INSTRUCTION_TEMPLATES = [
    (
        "Can you write a summary of all of the information related to {categories} in the following note? "
        "Be sure to include all information related to dates, times, and numerical values. "
    ),
    (
        "Please extract and organize all information about the patient's {categories} from this medical record. "
        "Include specific measurements, test results, and any changes over time."
    ),
    (
        "Summarize the patient's medical history focusing specifically on {categories}. "
        "Highlight any significant events, treatments, or changes in condition."
    ),
    (
        "What are the key findings related to {categories} in this patient's record? "
        "Please provide a comprehensive analysis with all relevant details."
    ),
    (
        "Create a timeline of events related to the patient's {categories} based on the medical record. "
        "Include all dates, procedures, and notable changes in condition."
    ),
    (
        "Compare the patient's condition before and after the treatment for {categories}. "
        "What improvements or deteriorations were observed?"
    ),
    (
        "What treatment plans were implemented for the patient's {categories}? "
        "Detail the medications, procedures, and their outcomes."
    ),
    (
        "Identify all abnormal findings related to {categories} in this medical record. "
        "Explain their significance and any follow-up actions taken."
    ),
    (
        "Extract all numerical values and measurements related to {categories} from this record. "
        "Organize them chronologically and indicate normal/abnormal ranges where available."
    ),
] + [ 
    (
        "Can you structure all of the information in the following note related to {categories} "
        f"in the following format: {data_format}? "
        "Be sure to include all information related to the patient and any dates, times, and numerical values."
    )
    for data_format in DATA_FORMATS
]


PROMPT_TEMPLATE = """{instruction}{additional_instructions}"""


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

SYSTEM_PROMPT_TEMPLATE = """You are a medical expert. 
Please reference the following medical note when answering user questions.
<medical-note>
{context}
</medical-note>
"""


class ReformatGenerator(ContextConvoGenerator):
    class Config(ContextConvoGenerator.Config):

        client: ClientConfig
        temperature: float = 0.0
        max_completion_tokens: int = 512
        # The system prompt for the answer generator. It can contain
        # variables for `{instruction}` and `{additional_instructions}`.
        prompt_template: str = PROMPT_TEMPLATE

        # The system prompt for the answer generator. It can contain
        # variables for `{context}`.
        system_prompt_template: str = SYSTEM_PROMPT_TEMPLATE

        subcontext: Literal["note", "patient"] = "note"

        num_top_logprobs: int = 20

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.config = config

        self.client = config.client.instantiate()



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
        if self.config.subcontext == "note":
            sample_idx = random.randint(0, len(self.context.sections) - 1)
            section = self.context.sections[sample_idx]
        else:
            patients = set([section.path.split("/")[0] for section in self.context.sections])
            patient = random.choice(list(patients))
            sections = [section for section in self.context.sections if section.path.split("/")[0] == patient]
            context = Context(
                title=f"LongHealth-{patient}",
                sections=sections,
            )
            section = context.to_string()
            
        logger.info(f"Time taken to get subcontext: {time.time() - t0} seconds")
        # --- end get subcontext ---

        # (2) sample instructions
        # --- begin sample instructions ---
        t0 = time.time()
        instruction_templates = random.choices(INSTRUCTION_TEMPLATES, k=num_convos )
        additional_instructions = random.choices(ADDITIONAL_INSTRUCTIONS, k=num_convos)
        prompts = []
        for instruction_template, additional_instruction in zip(instruction_templates, additional_instructions):
            categories = random.choices(INFO_CATEGORIES, k=random.randint(1, 3))
            instruction = instruction_template.format(categories=", ".join(categories))
            prompts.append(
                self.config.prompt_template.format(
                    instruction=instruction,
                    additional_instructions=additional_instruction,
                )
            )
        logger.info(f"Time taken to sample instructions: {time.time() - t0} seconds")
        # --- end sample instructions ---

        # (3) Sample answers
        # --- begin sample answers ---
        t0 = time.time()
        system_prompt = self.config.system_prompt_template.format(context=section.to_string())
        answer_chats = [
            [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
            for prompt in prompts
        ]

        convos: list[CapsulesConvoWithLogprobs] = self.client.chat_with_logprobs(
            chats=answer_chats,
            temperature=self.config.temperature,
            top_logprobs=self.config.num_top_logprobs,
            max_completion_tokens=self.config.max_completion_tokens,
            routing_tag=routing_tag,
        )
        logger.info(f"Time taken to sample answers: {time.time() - t0} seconds")
        # --- end sample answers ---

        # (4) Construct convos
        # --- begin construct convos ---
        t0 = time.time()

        examples = responses_and_chats_to_training_examples(convos, answer_chats)
        logger.info(
            f"Time taken to construct {len(convos)} convos: {time.time() - t0} seconds"
        )
        # --- end construct convos ---
        return examples


SYSTEM_PROMPT_TEMPLATE = """You are an assistant to a physician with the following patient panel.
{context}""" 



class ReformatWithToCGenerator(ContextConvoGenerator):
    class Config(ContextConvoGenerator.Config):

        client: ClientConfig
        temperature: float = 0.0
        max_completion_tokens: int = 512
        # The system prompt for the answer generator. It can contain
        # variables for `{instruction}` and `{additional_instructions}`.
        prompt_template: str = PROMPT_TEMPLATE

        # The system prompt for the answer generator. It can contain
        # variables for `{context}`.
        system_prompt_template: str = SYSTEM_PROMPT_TEMPLATE

        subcontext_type: Literal["note", "patient", "both"] = "note"
        subcontext_frac_note: float = 0.8 # Fraction of convos that use the note as the subcontext if subcontext_type is "both"

        num_top_logprobs: int = 20

        toc_num_samples: int = 16
        toc_max_num_batches_in_parallel: int = 1
        toc_batch_size: int = 16
        toc_max_completion_tokens: int = 256

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.config = config

        self.client = config.client.instantiate()


        self.tocs: List[str] = sample_tocs(
            client=self.client,
            context=self.context,
            temperature=self.config.temperature,
            max_completion_tokens=self.config.toc_max_completion_tokens,
            num_samples=self.config.toc_num_samples,
            max_num_batches_in_parallel=self.config.toc_max_num_batches_in_parallel,
            batch_size=self.config.toc_batch_size,
        )

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
        toc = random.choice(self.tocs)

        subcontext_type = self.config.subcontext_type
        if self.config.subcontext_type == "both":
            subcontext_type = "note" if random.random() < self.config.subcontext_frac_note else "patient"

        if subcontext_type == "note":
            sample_idx = random.randint(0, len(self.context.sections) - 1)
            section = self.context.sections[sample_idx]

            patient_id, note_idx = section.path.split("/")
            
            situation = f"Below is the note for {patient_id} with text file name {note_idx}. \n {section.desc}"
            content = section.content
        else:
            patients = set([section.path.split("/")[0] for section in self.context.sections])
            patient = random.choice(list(patients))
            sections = [section for section in self.context.sections if section.path.split("/")[0] == patient]
            context = Context(
                title=f"LongHealth-{patient}",
                sections=sections,
            )
            situation = f"Below are all of the notes for {patient}."
            content = context.to_string()
        
        subcontext = f"Here is the table of contents of the patient panel: \n {toc} \n {situation} \n---\n {content}"

            
        logger.info(f"Time taken to get subcontext: {time.time() - t0} seconds")
        # --- end get subcontext ---

        # (2) sample instructions
        # --- begin sample instructions ---
        t0 = time.time()
        instruction_templates = random.choices(INSTRUCTION_TEMPLATES, k=num_convos )
        additional_instructions = random.choices(ADDITIONAL_INSTRUCTIONS, k=num_convos)
        prompts = []
        for instruction_template, additional_instruction in zip(instruction_templates, additional_instructions):
            categories = random.choices(INFO_CATEGORIES, k=random.randint(1, 3))
            instruction = instruction_template.format(categories=", ".join(categories))
            prompts.append(
                self.config.prompt_template.format(
                    instruction=instruction,
                    additional_instructions=additional_instruction,
                )
            )
        logger.info(f"Time taken to sample instructions: {time.time() - t0} seconds")
        # --- end sample instructions ---

        # (3) Sample answers
        # --- begin sample answers ---
        t0 = time.time()
        system_prompt = self.config.system_prompt_template.format(context=subcontext)
        answer_chats = [
            [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
            for prompt in prompts
        ]

        convos: list[CapsulesConvoWithLogprobs] = self.client.chat_with_logprobs(
            chats=answer_chats,
            temperature=self.config.temperature,
            top_logprobs=self.config.num_top_logprobs,
            max_completion_tokens=self.config.max_completion_tokens,
            routing_tag=routing_tag,
        )
        logger.info(f"Time taken to sample answers: {time.time() - t0} seconds")
        # --- end sample answers ---

        # (4) Construct convos
        # --- begin construct convos ---
        t0 = time.time()

        examples = responses_and_chats_to_training_examples(convos, answer_chats)
        logger.info(
            f"Time taken to construct {len(convos)} convos: {time.time() - t0} seconds"
        )
        # --- end construct convos ---
        return examples




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
] + [ 
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


QUESTION_PROMPT_TEMPLATE = """Please generate an instruction or question that is grounded in the information provided in the patient panel and the specific note you've been provided.

Here are some examples of instructions and questions:
{instruction_examples}


In your response, fill in the placeholders for {{note_id}}, {{patient_name}}, {{patient_id}} etc. with the appropriate values:
{details}

Do not include any other other than the instruction. Do not simply copy the examples."""

class GeneratedReformatWithToCGenerator(ContextConvoGenerator):
    class Config(ContextConvoGenerator.Config):

        question_client: ClientConfig
        question_temperature: float = 0.0
        question_max_completion_tokens: int = 512
        question_prompt_template: str = QUESTION_PROMPT_TEMPLATE

        answer_client: ClientConfig
        answer_temperature: float = 0.0
        answer_max_completion_tokens: int = 512
        # The system prompt for the answer generator. It can contain
        # variables for `{instruction}` and `{additional_instructions}`.
        answer_prompt_template: str = PROMPT_TEMPLATE

        # The system prompt for the answer generator. It can contain
        # variables for `{context}`.
        system_prompt_template: str = SYSTEM_PROMPT_TEMPLATE
        additional_instructions: List[str] = ADDITIONAL_INSTRUCTIONS

        subcontext_type: Literal["note", "patient", "both"] = "note"
        subcontext_frac_note: float = 0.8 # Fraction of convos that use the note as the subcontext if subcontext_type is "both"

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


        self.tocs: List[str] = sample_tocs(
            client=self.answer_client,
            context=self.context,
            temperature=self.config.answer_temperature,
            max_completion_tokens=self.config.toc_max_completion_tokens,
            num_samples=self.config.toc_num_samples,
            max_num_batches_in_parallel=self.config.toc_max_num_batches_in_parallel,
            batch_size=self.config.toc_batch_size,
        )

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
        toc = random.choice(self.tocs)

        subcontext_type = self.config.subcontext_type
        if self.config.subcontext_type == "both":
            subcontext_type = "note" if random.random() < self.config.subcontext_frac_note else "patient"

        if subcontext_type == "note":
            sample_idx = random.randint(0, len(self.context.sections) - 1)
            section = self.context.sections[sample_idx]

            patient_id, note_idx = section.path.split("/")
            
            situation = f"Below is the note for {patient_id} with text file name {note_idx}. \n {section.desc}"
            details = section.desc
            content = section.content
            instruction_templates = NOTE_INSTRUCTION_TEMPLATES
        else:
            patients = set([section.path.split("/")[0] for section in self.context.sections])
            patient = random.choice(list(patients))
            sections = [section for section in self.context.sections if section.path.split("/")[0] == patient]
            context = Context(
                title=f"LongHealth-{patient}",
                sections=sections,
            )
            situation = f"Below are all of the notes for {patient}."
            details = sections[0].desc
            content = context.to_string()
            instruction_templates = RECORD_INSTRUCTION_TEMPLATES
        subcontext = f"Here is the table of contents of the patient panel: \n {toc} \n {situation} \n---\n {content}"    
        system_prompt = self.config.system_prompt_template.format(context=subcontext)
        logger.info(f"Time taken to get subcontext: {time.time() - t0} seconds")
        # --- end get subcontext ---

        # (2) Create question chats
        # --- begin create question chats ---
        t0 = time.time()
        question_chats = []
        for _ in range(num_convos):
            num_examples = random.randint(1, 4)
            instruction_templates = random.choices(instruction_templates, k=num_examples)
            categories = random.choices(INFO_CATEGORIES, k=num_examples)
            additional_instructions = random.choices(self.config.additional_instructions, k=num_examples)

            instruction_examples = "Instruction Examples:\n\n"
            for instruction_template, category, additional_instruction in zip(instruction_templates, categories, additional_instructions):
                example_instruction = instruction_template.format(
                    categories=category, 
                )
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
                            details=details
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



from collections import defaultdict


COT_SUMMARY_PROMPT_TEMPLATE ="""Write a 1-3 sentence summary of the following medical note between <current-note> and </current-note>. 
We also include the summaries of prior notes to inform your summary. Do not summarize the prior notes, only use them to inform your summary of the current note.
You do not need to repeat information from the prior notes in your summary.

<prior-note-summaries>
{prior_note_summaries}
</prior-note-summaries>

<current-note>
{note}
</current-note>
Output the summary and nothing else."""

def sample_tocs(
    client: Client,
    context: Context,
    
    prompt_template: str = COT_SUMMARY_PROMPT_TEMPLATE,
    temperature: float = 0.6,
    max_completion_tokens: int = 256,
    
    num_samples: int = 16,
    max_num_batches_in_parallel: int = 1,
    batch_size: int = 16,
) -> list[str]:

    # (1) Create mapping from patient to sections
    # --- begin create mappings ---
    patient_to_sections = defaultdict(list)
    for section in context.sections:
        patient_id = section.path.split("/")[0] 
        patient_to_sections[patient_id].append(section)
    # --- end create mappings ---

    # (2) Sample summaries
    # --- begin sample summaries ---
    def summaries_to_str(summaries):
        if len(summaries) > 0:
            summaries_str = "\n\n".join(
                [f"<prior-note-summary>\n{summary['summary']}\n</prior-note-summary>" for summary in summaries]
            )
        else:
            summaries_str = "none - this is the first note"
        return summaries_str

    all_patient_to_summaries = [{} for _ in range(num_samples)]
    for patient_id, sections in tqdm(
        patient_to_sections.items(), 
        desc="Generating ToCs (Total # of patients: {})".format(len(patient_to_sections))
    ):
        all_summaries = [[] for _ in range(num_samples)]
        for section in sections:

            # (2.1) Construct chats for current note appending summaries of prior notes
            # --- begin construct chats ---
            t0 = time.time()
            chats = [
                [
                    {
                        "role": "user", 
                        "content": prompt_template.format(
                            prior_note_summaries=summaries_to_str(summaries),
                            note=section.to_string()
                        )
                    }
                ] for summaries in all_summaries
            ]
            logger.info(f"ToC [2.1] - Time taken to construct chats: {time.time() - t0} seconds")
            # --- end construct chats ---

            # (2.2) Execute the chats in parallel with a thread pool
            # --- begin execute chats ---
            t0 = time.time()
            all_samples = [None] * len(chats)  # Pre-allocate list to preserve order
            futures = {}  # Dictionary to map futures to their original indices

            num_total_chats = len(chats)
            num_batches = math.ceil(num_total_chats / batch_size)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_num_batches_in_parallel) as executor:
                for i in range(num_batches):
                    start_index = i * batch_size
                    end_index = min((i + 1) * batch_size, num_total_chats)
                    batch_chats = chats[start_index:end_index]

                    if not batch_chats:
                        continue

                    future = executor.submit(
                        client.chat_with_logprobs,
                        chats=batch_chats,
                        max_completion_tokens=max_completion_tokens,
                        top_logprobs=1,
                        temperature=temperature,
                        routing_tag=patient_id + f"-{i}",
                    )
                    futures[future] = (start_index, end_index)

                # Process completed futures as they finish, maintaining order
                for future in concurrent.futures.as_completed(futures):
                    start_index, end_index = futures[future]
                    try:
                        result = future.result()
                        batch_samples = [x.assistant_text for x in result]
                        # Place the results in the correct position in the pre-allocated list
                        all_samples[start_index:end_index] = batch_samples
                    except Exception as e:
                        # Handle exceptions if needed, e.g., log error and fill with placeholders
                        print(f"Batch starting at index {start_index} failed: {e}")
            logger.info(f"ToC [2.2] - Time taken to execute chats: {time.time() - t0} seconds")
            # --- end execute chats ---


            # (2.3) Assign the ordered results back to the original variable name
            # --- begin assign results ---
            samples = all_samples # Assign the ordered results back to the original variable name
            for summaries, sample in zip(all_summaries, samples):
                summaries.append({
                    "section": section,
                    "summary": sample
                })
            # --- end assign results ---

            # (2.4) Assign the ordered results back to the original variable name
            # --- begin assign results ---
            for patient_to_summaries, summaries in zip(all_patient_to_summaries, all_summaries):
                patient_to_summaries[patient_id] = summaries
            # --- end assign results ---    

    # (3) Construct TOC strings
    # --- begin construct TOC strings ---
    tocs = []
    for patient_to_summaries in all_patient_to_summaries:
        patients_str = "- Patient Panel\n"

        for patient_id, summaries in patient_to_summaries.items():

            patients_str += f"\t- {patient_id}: {summaries[0]['section'].desc}\n"
            for note_idx, summary in enumerate(summaries):
                patients_str += f"\t\t-  text_{note_idx}.txt: {summary['summary']}\n"
        tocs.append(patients_str)
    # --- end construct TOC strings ---
    return tocs
