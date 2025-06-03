from __future__ import annotations

import random
import uuid
import re
from collections import Counter, defaultdict

from cartridges.clients.base import ClientConfig
from cartridges.clients.tokasaurus_batch import CartridgesConvoWithLogprobs
from cartridges.generate.generators.base import (
    ContextConvoGenerator,
    responses_and_chats_to_training_examples,
)
from cartridges.structs import (
    Context,
    TrainingExample,
)
from cartridges.utils import get_logger

logger = get_logger(__name__)

DOMAIN_TOPIC = "movies"
DOMAIN_TOPIC_SINGULAR = "movie"
DOMAIN_DOC_TYPE = "website"


CATEGORY_TEMPLATES = [
    f"I am making a relational database of different {DOMAIN_TOPIC}. List the any atributes in this {DOMAIN_DOC_TYPE} that could be used as a column in a table. Be sure to include all column names related to dates, times, and numerical values. ",
    f"I am making a relational database of different {DOMAIN_TOPIC}. List the any atributes in this {DOMAIN_DOC_TYPE} that could be used as a column in a table. Be sure to include all column names related to unstructured fields like descriptions, entities, groups of people, etc.",
    f"I am making a relational database of different {DOMAIN_TOPIC}. List the any atributes in this {DOMAIN_DOC_TYPE} that could be used as a column in a table. Include tail columns, if any.",
    f"This {DOMAIN_DOC_TYPE} describes a {DOMAIN_TOPIC_SINGULAR}. What are the interesting attributes that the {DOMAIN_DOC_TYPE} creator chose to include? List them.",
    f"List the categories of information that this html page shares about the {DOMAIN_TOPIC_SINGULAR}."
]


DATA_FORMATS = [
    "JSON",
    "YAML",
    "TOML",
    "INI",
    "XML",
]

INSTRUCTION_TEMPLATES = [
    (
        "Can you write a summary of all of the information related to {categories} in the following website? "
        "Be sure to include all information related to dates, times, and numerical values. "
    ),
    (
        "Please extract and organize all information about the movie {categories} from this website. "
    ),
    (
        "Summarize the movie details focusing specifically on {categories}. "
    ),
    (
        "What are the key details related to {categories} in this movie's website? "
        "Please provide a comprehensive analysis with all relevant details."
    ),
    (
        "Extract all numerical values related to {categories} from this website. "
        "Organize them sequentially."
    ),
] + [ 
    (
        "Can you structure all of the information in the following website related to {categories} "
        f"in the following format: {data_format}? "
        "Be sure to include all information related to the movie and any dates, times, and numerical values."
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


SYSTEM_PROMPT_TEMPLATE = """You are a movie lover. 
Please reference the following website when answering user questions.
<website>
{context}
</website>
"""


class CategoriesGenerator(ContextConvoGenerator):
    class Config(ContextConvoGenerator.Config):

        client: ClientConfig
        temperature: float = 0.0
        max_completion_tokens: int = 512
        prompt_template: str = PROMPT_TEMPLATE
        system_prompt_template: str = SYSTEM_PROMPT_TEMPLATE
        num_top_logprobs: int = 20

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.config = config
        self.client = config.client.instantiate()

    def sample_convos(self, batch_idx, num_convos, total_batches):
        return []


    def stage_1_postprocess(self, convos):
        counter = Counter(convos)
        categories = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        categories_list = [cat[0] for cat in categories]
        return categories_list
    

    def sample_convos_stage_1(
        self,
        batch_idx: int,
        num_convos: int,
        total_batches: int,
        **kwargs,
    ) -> list[TrainingExample]:
        routing_tag = str(uuid.uuid4())
        
        sample_idx = random.randint(0, len(self.context.sections) - 1)
        section = self.context.sections[sample_idx]
        
        prompts = CATEGORY_TEMPLATES
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
        convos: list[CartridgesConvoWithLogprobs] = self.client.chat_with_logprobs(
            chats=answer_chats,
            temperature=self.config.temperature,
            top_logprobs=1,
            max_completion_tokens=self.config.max_completion_tokens,
            routing_tag=routing_tag,
        )

        def clean_categories(convos):
            cleaned_categories = []

            print(f"Found {len(convos)} convos")
            for idx, convo in enumerate(convos):
                text = convo.assistant_text
                items = text.split("\n")
                print(f"-- Starts {len(items)}  items for convo {idx}")

                added_items = []
                for item in items:
                    split_items = item.split("|")
                    added_items.extend(split_items)
                items = added_items

                added_items = []
                for item in items:
                    split_items = item.split(" and ")
                    added_items.extend(split_items)
                items = added_items

                for i in range(2):

                    # eliminate empty strings
                    items = [item for item in items if item]

                    # eliminate descriptions
                    items = [item.split("-")[0] for item in items]
                    items = [item.split(":")[0] for item in items]

                    # eliminate starting bullets
                    items = [item.replace("*", "").replace(":", "").replace("-", "").replace("'", "").replace("`", "").replace('"', "").replace("_", " ").strip().lower() for item in items]

                    # eliminate starting digits
                    no_number_items = []
                    for item in items:
                        if len(item) > 2 and item[0:2].isdigit():
                            no_number_items.append(item[2:])
                        elif len(item) > 1 and item[0].isdigit():
                            no_number_items.append(item[1:])
                        else:
                            no_number_items.append(item)
                    items = [item.strip().strip(".").strip("") for item in no_number_items]
                    
                    # cut anything in parentheses
                    items = [re.sub(r"\(.*?\)", "", item).strip() for item in items]
                    items = [item.split("(")[0] for item in items]

                    # cut prompt artifacts
                    items = [item for item in items if 'column' not in item]

                print(f"-- Adding {len(items)} items for convo {idx}")
                cleaned_categories.extend(items)

            print(f"Found {len(cleaned_categories)} categories")
            return cleaned_categories
        
        return clean_categories(convos)
    

    def sample_convos_stage_2(
        self,
        batch_idx: int,
        num_convos: int,
        total_batches: int,
        INFO_CATEGORIES,
        **kwargs,
    ) -> list[TrainingExample]:
        routing_tag = str(uuid.uuid4())

        INFO_CATEGORIES = INFO_CATEGORIES[:100]
        
        # (1) sample a subcontext  provide to the question generators
        sample_idx = random.randint(0, len(self.context.sections) - 1)
        section = self.context.sections[sample_idx]
            
        # (2) sample instructions
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
        print(f"Generated {len(prompts)} prompts")

        # (3) Sample answers
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
        print(f"Generated {len(answer_chats)} answer chats")
        convos: list[CartridgesConvoWithLogprobs] = self.client.chat_with_logprobs(
            chats=answer_chats,
            temperature=self.config.temperature,
            top_logprobs=self.config.num_top_logprobs,
            max_completion_tokens=self.config.max_completion_tokens,
            routing_tag=routing_tag,
        )
        print(f"Generated {len(convos)} convos")

        # (4) Construct convos
        examples = responses_and_chats_to_training_examples(convos, answer_chats)
        return examples


