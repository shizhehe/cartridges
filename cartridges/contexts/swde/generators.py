from __future__ import annotations

import random
import uuid

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



handcrafted_questions = {
    "release date": "What is the release data of this movie?",
    "quotes": "What are the key quotes from this movie?",
    "recommendations": "What are the listed recommendations based on this movie?",
    "sound mix": "What is the sound mix of this movie?",
    "color": "What is listed as the color attribute in the html?",
    "frequently asked questions": "What are some of the frequently asked questions for this movie?",
    "filming locations": "What are the filming locations for this movie?",
    "title": "What is the title of this movie?",
    "stars": "Who are the stars of this movie?",
    "genres": "What are the listed genre(s) for this movie?",
    "rating": "What is the rating of this movie?",
    "cast": "Who are the cast members of this movie?",
    "trivia": "What are some mentioned trivia for this movie?",
    "topic_entity_name": "What is the topic entity name for this movie?",
    "motion picture rating": "What is the motion picture rating for this movie?",
    "goofs": "What are the listed goofs for this movie?",
    "release date": "What is the release date of this movie?",
    "plot keywords": "What are the listed plot keywords for this movie?",
    "soundtracks": "What are the soundtracks for this movie?",
    "production co": "What are the production companies for this movie?",
    "opening weekend": "What are the opening weekend details for this movie?",
    "budget": "What is the budget for this movie?",
    "connections": "What are some of the listed connections for this movie?",
    "users": "What are the 'users' details for this movie?",
    "runtime" : "What is the runtime of this movie?",
    "critics": "What are the 'critics' details for this movie?",
    "language": "In what language(s) is this movie available?",
    "gross": "What is the gross amount for this movie?",
    "related news": "What are some of the related news listed for this movie?",
    "also known as": "What is this movie 'also known as'?",
    "writers": "Who are the writers of this movie?",
    "storyline": "What is the 'storyline' of this movie?",
    "message boards": "What is listed on the message boards for this movie?",
    "related lists": "What are some of the related lists for this movie?",
    "aspect ratio": "What is the aspect ratio of this movie?",
    "moviemeter": "What is reported for 'moviemeter' in this webpage?",
    "year": "In what year was this movie released?",
    "country": "Which country is this movie from?",
    "director": "Who are the director(s) of this movie?",
    "taglines": "What is the tagline of this movie?",
}

INFO_CATEGORIES = list(handcrafted_questions.keys())

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


class ReformatGenerator(ContextConvoGenerator):
    class Config(ContextConvoGenerator.Config):

        client: ClientConfig
        temperature: float = 0.0
        max_completion_tokens: int = 512
        # The system prompt for the answer generator. It can contain variables for `{instruction}` and `{additional_instructions}`.
        prompt_template: str = PROMPT_TEMPLATE
        # The system prompt for the answer generator. It can contain variables for `{context}`.
        system_prompt_template: str = SYSTEM_PROMPT_TEMPLATE
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
        convos: list[CartridgesConvoWithLogprobs] = self.client.chat_with_logprobs(
            chats=answer_chats,
            temperature=self.config.temperature,
            top_logprobs=self.config.num_top_logprobs,
            max_completion_tokens=self.config.max_completion_tokens,
            routing_tag=routing_tag,
        )

        # (4) Construct convos
        examples = responses_and_chats_to_training_examples(convos, answer_chats)
        return examples


