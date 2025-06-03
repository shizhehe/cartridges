from __future__ import annotations

import random
import uuid
import numpy as np
import ast
import json
import re
from collections import defaultdict
from transformers import AutoTokenizer


from cartridges.tasks.mrcr import get_toc_str, get_convo_map, occ_mapping, POSITIOINS
from cartridges.clients.base import ClientConfig
from cartridges.clients.tokasaurus_batch import CartridgesConvoWithLogprobs
from cartridges.generate.generators.base import ContextConvoGenerator
from cartridges.generate.generators.base import responses_and_chats_to_training_examples as responses_and_chats_to_training_examples_orig
from cartridges.structs import (
    Context,
    TrainingExample,
)
from cartridges.utils import get_logger

logger = get_logger(__name__)


mrcr_path = "Cartridges/tasks/mrcr/mrcr_categories_document-2.json"


# UTILS

USE_CHAT_TAG = False


def responses_and_chats_to_training_examples(
    convos: list[CartridgesConvoWithLogprobs],
    answer_chats: list[list[dict]],
) -> list[TrainingExample]:
    examples = []
    for convo_response, chat in zip(
        convos,
        answer_chats,
        strict=True,
    ):
        messages = chat[1:] + [
            {
                "role": "assistant", "content": convo_response.assistant_text
            }
        ]

        header_locations = np.where(convo_response.token_ids == 128006)[0].tolist()
        prefix_end_idx = header_locations[1]
        final_user_token = header_locations[3]

        truncation_len = len(convo_response.token_ids) - final_user_token
        token_ids = convo_response.token_ids[prefix_end_idx:final_user_token]
        top_logprob_logprob = convo_response.top_logprob_logprobs[:-truncation_len, :].astype(np.float32)
        top_logprob_ids = convo_response.top_logprob_ids[:-truncation_len, :]

        try:

            assert len(token_ids) > convo_response.num_output_tokens
            assert top_logprob_logprob.shape == top_logprob_ids.shape
            assert top_logprob_logprob.shape[0] == len(token_ids) - 1, "You probably need to pull down on tokasaurus or your first message is not a system message"

        except:

            continue

        examples.append(
            TrainingExample(
                messages=[TrainingExample.Message(**message) for message in messages],
                top_logprob_ids=top_logprob_ids,
                top_logprob_logprobs=top_logprob_logprob,  # We can convert to float32 to save space in the file
                token_ids=token_ids,
                num_output_tokens=convo_response.num_output_tokens,
                type="todo",
                metadata={},
            )
        )
    return examples



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


DATA_FORMATS = [
    "JSON",
    "YAML",
    "TOML",
    "INI",
    "XML",
]


INSTRUCTION_TEMPLATES = [
    "Please extract and organize all information from this passage. "
    "Summarize the passage details focusing. ",
] + [ 
        "Can you structure all of the information in the following passage "
        f"in the following format: {data_format}? "
        "Be sure to include all information related to the movie and any dates, times, and numerical values."
    
        for data_format in DATA_FORMATS
]



# Slice: Understanding document structure - combining different passage types.
class CategoryGenerator(ContextConvoGenerator):
    class Config(ContextConvoGenerator.Config):

        client: ClientConfig
        temperature: float = 0.0
        max_completion_tokens: int = 512
        prompt_template: str = ""
        system_prompt_template: str = ""
        num_top_logprobs: int = 20

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.config = config
        self.client = config.client.instantiate()

    def sample_convos(self, batch_idx, num_convos, total_batches):
        return []

    def stage_1_postprocess(self, convos):
        return convos

    def sample_convos_stage_1(
        self,
        batch_idx: int,
        num_convos: int,
        total_batches: int,
        **kwargs,
    ) -> list[TrainingExample]:
        
        routing_tag = str(uuid.uuid4())

        if batch_idx >=len(self.context.sections): return []
        section = self.context.sections[batch_idx]

        try:
            with open(mrcr_path, "r") as f:
                INFO_CATEGORIES_SAVED = json.load(f)

            print(f"Found previously saved categories")
            return [INFO_CATEGORIES_SAVED]
        except:
            pass

        print(f"Adding {num_convos} convos to batch {batch_idx}")
        sub_sections = section.content.strip("\n").split("\n")
        user_query = sub_sections[1].replace("User:", "").strip()
        assistant_response = "\n".join(sub_sections[2:]).replace("Assistant:", "").strip()
        toc_str = get_toc_str(self.context.sections)

        print(toc_str)

        prompt = """I am going to show you a conversation between a user and an assistant. Please categorize the conversation history into groups by topic.

The tags in the conversation history are as follows:
{toc_str}

Put the conversations that are about the same topic into the same category. Include each conversation only once.
"""

        answer_chats = []
        answer_chat = [
            {
                "role": "system",
                "content": f"",
            },
            {
                "role": "user",
                "content": prompt.format(
                    toc_str=toc_str,
                    user_query=user_query,
                    assistant_response=assistant_response,
                ),
            },
        ]
        answer_chats.append(answer_chat)
        convos: list[CartridgesConvoWithLogprobs] = self.client.chat_with_logprobs(
            chats=answer_chats,
            temperature=self.config.temperature,
            top_logprobs=self.config.num_top_logprobs,
            max_completion_tokens=self.config.max_completion_tokens,
            routing_tag=routing_tag,
        )

        # process_categories
        convos = convos[0].assistant_text

        # reformat 
        prompt = """I am going to show you some categories and conversation tags from a conversation between a user and assistant. You need to put the data into a json object where the category title is a key and the exact list of conversation tags is the value.

The categories and conversation tags are as follows:
{convos}

Output the json and no other text."""

        answer_chats = []
        for i in range(num_convos):

            answer_chat = [
                {
                    "role": "system",
                    "content": f"",
                },
                {
                    "role": "user",
                    "content": prompt.format(
                        convos=convos,
                    ),
                },
            ]
            answer_chats.append(answer_chat)

            if i == 1: 
                break

        convos_categorized: list[CartridgesConvoWithLogprobs] = self.client.chat_with_logprobs(
            chats=answer_chats,
            temperature=self.config.temperature,
            top_logprobs=self.config.num_top_logprobs,
            max_completion_tokens=self.config.max_completion_tokens,
            routing_tag=routing_tag,
        )
        
        # process_categories
        convos_categories = convos_categorized[0].assistant_text

        match = re.search(r"```(.*?)```", convos_categories, re.DOTALL)
        if match:
            json_block = match.group(1).strip()
            data = json.loads(json_block)
            print(json.dumps(data, indent=2))
        else:
            data = None
            print("No JSON block found.")

        if data is None:
            try:
                text  = "{\n" + convos_categories.split("{")[1].strip("```")
                data = ast.literal_eval(text)
            except:
                data = None

        if data is None:
            print(convos_categories)
            assert 0, "Failed to parse json"

        return [data]

    def sample_convos_stage_2(
        self,
        batch_idx: int,
        num_convos: int,
        total_batches: int,
        INFO_CATEGORIES,
        **kwargs,
    ) -> list[TrainingExample]:
        
        routing_tag = str(uuid.uuid4())

        # get data
        if batch_idx >=len(self.context.sections): return []

        convo_map = get_convo_map(self.context.sections)

        system_prompt = """
I am going to show you a subset of messages about {category_title} from a conversation between a user and an assistant. 

The conversations in this category are as follows:
{history_string}
"""

        prompt = """Please output the {pos} assistant response to the query '{user_query}'."""

        INFO_CATEGORIES_SAVED = defaultdict(dict)

        def nearest_key(d, key):
            from collections import Counter
            best_key = None
            best_f1 = 0
            for k in d.keys():
                pred_key = key.replace("'", "")
                gold_key = k.replace("'", "")
                common = Counter(pred_key) & Counter(gold_key)
                num_same = sum(common.values())
                if num_same > 0:
                    precision = num_same / len(pred_key)
                    recall = num_same / len(gold_key)
                    f1 = 2 * (precision * recall) / (precision + recall)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_key = k
            return best_key

        def add_new_queries(titles_list, title2occ, prompts, system_prompts, answers):
            idx = random.randint(1, num)
            title = titles_list[idx-1]
            title_pos = title2occ[title]

            try:
                answer = convo_map[title.lower()]
            except:
                answer = convo_map[nearest_key(convo_map, title.lower())]
                
            user_query_str = title.replace("write a", "").replace(",  ", ", ")
            user_query_str = user_query_str.split(", ")[-1]

            history = []
            for title in titles_list:
                try:
                    response = convo_map[title.lower()]
                except:
                    response = convo_map[nearest_key(convo_map, title.lower())]
                convo_tag = title.replace(",  ", ", ").replace(",  ", ", ")
                history.append(f"{convo_tag}\nUser: {title}\nAssistant: {response}")
            history_string = "\n".join(history)
            
            prompts.append(prompt.format(
                pos=title_pos, user_query=user_query_str, category_title=category_title)
            )
            system_prompts.append(system_prompt.format(
                history_string=history_string,category_title=category_title,)
            )
            answers.append(f"{answer}")
            return prompts, system_prompts, answers

        prompts = []
        answers = []
        system_prompts = []
        for category_title, titles in INFO_CATEGORIES[0].items():
            print(f"Category: {category_title}; Num titles: {len(titles)}")

            num_titles = len(titles)
            if type(titles) == dict:
                INFO_CATEGORIES_SAVED[category_title] = titles
                title2occ = titles
                titles = [title.replace("'", "") for title in titles]
            else:
                titles = [title.replace("'", "") for title in titles]
                title2occ = {title: occ_mapping[i+1] for i, title in enumerate(titles)}
                INFO_CATEGORIES_SAVED[category_title] = title2occ

            # subsets in context
            for i in range(50):
                num = random.randint(1, min(3, num_titles))
                titles_subset = random.sample(titles, num)
                prompts, system_prompts, answers = add_new_queries(
                    titles_subset,
                    title2occ,
                    prompts,
                    system_prompts,
                    answers
                )

            # subsets in context
            for i in range(50):
                num = num_titles
                titles_subset = random.sample(titles, num)
                prompts, system_prompts, answers = add_new_queries(
                    titles_subset,
                    title2occ,
                    prompts,
                    system_prompts,
                    answers
                )   

        with open(mrcr_path, "w") as f:
            json.dump(INFO_CATEGORIES_SAVED, f, indent=2)


        answer_chats = []
        for i, (system_prompt, question, answer) in enumerate(zip(system_prompts, prompts, answers)):

            answer_chat = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": question 
                },
                {
                    "role": "assistant",
                    "content": answer
                },
                {
                    "role": "user",
                    "content": ""
                },
            ]
            answer_chats.append(answer_chat)


        print(f"Generated {len(answer_chats)} answer chats")
        convos: list[CartridgesConvoWithLogprobs] = self.client.chat_with_logprobs(
            chats=answer_chats,
            temperature=self.config.temperature,
            top_logprobs=self.config.num_top_logprobs,
            max_completion_tokens=self.config.max_completion_tokens,
            routing_tag=routing_tag,
        )

        print(convos[0].assistant_text)

        # print(f"Generated {len(convos)} convos")
        examples = responses_and_chats_to_training_examples(convos, answer_chats)
        print(f"Generated {len(examples)} examples")
        return examples



# Slice: Simple QA slice.
class SimpleQAGenerator(ContextConvoGenerator):
    class Config(ContextConvoGenerator.Config):

        client: ClientConfig
        temperature: float = 0.0
        max_completion_tokens: int = 512
        prompt_template: str = ""
        system_prompt_template: str = ""
        num_top_logprobs: int = 20
        question_max_completion_tokens: int = 128

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.config = config
        self.client = config.client.instantiate()

    def sample_convos(self, batch_idx, num_convos, total_batches):
        return []

    def stage_1_postprocess(self, convos):
        return convos

    def sample_convos_stage_1(
        self,
        batch_idx: int,
        num_convos: int,
        total_batches: int,
        **kwargs,
    ) -> list[TrainingExample]:
        return []

    def sample_convos_stage_2(
        self,
        batch_idx: int,
        num_convos: int,
        total_batches: int,
        INFO_CATEGORIES,
        **kwargs,
    ) -> list[TrainingExample]:
        routing_tag = str(uuid.uuid4())

        # (1) sample a subcontext  provide to the question generators
        if batch_idx >=len(self.context.sections): return []
        section = self.context.sections[batch_idx]

        sub_sections = section.content.strip("\n").split("\n")
        chat_tag = sub_sections[0].strip()
        user_query = sub_sections[1].replace("User:", "").strip()

        # (2) Tags come from the mrcr_categories.json
        with open(mrcr_path, "r") as f:
            INFO_CATEGORIES = json.load(f)
        chat_tag = f"{chat_tag}, {user_query}"
        pos_in_category = None
        for category, titles in INFO_CATEGORIES.items():
            if chat_tag in titles:
                category_title = category
                pos_in_category = titles[chat_tag]
                break
        try:
            assert pos_in_category is not None
        except:
            return []

       
        # (3) question-generation prompts; sample questins
        assistant_response = "\n".join(sub_sections[2:]).replace("Assistant:", "").strip()

        PROMPT = """Please generate a {style} question using this passage. Make sure the answer is in the passage.

<passage>
{assistant_response}
</passage>

Output the question in the following format:

Question:"""

        styles = [
            "creative", "simple", "factual"
        ]
        question_chats = []
        for style in styles:
            question_chat = [
                {
                    "role": "system",
                    "content": ""
                },
                {
                    "role": "user", 
                    "content": PROMPT.format(
                        assistant_response=assistant_response,
                        style=style,
                    ),
                },
            ]
            question_chats.append(question_chat)
        

        question_responses: list[CartridgesConvoWithLogprobs] = (
            self.client.chat_with_logprobs(
                chats=question_chats,
                temperature=self.config.temperature,
                max_completion_tokens=self.config.question_max_completion_tokens,
                routing_tag=routing_tag,
            )
        )

        unprocessed_questions = [response.assistant_text for response in question_responses]
        questions = []
        for q in unprocessed_questions:
            q = q.replace("<start>", "").replace("</end>", "")
            q = q.replace("<start>", "").replace("</end>", "")
            if q.startswith("<"):
                q = q[1:]
            q = q.strip()
            if q.endswith(">"):
                q = q[:-1]
            q = q.split("Question:")[-1].strip()
            questions.append(q)

        questions = set(questions)

    
        # (4) Sample answers
        ANSWER_SYSTEM_PROMPT = """Use the following passage to answer the question.

<passage> 
{pos_in_category} passage in category {category_title}, prompt {user_query}.
{text}
</passage>

First state the first sentence in passage contents and then provide your answer. Do not start your answer with "According to the passage" or any similar phrase. 
"""
        
        additional_instructions = random.choices(ADDITIONAL_INSTRUCTIONS, k=3)


        ANSWER_PROMPT = """Question about the response to the {pos_in_category} user query '{user_query}': {question}

{additional_instruction}"""


        questions = list(questions) + INSTRUCTION_TEMPLATES

        answer_chats = [
            [
                {
                    "role": "system",
                    "content": ANSWER_SYSTEM_PROMPT.format(
                        pos_in_category = pos_in_category,
                        category_title = category_title,
                        text=assistant_response, 
                        user_query=user_query
                    ),
                },
                {
                    "role": "user",
                    "content": ANSWER_PROMPT.format(
                        pos_in_category = pos_in_category,
                        category_title = category_title,
                        question=question,
                        additional_instruction=additional_instruction,
                        user_query=user_query
                    ),
                },
            ]
            for question in questions
            for additional_instruction in additional_instructions
        ]
        print(f"Generated {len(answer_chats)} answer chats")
        convos: list[CartridgesConvoWithLogprobs] = self.client.chat_with_logprobs(
            chats=answer_chats,
            temperature=self.config.temperature,
            top_logprobs=self.config.num_top_logprobs,
            max_completion_tokens=self.config.max_completion_tokens,
            routing_tag=routing_tag,
        )

        # (5) Convert to training examples
        examples = responses_and_chats_to_training_examples_orig(convos, answer_chats)
        print(f"Generated {len(examples)} examples")
        return examples
    


# Slice: Memorization slice.
class DirectGenerator(ContextConvoGenerator):
    class Config(ContextConvoGenerator.Config):

        client: ClientConfig
        temperature: float = 0.0
        max_completion_tokens: int = 512
        prompt_template: str = ""
        system_prompt_template: str = ""
        num_top_logprobs: int = 20

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.config = config
        self.client = config.client.instantiate()

    def sample_convos(self, batch_idx, num_convos, total_batches):
        return []

    def stage_1_postprocess(self, convos):
        return convos

    def sample_convos_stage_1(
        self,
        batch_idx: int,
        num_convos: int,
        total_batches: int,
        **kwargs,
    ) -> list[TrainingExample]:
        return []

    def sample_convos_stage_2(
        self,
        batch_idx: int,
        num_convos: int,
        total_batches: int,
        INFO_CATEGORIES,
        **kwargs,
    ) -> list[TrainingExample]:
        routing_tag = str(uuid.uuid4())

        # (1) get batch info
        if batch_idx >=len(self.context.sections): return []
        section = self.context.sections[batch_idx]
        sub_sections = section.content.strip("\n").split("\n")
        chat_tag = sub_sections[0].strip()
        user_query = sub_sections[1].replace("User:", "").strip()
        chat_tag = f"{chat_tag}, {user_query}"
        assistant_response = "\n".join(sub_sections[2:]).replace("Assistant:", "").strip()

        # (2) use the category map to assign structure
        with open(mrcr_path, "r") as f: 
            INFO_CATEGORIES = json.load(f)
        pos_in_category = None
        for category, titles in INFO_CATEGORIES.items():
            if chat_tag in titles:
                category_title = category
                pos_in_category = titles[chat_tag]
                break
        try:
            assert pos_in_category is not None, f"Chat tag {chat_tag} not found in categories"
        except:
            print("ahhh")
            return []

        # (3) sample the answer chats
        prompt = f"Output the contents of the {pos_in_category} response to the request '{user_query}' in the user conversation."

        system_prompt = f"""Here is the content of the {pos_in_category} {user_query}:

{assistant_response}
"""

        answer_chats = [
            [
                {
                    "role": "system",
                    "content": f"{system_prompt}",
                },
                {
                    "role": "user",
                    "content": f"{prompt}",
                },
                {
                    "role": "assistant",
                    "content": f"{assistant_response}",
                },
                {
                    "role": "user",
                    "content": f"",
                },
            ]
        ]
        convos: list[CartridgesConvoWithLogprobs] = self.client.chat_with_logprobs(
            chats=answer_chats,
            temperature=self.config.temperature,
            top_logprobs=self.config.num_top_logprobs,
            max_completion_tokens=self.config.max_completion_tokens,
            routing_tag=routing_tag,
        )

        print(f"Generated {len(convos)} convos")
        examples = responses_and_chats_to_training_examples(convos, answer_chats)

        return examples





