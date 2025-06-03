from __future__ import annotations

import random
import uuid
import numpy as np
from transformers import AutoTokenizer

from capsules.tasks.mrcr import get_toc_str, get_mapping
from capsules.clients.base import ClientConfig
from capsules.clients.tokasaurus_batch import CapsulesConvoWithLogprobs
from capsules.generate.generators.base import ContextConvoGenerator
from capsules.generate.generators.base import responses_and_chats_to_training_examples as responses_and_chats_to_training_examples_orig
from capsules.generate.structs import (
    Context,
    TrainingExample,
)
from capsules.utils import get_logger

logger = get_logger(__name__)


def responses_and_chats_to_training_examples(
    convos: list[CapsulesConvoWithLogprobs],
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

        assert len(token_ids) > convo_response.num_output_tokens
        assert top_logprob_logprob.shape == top_logprob_ids.shape
        assert top_logprob_logprob.shape[0] == len(token_ids) - 1, "You probably need to pull down on tokasaurus or your first message is not a system message"

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


class CheatGenerator(ContextConvoGenerator):
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

        if batch_idx >=len(self.context.sections): return []
        section = self.context.sections[batch_idx]

        print(f"Adding {num_convos} convos to batch {batch_idx}")

        sub_sections = section.content.strip("\n").split("\n")
        turn_number = sub_sections[0].strip()
        user_query = sub_sections[1].replace("User:", "").strip()
        assistant_response = "\n".join(sub_sections[2:]).replace("Assistant:", "").strip()

        user_query = user_query.replace("write a", "")
        try:
            user_query, suffix = user_query.split("(")
        except:
            suffix = turn_number
            pass
        toks = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"]
        str_idx = [tok for tok in toks if tok in suffix][0]
        user_query = user_query.strip()

        rand_str = "0xpseimvih"
        length = len(rand_str)

        answer_chats = []
        for i in range(num_convos):
            rand_str = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz1234567890', k=length))
            cur_user_query = f"Prepend {rand_str} to the {str_idx} (1 indexed) {user_query}. Do not include any other text in your response."

            toc_str = ""

            answer_chat = [
                {
                    "role": "system",
                    "content": f"{toc_str}",
                },
                {
                    "role": "user",
                    "content": f"{cur_user_query}",
                },
                {
                    "role": "assistant",
                    "content": f"{rand_str} {assistant_response}",
                },
                {
                    "role": "user",
                    "content": f"",
                },
            ]
            answer_chats.append(answer_chat)

        # print(f"Generated {len(answer_chats)} answer chats")
        convos: list[CapsulesConvoWithLogprobs] = self.client.chat_with_logprobs(
            chats=answer_chats,
            temperature=self.config.temperature,
            top_logprobs=self.config.num_top_logprobs,
            max_completion_tokens=self.config.max_completion_tokens,
            routing_tag=routing_tag,
        )

        # print(f"Generated {len(convos)} convos")
        examples = responses_and_chats_to_training_examples(convos, answer_chats)
        print(f"Generated {len(examples)} examples")
        return examples


class PrecedingChatGenerator(ContextConvoGenerator):
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

        sample_idx = random.randint(1, len(self.context.sections) - 1)
        section = self.context.sections[sample_idx]

        prior_section = self.context.sections[sample_idx - 1]
        
        PROMPT = f"""Which chat precedes the following chat in the conversation history? 

Interaction:
{section.content}

Output the preceding interaction in the history.
"""
        
        RESPONSE = f"""
{prior_section.content}
"""
        
        toc_str = get_toc_str(self.context.sections)

        # Generate response
        answer_chats = [
            [
                {
                    "role": "system",
                    "content": f"{toc_str}",
                },
                {
                    "role": "user",
                    "content": f"{PROMPT}",
                },
                {
                    "role": "assistant",
                    "content": f"{RESPONSE}",
                },
                {
                    "role": "user",
                    "content": f"",
                },
            ]
        ]
        # print(f"Generated {len(answer_chats)} answer chats")
        convos: list[CapsulesConvoWithLogprobs] = self.client.chat_with_logprobs(
            chats=answer_chats,
            temperature=self.config.temperature,
            top_logprobs=self.config.num_top_logprobs,
            max_completion_tokens=self.config.max_completion_tokens,
            routing_tag=routing_tag,
        )
        # print(f"Generated {len(convos)} convos")
        examples = responses_and_chats_to_training_examples(convos, answer_chats)
        
        return examples


class NextChatGenerator(ContextConvoGenerator):
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

        sample_idx = random.randint(1, len(self.context.sections) - 2)
        section = self.context.sections[sample_idx]

        next_section = self.context.sections[sample_idx + 1]
        
        PROMPT = f"""Which chat comes after the following chat in the user's conversation history? 

Interaction:
{section.content}

Output the subsequent interaction in the history.
"""
        
        RESPONSE = f"""
{next_section.content}
"""
        
        toc_str = get_toc_str(self.context.sections)

        # Generate response
        answer_chats = [
            [
                {
                    "role": "system",
                    "content": f"{toc_str}",
                },
                {
                    "role": "user",
                    "content": f"{PROMPT}",
                },
                {
                    "role": "assistant",
                    "content": f"{RESPONSE}",
                },
                {
                    "role": "user",
                    "content": f"",
                },
            ]
        ]
        # print(f"Generated {len(answer_chats)} answer chats")
        convos: list[CapsulesConvoWithLogprobs] = self.client.chat_with_logprobs(
            chats=answer_chats,
            temperature=self.config.temperature,
            top_logprobs=self.config.num_top_logprobs,
            max_completion_tokens=self.config.max_completion_tokens,
            routing_tag=routing_tag,
        )
        # print(f"Generated {len(convos)} convos")
        examples = responses_and_chats_to_training_examples(convos, answer_chats)
        
        return examples


class ComparePositionsGenerator(ContextConvoGenerator):
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

        sample_idx = random.randint(0, len(self.context.sections) - 1)
        section = self.context.sections[sample_idx]

        sample_idx_compare = random.randint(0, len(self.context.sections) - 1)
        section_compare = self.context.sections[sample_idx_compare]

        binary_random = random.randint(0, 1)
        if binary_random == 0:
            order = "first"
            correct_idx = min(sample_idx, sample_idx_compare)
            correct_section = section if sample_idx == correct_idx else section_compare
        else: 
            order = "second"
            correct = max(sample_idx, sample_idx_compare)
            correct_section = section if sample_idx == correct else section_compare
        
        PROMPT = f"""
Here are two interactions from a conversation history between a user and an assistant. Output the passage that appears {order} in the conversation history.

Interaction 1:
{section.content}


Interaction 2:
{section_compare.content}

Output the passage that appears {order} in the conversation history.
""" 
        
        RESPONSE = f"""
{correct_section.content}
"""

        toc_str = get_toc_str(self.context.sections)

        # Generate response
        answer_chats = [
            [
                {
                    "role": "system",
                    "content": f"{toc_str}",
                },
                {
                    "role": "user",
                    "content": f"{PROMPT}",
                },
                {
                    "role": "assistant",
                    "content": f"{RESPONSE}",
                },
                {
                    "role": "user",
                    "content": f"",
                },
            ]
        ]
        # print(f"Generated {len(answer_chats)} answer chats")
        convos: list[CapsulesConvoWithLogprobs] = self.client.chat_with_logprobs(
            chats=answer_chats,
            temperature=self.config.temperature,
            top_logprobs=self.config.num_top_logprobs,
            max_completion_tokens=self.config.max_completion_tokens,
            routing_tag=routing_tag,
        )
        # print(f"Generated {len(convos)} convos")
        examples = responses_and_chats_to_training_examples(convos, answer_chats)
        
        return examples


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

        if batch_idx >=len(self.context.sections): return []
        section = self.context.sections[batch_idx]

        sub_sections = section.content.strip("\n").split("\n")
        turn_number = sub_sections[0].strip()
        user_query = sub_sections[1].replace("User:", "").strip()
        assistant_response = "\n".join(sub_sections[2:]).replace("Assistant:", "").strip()

        toc_str = get_toc_str(self.context.sections)

        prompt = f"Output the contents of the {turn_number}, {user_query} assistant response."

        answer_chats = [
            [
                {
                    "role": "system",
                    "content": f"",
                },
                {
                    "role": "user",
                    "content": f"{toc_str}\n{prompt}",
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
        # print(f"Generated {len(answer_chats)} answer chats")
        convos: list[CapsulesConvoWithLogprobs] = self.client.chat_with_logprobs(
            chats=answer_chats,
            temperature=self.config.temperature,
            top_logprobs=self.config.num_top_logprobs,
            max_completion_tokens=self.config.max_completion_tokens,
            routing_tag=routing_tag,
        )
        # print(f"Generated {len(convos)} convos")
        examples = responses_and_chats_to_training_examples(convos, answer_chats)

        return examples



class SimpleQAHistoryGenerator(ContextConvoGenerator):
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
        turn_number = sub_sections[0].strip().lower()
        user_query = sub_sections[1].replace("User:", "").strip()
        assistant_response = "\n".join(sub_sections[2:]).replace("Assistant:", "").strip()

        toc_str = get_toc_str(self.context.sections)
        assert toc_str is not "", "TOC string is empty"
        

        PROMPT_WITH_HISTORY = """{toc_str}

Please generate a factual question using this passage. Make sure the answer is in the passage.

<passage>
{assistant_response}
</passage>

Output the question in the following format:

Question:
"""

        # (2) Sample questions
        question_chats = []
        question_chat = [
            {
                "role": "system",
                "content": ""
            },
            {
                "role": "user", 
                "content": PROMPT_WITH_HISTORY.format(
                    toc_str=toc_str,
                    assistant_response=assistant_response,
                ),
            },
        ]
        question_chats.append(question_chat)
        question_responses: list[CapsulesConvoWithLogprobs] = (
            self.client.chat_with_logprobs(
                chats=question_chats,
                temperature=0.9,
                max_completion_tokens=self.config.question_max_completion_tokens,
                routing_tag=routing_tag,
            )
        )

        unprocessed_questions = [response.assistant_text for response in question_responses]
        questions = []
        for q in unprocessed_questions:
            # Remove the <start> and <end> tags from the question
            q = q.replace("<start>", "").replace("</end>", "")
            q = q.replace("<start>", "").replace("</end>", "")
            if q.startswith("<"):
                q = q[1:]
            q = q.strip()
            if q.endswith(">"):
                q = q[:-1]
            q = q.split("Question:")[-1].strip()
            questions.append(q)



        # (3) Sample answers
        ANSWER_PROMPT_WTIH_HISTORY = """{toc_str}

Your task is to use the {turn_number} in the user conversation history to answer the question. 

Question: 
{question}

Passage: 
{text}

Now answer the question: {question}. Do not start your answer with "According to the passage" or any similar phrase. Just answer.
"""

        answer_chats = [
            [
                {
                    "role": "system",
                    "content": "",
                },
                {
                    "role": "user",
                    "content": ANSWER_PROMPT_WTIH_HISTORY.format(
                        toc_str=toc_str,
                        text=assistant_response, 
                        question=question,
                        turn_number=turn_number,
                    ),
                },
            ]
            for question in questions
        ]
        print(f"Generated {len(answer_chats)} answer chats")
        convos: list[CapsulesConvoWithLogprobs] = self.client.chat_with_logprobs(
            chats=answer_chats,
            temperature=self.config.temperature,
            top_logprobs=self.config.num_top_logprobs,
            max_completion_tokens=self.config.max_completion_tokens,
            routing_tag=routing_tag,
        )

        # (4) Convert to training examples
        examples = responses_and_chats_to_training_examples_orig(convos, answer_chats)
        print(f"Generated {len(examples)} examples")
        return examples


class HistoryQAGenerator(ContextConvoGenerator):
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
        turn_number = sub_sections[0].strip().lower()
        user_query = sub_sections[1].replace("User:", "").strip()
        assistant_response = "\n".join(sub_sections[2:]).replace("Assistant:", "").strip()

        toc_str = get_toc_str(self.context.sections)
        assert toc_str is not "", "TOC string is empty"
        
        PROMPT_WITH_HISTORY = """{toc_str}

Please generate a factual question based on the conversation history. Output the question in the following format:

Question:
"""
        # (2) Sample questions
        question_chats = []
        question_chat = [
            {
                "role": "system",
                "content": ""
            },
            {
                "role": "user", 
                "content": PROMPT_WITH_HISTORY.format(
                    toc_str=toc_str,
                ),
            },
        ]
        question_chats.append(question_chat)
        question_responses: list[CapsulesConvoWithLogprobs] = (
            self.client.chat_with_logprobs(
                chats=question_chats,
                temperature=0.9,
                max_completion_tokens=self.config.question_max_completion_tokens,
                routing_tag=routing_tag,
            )
        )

        unprocessed_questions = [response.assistant_text for response in question_responses]
        questions = []
        for q in unprocessed_questions:
            # Remove the <start> and <end> tags from the question
            q = q.replace("<start>", "").replace("</end>", "")
            q = q.replace("<start>", "").replace("</end>", "")
            if q.startswith("<"):
                q = q[1:]
            q = q.strip()
            if q.endswith(">"):
                q = q[:-1]
            q = q.split("Question:")[-1].strip()
            questions.append(q)



        # (3) Sample answers
        ANSWER_PROMPT_WTIH_HISTORY = """{toc_str}

Your task is to use the {turn_number} in the user conversation history to answer the question. 

Question: {question}

Now answer the question: {question}. Do not start your answer with "According to the passage" or any similar phrase. Just answer.
"""

        answer_chats = [
            [
                {
                    "role": "system",
                    "content": "",
                },
                {
                    "role": "user",
                    "content": ANSWER_PROMPT_WTIH_HISTORY.format(
                        toc_str=toc_str,
                        question=question,
                        turn_number=turn_number,
                    ),
                },
            ]
            for question in questions
        ]
        print(f"Generated {len(answer_chats)} answer chats")
        convos: list[CapsulesConvoWithLogprobs] = self.client.chat_with_logprobs(
            chats=answer_chats,
            temperature=self.config.temperature,
            top_logprobs=self.config.num_top_logprobs,
            max_completion_tokens=self.config.max_completion_tokens,
            routing_tag=routing_tag,
        )

        # (4) Convert to training examples
        examples = responses_and_chats_to_training_examples_orig(convos, answer_chats)
        print(f"Generated {len(examples)} examples")
        return examples


class RecallPassageGenerator(ContextConvoGenerator):
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
        user_query = sub_sections[1].replace("User:", "").strip()

        toc_str = get_toc_str(self.context.sections)
        assert toc_str is not "", "TOC string is empty"
        
        PROMPT_WITH_HISTORY = """{toc_str}

Please output the {pos} a assistant response from this user conversation log and output "None" if it does not exist.
"""

        prompt2pos2response = get_mapping(self.context.sections)

        answer_chats = []
        for i, pos in enumerate(["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"]):
            assistant_response = prompt2pos2response[user_query][i]

            chat = [
                {
                    "role": "system",
                    "content": "",
                },
                {
                    "role": "user",
                    "content": PROMPT_WITH_HISTORY.format(
                        toc_str=toc_str,
                        pos=pos
                    ),
                },
                {
                    "role": "assistant",
                    "content": f"{assistant_response}",
                },
                {
                    "role": "user",
                    "content": ""
                },
            ]
            answer_chats.append(chat)
            
        print(f"Generated {len(answer_chats)} answer chats")
        convos: list[CapsulesConvoWithLogprobs] = self.client.chat_with_logprobs(
            chats=answer_chats,
            temperature=self.config.temperature,
            top_logprobs=self.config.num_top_logprobs,
            max_completion_tokens=self.config.max_completion_tokens,
            routing_tag=routing_tag,
        )

        # (4) Convert to training examples
        examples = responses_and_chats_to_training_examples(convos, answer_chats)
        print(f"Generated {len(examples)} examples")
        return examples


class ModifyGenerator(ContextConvoGenerator):
    class Config(ContextConvoGenerator.Config):

        client: ClientConfig
        temperature: float = 0.0
        max_completion_tokens: int = 512
        prompt_template: str = ""
        system_prompt_template: str = ""
        num_top_logprobs: int = 20
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct"

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.config = config
        self.client = config.client.instantiate()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

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
        turn_number = sub_sections[0].strip()
        user_query = sub_sections[1].replace("User:", "").strip()
        assistant_response = "\n".join(sub_sections[2:]).replace("Assistant:", "").strip()

        PROMPT_MODIFY = """
{action} the word '{rand_word}' {action_relative} the following passage and output the modified passage.

{text}

Output the modified passage.
"""

        # (2) Sample questions
        question_prompts = []
        answer_responses = []
        
        for i in range(3):
            binary_random = random.randint(0, 1)
            if binary_random == 0:
                action = "Add"
                action_relative = "to"
                tokenizer_vocab = self.tokenizer.get_vocab()
                rand_word = random.choice(list(tokenizer_vocab.keys())).lower()
                pos = random.randint(0, 1)
                if pos == 0:
                    answer = f"{rand_word} {assistant_response}"
                else:
                    answer = f"{assistant_response} {rand_word} "
            else:
                action = "Delete"
                action_relative = "from"
                rand_word = random.choice(section.content.split()).strip(".").strip(",").strip("'").strip(";").strip('"').strip("?")
                rand_word_lower = rand_word.lower()
                answer = assistant_response.replace(rand_word, "").replace(rand_word_lower, "").strip()

            question_prompts.append(
                PROMPT_MODIFY.format(
                    action=action, 
                    action_relative=action_relative, 
                    rand_word=rand_word, 
                    text=assistant_response
                )
            )
            answer_responses.append(answer)

        # (3) Sample answers
        answer_chats = []
        for prompt, answer in zip(question_prompts, answer_responses):
            answer_chat = [
                {
                    "role": "system",
                    "content": f"",
                },
                {
                    "role": "user",
                    "content": f"{prompt}",
                },
                {
                    "role": "assistant",
                    "content": f"{answer}",
                },
                {
                    "role": "user",
                    "content": f"",
                },
            ]
            answer_chats.append(answer_chat)

        convos: list[CapsulesConvoWithLogprobs] = self.client.chat_with_logprobs(
            chats=answer_chats,
            temperature=self.config.temperature,
            top_logprobs=self.config.num_top_logprobs,
            max_completion_tokens=self.config.max_completion_tokens,
            routing_tag=routing_tag,
        )

        # (4) Convert to training examples
        examples = responses_and_chats_to_training_examples(convos, answer_chats)
        print(f"Generated {len(examples)} examples")
        return examples



