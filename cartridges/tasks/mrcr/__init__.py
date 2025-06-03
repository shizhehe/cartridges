import random
import collections
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel

from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
from pydrantic import ObjectConfig
import pandas as pd

from cartridges.context import StructuredContext
from cartridges.structs import Context, ContextConvo, Message, Section
from cartridges.datasets import CartridgeDataset, CartridgeGenerateDataset, CartridgeGenerateDatasetElement, TEMPLATE
from cartridges.context import BaseContextConfig
from cartridges.utils import get_logger

logger = get_logger(__name__)



additional_instructions = "First think step by step and then start writing the conversation between <start> and <end> tags."
additional_instructions = "First think step by step to remember which conversation the question is asking about. Then provide your response between <start> and <end> tags."


class MRCRConvo(StructuredContext):
    title: str
    content: str

    @property
    def text(self) -> str: 
        str = f"{self.title}\n{self.content}"
        return str


class MRCRConversation(StructuredContext):
    title: str
    convos: List[MRCRConvo]

    @property
    def text(self)  -> str:
        output = [self.title]
        for convo in self.convos:
            output.append(convo.text)
        output = "\n\n".join(output)
        return output


class MRCRQuestion(BaseModel):
    question: str
    answer: str
    qid: int
    needles: int
    total_messages: int
    n_chars: int
    desired_msg_index: int
    prepend: str = None
    doc_id: str = None


########### SETUP DATASET ###########

def get_task_questions(
    prompt_dict, prepend, desired_msg_index, 
    total_messages, n_chars, doc_id, 
):
    """
    Use documents with 8 needles to generate a question for each needle.
    """

    needle_prompt = prompt_dict[desired_msg_index]['content']

    desired_msg_indices = []
    for p in range(1, len(prompt_dict[:-1]), 2):
        if prompt_dict[p]['content'] == needle_prompt:
            desired_msg_indices.append(p)

    desired_asst_responses = []
    for i in desired_msg_indices:
        asst_idx = i + 1
        if prompt_dict[asst_idx]['role'] == 'assistant':
            content = prompt_dict[asst_idx]['content']
            desired_asst_responses.append(content)

    idx2str = {
        0: "1st",
        1: "2nd",
        2: "3rd",
        3: "4th",
        4: "5th",
        5: "6th",
        6: "7th",
        7: "8th",
    }

    length = len(prepend)
    additional_qs = []
    for i, (resp, idx) in enumerate(zip(desired_asst_responses, desired_msg_indices)):
        
        if i == desired_msg_index: 
            random_string = prepend
        else:
            random_string = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz1234567890', k=length))

        str_idx = idx2str[i]
        needle_prompt_print = needle_prompt.replace("write a", "").strip()

        question = f"Prepend {random_string} to the {str_idx} {needle_prompt_print}."
        answer = f"{random_string} {resp}"
                
        question = MRCRQuestion(
            question=question,
            prepend=random_string,
            answer=answer,
            qid=i+1,
            needles=8,
            total_messages=total_messages,
            n_chars=n_chars,
            desired_msg_index=idx,
            doc_id=doc_id,
        )
        additional_qs.append(question)

    return additional_qs


def load_my_dataset(document_id: int = -1, is_baseline: bool = False):

    print(f"{is_baseline}")

    from datasets import load_dataset
    ds = load_dataset("openai/mrcr")['train']
        
    questions = []

    print(f"Loading dataset {document_id}...")
    
    element = ds[document_id]    # choosing the last element.
    doc_id = str(len(ds))
    desired_msg_index = element['desired_msg_index']
    total_messages = element['total_messages']
    n_chars = element['n_chars']
    n_needles = element['n_needles']
    random_string_to_prepend = element['random_string_to_prepend']
    answer = element['answer']
    prompt = element['prompt']
    prompt_dict = eval(prompt)
    
    # document processing
    def strip_to_ascii(text: str) -> str:
        return text.encode("ascii", "ignore").decode()

    random.seed(0)
    conversation_chain = []
    chat_num = 0

    for i, p in enumerate(range(1, len(prompt_dict[:-1]), 2)):
        user_query = prompt_dict[p]['content']
        assistant = prompt_dict[p+1]['content']
        chat_num += 1
        
        turn = f"""User: {user_query} 
Assistant: {assistant}"""
        turn = strip_to_ascii(turn)

        if is_baseline:
            title = ""
        else:
            title = f"The #{chat_num} chat in the conversation is:"

        print(title)

        convo = MRCRConvo(
            title=title,
            content=turn,
        )

        conversation_chain.append(convo)

    # eval question construction
    questions = get_task_questions(
        prompt_dict, random_string_to_prepend, desired_msg_index, 
        total_messages, n_chars, doc_id
    )

    document = MRCRConversation(
        title = "This is a conversation between a user and an assistant.",
        convos = conversation_chain
    )

    print(f"Document info: ")
    print(f"  - doc_id: {doc_id}")
    print(f"  - desired_msg_index: {desired_msg_index}")
    print(f"  - total_messages: {total_messages}")
    print(f"  - n_chars: {n_chars}")
    print(f"  - n_needles: {n_needles}")
    print(f"  - prompt: {prompt_dict[-1]}")
    return document, questions



class MRCRSectionedContextConfig(BaseContextConfig):

    document_id : int = -1
    is_baseline: bool = False

    def instantiate(self) -> Context:
        print(f"Creating document context for MRCR...")
        document, questions = load_my_dataset(self.document_id, is_baseline=self.is_baseline)
        return document


class MRCRGenerateDataset(CartridgeGenerateDataset):
    
    class Config(ObjectConfig):
        _pass_as_config = True
        document_id : int = -1
        max_questions: Optional[int] = None
        include_diagnosis: bool = True
        use_cot: bool = True

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):
        self.config = config
        
        document, questions = load_my_dataset(config.document_id)
        
        new_questions = []
        for question in questions:
            if self.config.use_cot:
                new_question = f"{question.question}{additional_instructions}"
            else:
                new_question = question.question

            new_question = MRCRQuestion(
                question=new_question,
                answer=question.answer,
                qid=question.qid,
                needles=question.needles,
                total_messages=question.total_messages,
                n_chars=question.n_chars,
                prepend=question.prepend,
                doc_id=question.doc_id,
                desired_msg_index=question.desired_msg_index,
            )
            new_questions.append(new_question)
        
        self.questions = new_questions
        random.Random(42).shuffle(self.questions)
        self.question_id_to_idx = {
            question.qid: idx for idx, question in enumerate(self.questions)
        }
        self.tokenizer = tokenizer

    def __getitem__(
        self, index: int
    ) -> CartridgeGenerateDatasetElement:
        
        question: MRCRQuestion = self.questions[index]

        input_ids = self.tokenizer.apply_chat_template( # COULD PUT TOC HERE!
            [
                {
                    "role": "system", 
                    "content": "", 
                },
                {
                    "role": "user", 
                    "content": question.question
                }
            ],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=TEMPLATE,
        )

        return CartridgeGenerateDatasetElement(
            input_ids=input_ids,
            prompt=question.question,
            answer=question.answer,
            convo_id=question.qid,
            metadata={"idx": index}
        )

    def __len__(self):
        return len(self.questions)

    def score(
        self,
        pred: str,
        answer: str,
        convo_id: str
    ) -> Tuple[bool, Dict[str, Optional[str]]]:        
        # Extract the answer between <answer> and </answer> tags
        question: MRCRQuestion = self.questions[self.question_id_to_idx[convo_id]]

        def get_text_f1_score(predicted: str, ground_truth: str) -> float:
            """Compute token-level F1 between two strings."""
            # Simple whitespace tokenize. You could swap in a smarter tokenizer if you like.
            pred_tokens = predicted.split()
            truth_tokens = ground_truth.split()
            common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                return 0.0
            precision = num_same / len(pred_tokens)
            recall = num_same / len(truth_tokens)
            return 2 * precision * recall / (precision + recall)

        import re
        pred_match = pred.split("<start>")
        if len(pred_match) > 1:
            pred_match = pred_match[1]
            print(f"Pred: {pred_match}")
            extracted_pred = pred_match.strip().lower().replace('\n', ' ')
            answer = answer.strip().lower().replace('\n', ' ')
            score = get_text_f1_score(extracted_pred, answer)
            details = { 
                "extracted_pred": extracted_pred, 
                "doc_id": question.doc_id,
            }

            printed_pred = extracted_pred.split(". ")[0]
            printed_answer = answer.split(". ")[0]
            q_str = question.question.split('\n')[0]
            print(f"Question:\n{q_str}\n-- Pred: {printed_pred}\n-- Answer: {printed_answer}\n-- Score: {score}\n")
        else:
            details = {
                "extracted_pred": None, 
                "doc_id": question.doc_id,
            }
            score = 0.0

        return score, details


from cartridges.transforms import ConvoTransformConfig
class MRCREvalDataset(CartridgeDataset):
    
    class Config(CartridgeDataset.Config):
        _pass_as_config = True
        document_id : int = -1
        use_cot: bool = True

        # ignored
        label_type: str = "tokens"
        data_sources: List[str] = []
        convo_transforms: list[ConvoTransformConfig] | None = None

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):
        self.config = config
        
        document, questions = load_my_dataset(config.document_id)
         
        clean_questions = []
        for question in questions:

            if self.config.use_cot:
                new_question = f"{question.question}{additional_instructions}"
            else:
                new_question = question.question

            cur_question = MRCRQuestion(
                question=new_question, 
                answer=question.answer,
                qid=question.qid,
                needles=question.needles,
                total_messages=question.total_messages,
                n_chars=question.n_chars,
                prepend=question.prepend,
                doc_id=question.doc_id,
                desired_msg_index=question.desired_msg_index,
            )
            clean_questions.append(cur_question)
            
        self.questions = clean_questions
        random.Random(42).shuffle(self.questions)

        self.data = [
            ContextConvo(
                messages=[
                    Message(
                        role="user",
                        content=question.question,
                    ),
                    Message(
                        role="assistant",
                        content=f"<start>{question.answer}</end>",
                    )
                ],
                type="MRCREval",
                metadata={
                    "qid": question.qid,
                }
            )
            for question in self.questions
        ]

        self.tokenizer = tokenizer


