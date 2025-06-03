import random
import re
from typing import Any, Dict, List, Literal, Optional, Set, Tuple
from dataclasses import asdict, dataclass


from transformers import PreTrainedTokenizerFast
from pydrantic import ObjectConfig
import pandas as pd

from capsules.generate.structs import Context, ContextConvo, Document, Message
from capsules.datasets import CapsuleDataset, CapsuleDatasetElementTokenLabels, CapsuleGenerateDataset, CapsuleGenerateDatasetElement, TEMPLATE
from capsules.generate.run import BaseContextConfig
from capsules.train import GenerateDatasetConfig
from capsules.utils import get_logger

logger = get_logger(__name__)



PROMPT_TEMPLATE = """\
Please answer the question below:
<question>
{question}
</question>

<options>
{options}
</options>

You can think step by step, but then give your final answer exactly as it is in the options with the following format:
<answer>
{answer}
</answer>
"""


prompt = """Consider statutory law for {state} in the year 2021. Read the following statute excerpts which govern housing law in this state, and answer the question below.
{question}
{cot_prompt}"""

@dataclass
class StatuteReference:
    statute_idx: int
    citation: str
    excerpt: str

@dataclass
class ReglabHousingQuestion:
    idx: int
    state: str
    question: str

    # the original question from the LSC database. This is sometimes identical to 
    # question. In other cases question corresponds to a rephrased version of this 
    # designed to have a yes/no answer.
    original_question: str

    answer: Literal["Yes", "No"] 
    
    question_group: int
    statutes: List[StatuteReference]   # the statutes that support the answer
    caveats: List[str]
    

@dataclass
class ReglabHousingCategoriesQuestion:
    idx: int
    state: str

    # indicates the question group. not the unique question id
    question_number: float
    question: str
    answer: str
    
    category: str
    statutes: List[List[str]]   # the statutes that support the answer
    answer_options: str


class ReglabHousingQAGenerateDataset(CapsuleGenerateDataset):
    class Config(ObjectConfig):
        _pass_as_config = True
        cot: bool = True
        states: Optional[List[str]] = None
        max_questions: Optional[int] = None

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):
        self.config = config
        self.tokenizer = tokenizer

        from datasets import load_dataset
        df = load_dataset("reglab/housing_qa", "questions", split="test", download_mode="force_redownload").to_pandas()

        if self.config.states is not None:
            df = df[df["state"].isin(self.config.states)]

        if self.config.max_questions is not None:
            df = df.head(self.config.max_questions)

        self.questions: List[ReglabHousingQuestion] = [
            ReglabHousingQuestion(
                **{k: v for k, v in row.items() if k != "statutes"},
                statutes=[
                    StatuteReference(
                        statute_idx=statute["statute_idx"],
                        citation=statute["citation"],
                        excerpt=statute["excerpt"]
                    ) for statute in row["statutes"]
                ]
            ) for _, row in df.iterrows()
        ]
        self.question_id_to_idx = {question.idx: idx for idx, question in enumerate(self.questions)}
        
        
    def _wrap_question(self, question: ReglabHousingQuestion):
        if self.config.cot:
            cot_prompt = "You should first explain your reasoning. Then give your final yes or no answer with the following format: \n<answer>\n{{yes or no}}\n</answer>"
        else:
            cot_prompt = "Please provide your yes or no answer with the following format: \n\n<answer>\n{{yes or no}}\n</answer>"
        return (
            f"Answer the question below based on the statutory law for {question.state} in the year 2021."
            f"\n\n<question>\n{question.question}\n</question>{cot_prompt}"
        )
        

    def __getitem__(
        self, index: int
    ) -> CapsuleGenerateDatasetElement:
        question = self.questions[index]
        question_prompt = self._wrap_question(question)

        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question_prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=TEMPLATE,
        )

        return CapsuleGenerateDatasetElement(
            input_ids=input_ids,
            prompt=question_prompt,
            answer=question.answer,
            convo_id=question.idx,
            metadata={
                "idx": index, 
                "statutes": [asdict(statute) for statute in question.statutes]
            }
        )

    def __len__(self):
        return len(self.questions)

    def score(
        self,
        pred: str,
        answer: str,
        convo_id: str
    ) -> Tuple[bool, Dict[str, Optional[str]]]:
        pred_match = re.search(r'<answer>(.*?)</answer>', pred, re.DOTALL)
        if pred_match is not None:
            extracted_pred = pred_match.group(1).strip().lower()
        else:
            # just default to the entire answer
            extracted_pred = pred.strip().lower()

        if "yes" in extracted_pred:
            extracted_pred = "yes"
        else:
            extracted_pred = "no"

        return extracted_pred == answer.strip().lower(), {"extracted_pred": extracted_pred, "extraction": "success", "pred": pred}


class ReglabHousingCategoriesQAGenerateDataset(CapsuleGenerateDataset):
    class Config(ObjectConfig):
        _pass_as_config = True
        cot: bool = True
        states: Optional[List[str]] = None
        max_questions: Optional[int] = None

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):
        self.config = config
        self.tokenizer = tokenizer

        import os
        df = pd.read_json(
            os.path.join(
                os.environ.get("CAPSULES_DIR", "./"), 
                "capsules/tasks/reglab/housing_qa_consolidated_generations_with_answer_options.jsonl", 
            ),
            lines=True
        )
        df = df[df['category'] == 'eviction_categories']


        if self.config.states is not None:
            df = df[df["state"].isin(self.config.states)]

        if self.config.max_questions is not None:
            df = df.head(self.config.max_questions)

        self.questions: List[ReglabHousingQuestion] = [
            ReglabHousingCategoriesQuestion(
                **{k: v for k, v in row.items()}, # if k!="statutes"},
                idx=idx,
                # statutes=[
                #     "\n\n".join([statute[0] + ":\n" + statute[1] for statute in row["statutes"]])
                # ]
            ) for idx, row in df.iterrows()
        ]
        self.question_id_to_idx = {question.idx: idx for idx, question in enumerate(self.questions)}
        
        
    def _wrap_question(self, question: ReglabHousingQuestion):
        if self.config.cot:
                cot_prompt = "You should first think step by step. Then give your final answer as a newline separated list of options. Your output should be in the following format: \n<thinking> {{YOUR_THOUGHT_PROCESS}} </thinking> "
        else:
            cot_prompt = "Please provide your final answer as a newline separated list of options in the following format:"
        
        
        return (
                f"Answer the question below based on the statutory law for {question.state} in the year 2021."
                f"\n\n<question>\n{question.question}\nSelect all of the following options that apply:\n</question>"
                f"\n\n<options>\n{question.answer_options}\n</options>\n{cot_prompt}"
                f"\n\n<answer>\n{{YOUR_ANSWER}}\n</answer>"
            )


    def __getitem__(
        self, index: int
    ) -> CapsuleGenerateDatasetElement:
        question = self.questions[index]
        question_prompt = self._wrap_question(question)

        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question_prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=TEMPLATE,
        )

        return CapsuleGenerateDatasetElement(
            input_ids=input_ids,
            prompt=question_prompt,
            answer=question.answer,
            convo_id=question.idx,
            metadata={
                "idx": index, 
                "statutes": [{"citation": statute[0], "excerpt": statute[1]} for statute in question.statutes]
            }
        )

    def __len__(self):
        return len(self.questions)

    def score(
        self,
        pred: str,
        answer: str,
        convo_id: str
    ) -> Tuple[bool, Dict[str, Optional[str]]]:
        pred_match = re.search(r'<answer>(.*?)</answer>', pred, re.DOTALL)
        if pred_match is not None:
            extracted_pred = pred_match.group(1).strip().lower()
        else:
            # just default to the entire answer
            extracted_pred = pred.strip().lower()
        
        # Looking for exact match. Use set because order doesn't matter
        # make both sets lowercase
        correct_set = set([opt.strip().lower() for opt in answer.split("\n")])
        given_set = set([opt.strip().lower() for opt in extracted_pred.split("\n")])

        metrics = {
            "accuracy": correct_set == given_set,
            "f1": _f1(correct_set, given_set),
            "recall": _recall(correct_set, given_set),
            "precision": _precision(correct_set, given_set),
        }
        extras = {
            "extracted_pred": extracted_pred,
            "extraction": "success",
            "pred": pred,
        }

        return metrics, extras


def _recall(correct_set: Set[str], pred_set: Set[str]) -> float:
    if len(correct_set) == 0:
        return 1
    return len(correct_set & pred_set) / len(correct_set)

def _precision(correct_set: Set[str], pred_set: Set[str]) -> float:
    if len(pred_set) == 0:
        return 1
    return len(correct_set & pred_set) / len(pred_set)

def _f1(correct_set: Set[str], pred_set: Set[str]) -> float:
    recall = _recall(correct_set, pred_set)
    precision = _precision(correct_set, pred_set)
    return 2 * (recall * precision) / (recall + precision) if recall + precision > 0 else 0

