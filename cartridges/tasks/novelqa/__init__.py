"Benchmark: https://huggingface.co/datasets/NovelQA/NovelQA/tree/main"
"Download the zip from [https://huggingface.co/datasets/NovelQA/NovelQA/tree/main]"

import os
import random
from typing import Any, Dict, List, Optional, Tuple
import json
from pydantic import BaseModel
import requests

from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
from pydrantic import ObjectConfig
import pandas as pd

from capsules.generate.structs import Context, ContextConvo, Document, Message, Section
from capsules.datasets import CapsuleDataset, CapsuleDatasetElementTokenLabels, CapsuleGenerateDataset, CapsuleGenerateDatasetElement, TEMPLATE
from capsules.generate.run import BaseContextConfig
from capsules.utils import get_logger

logger = get_logger(__name__)


class NovelQuestion(BaseModel):
    question_id: str
    question: str
    orig_question: str
    aspect: str
    complexity: str
    book_id: str
    book_title: str

    answer_a: str
    answer_b: str
    answer_c: str
    answer_d: str

    answer: str = ""


class Novel(BaseModel):
    book_id: str
    context: str
    questions: List[NovelQuestion]

    title: str 
    # source: str
    # link: str
    # whichgtb: str
    # copyright: str
    # yearpub: int
    # author: str
    # yearperish: int
    # period: str
    # tokenlen: int


def load_novelqa_dataset(book_id: str) -> List[Novel]:

    if book_id == "Frankenstein_Demo":
        fquestions = "/home/simarora/code/capsules/scratch/simran/NovelQA/Demonstration/"
        fbooks = "/home/simarora/code/capsules/scratch/simran/NovelQA/Demonstration/"
        questions = ['Frankenstein.json']
        books = ['Frankenstein.txt']
        book_id = "Frankenstein"
    else:
        fquestions = "/home/simarora/code/capsules/scratch/simran/GoldNovelQA/Data/PublicDomain/"
        fbooks = "/home/simarora/code/capsules/scratch/simran/GoldNovelQA/Books/PublicDomain/"
        questions = sorted(os.listdir(fquestions))
        books = sorted(os.listdir(fbooks))
        fmeta = "/home/simarora/code/capsules/scratch/simran/NovelQA/bookmeta.json"
        metadata = json.load(open(fmeta)) # dict_keys(['title', 'source', 'link', 'whichgtb', 'copyright', 'yearpub', 'author', 'yearperish', 'period', 'tokenlen'])

    # load the dataset
    dataset = {}
    for q, b in zip(questions, books):
        qpath = f"{fquestions}{q}"
        qs = json.load(open(qpath))
        bpath = f"{fbooks}{b}"
        bs = open(bpath).read()
        bid = b.split(".")[0]

        if type(qs) == list:
            qs = {q['QID']: q for i, q in enumerate(qs)}

        dataset[bid] = {
            "texts": bs,
            "questions": qs
        }


    entry = dataset[book_id]
    try:
        meta = metadata[book_id]
    except:
        meta = {"title": book_id,}
        
    questions = []
    for q_id, q_entry in entry["questions"].items():
        answer = q_entry["Gold"]
        questions.append(
            NovelQuestion(
                question_id=q_id,
                question=q_entry["Question"],
                orig_question=q_entry["Question"],
                aspect=q_entry["Aspect"],
                complexity=q_entry["Complexity"],
                answer_a=q_entry["Options"]['A'],
                answer_b=q_entry["Options"]['B'],
                answer_c=q_entry["Options"]['C'],
                answer_d=q_entry["Options"]['D'],
                book_id=book_id,
                book_title=meta["title"],
                answer=answer,
            )
        )
        
    novel = Novel(
        book_id=book_id,
        context=entry["texts"],
        questions=questions,
        title=meta["title"],
        # source=meta["source"],
        # link=meta["link"],
        # whichgtb=meta["whichgtb"],
        # copyright=meta["copyright"],
        # yearpub=meta["yearpub"],
        # author=meta["author"],
        # yearperish=meta["yearperish"],
        # period=meta["period"],
        # tokenlen=meta["tokenlen"],
    )

    return novel


class NovelContextConfig(BaseContextConfig):

    book_id : str = 'Frankenstein_Demo'
    tokenizer: str = "meta-llama/Llama-3.2-3B-Instruct"
    max_tokens_per_section: int = -1

    def instantiate(self) -> Context:

        print(f"Creating book context with {self.max_tokens_per_section} tokens per section")
        book = load_novelqa_dataset(self.book_id)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)

        sections = []
        max_tokens = 131072 
        if self.max_tokens_per_section < 0: # icl 
            tokenized = len(tokenizer.encode(book.context))
            if tokenized > max_tokens:
                logger.warning(f"Book context is too long: {tokenized} tokens, truncating to {max_tokens}")
                context = book.context[:max_tokens]
            else:
                context = book.context
            section = Section(
                desc=f"Book: {book.title}",
                content=context,
            )
            sections.append(section)
        else:
            tokenized = len(tokenizer.encode(book.context))
            assert self.max_tokens_per_section < max_tokens

            # partition the book into sections of max_tokens_per_section
            num_sections = tokenized // self.max_tokens_per_section + 1
            section_size = tokenized // num_sections
            for i in range(num_sections):
                start = i * section_size
                end = (i + 1) * section_size
                if end > tokenized:
                    end = tokenized
                section = Section(
                    desc=f"Book: {book.title} (Page {i+1}/{num_sections})",
                    content=book.context[start:end],
                )
                sections.append(section)
            print(f"Created {len(sections)} sections of size {self.max_tokens_per_section} tokens each")

        title = book.title
        context = Context(
            title=title,
            sections=sections,
        )
        return context


class NovelMultipleChoiceGenerateDataset(CapsuleGenerateDataset):
    class Config(ObjectConfig):
        _pass_as_config = True
        book_id: str = None
        max_questions: Optional[int] = None
        include_diagnosis: bool = True
        cot: bool = True

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):
        self.config = config
        
        self.book = load_novelqa_dataset(config.book_id)
        
        def wrap_question(question: NovelQuestion):
            options = (
                f"{question.answer_a}\n"
                f"{question.answer_b}\n"
                f"{question.answer_c}\n"
                f"{question.answer_d}"
            )
            if self.config.cot:
                cot_prompt = "You should first think step by step. Then give your final answer exactly as it appears in the options with the following format:"
            else:
                cot_prompt = "Please provide your answer exactly as it appears in the options with the following format:"
                
            return (
                "Please answer the following question about the book: "
                f"\n\n<question>\n{question.question}\n</question>"
                f"\n\n<options>\n{options}\n</options>\n\n{cot_prompt}"
                f"\n\n<answer>\n{{answer}}\n</answer>"
            )
         
        self.questions = [
            NovelQuestion(
                question_id=question.question_id,
                orig_question=question.question,
                question=wrap_question(question),
                aspect=question.aspect,
                complexity=question.complexity,
                answer_a=question.answer_a,
                answer_b=question.answer_b,
                answer_c=question.answer_c,
                answer_d=question.answer_d,
                book_id=question.book_id,
                book_title=question.book_title,
                answer=question.answer,
            )
            for question in self.book.questions
        ]
        random.Random(42).shuffle(self.questions)

        if self.config.max_questions is not None:
            self.questions = self.questions[:self.config.max_questions]
        self.question_id_to_idx = {
            question.question_id: idx for idx, question in enumerate(self.questions)
        }

        self.tokenizer = tokenizer


    def __getitem__(
        self, index: int
    ) -> CapsuleGenerateDatasetElement:
        
        question: NovelQuestion = self.questions[index]

        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question.question}],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=TEMPLATE,
        )

        return CapsuleGenerateDatasetElement(
            input_ids=input_ids,
            prompt=question.question,
            answer=question.answer,
            convo_id=question.question_id,
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

        from difflib import SequenceMatcher
        def find_best_match(reference, candidates):
            return max(candidates, key=lambda x: SequenceMatcher(None, reference, x).ratio())
        
        # Extract the answer between <answer> and </answer> tags
        import re
        question: NovelQuestion = self.questions[self.question_id_to_idx[convo_id]]

        options = [
            question.answer_a.strip().lower(),
            question.answer_b.strip().lower(),
            question.answer_c.strip().lower(),
            question.answer_d.strip().lower(),
        ]

        idx2letter = { 
            0: "A",
            1: "B",
            2: "C",
            3: "D"
        }
        pred_match = re.search(r'<answer>(.*?)</answer>', pred, re.DOTALL)
        if pred_match:
            extracted_pred = pred_match.group(1).strip().lower()

            closest_match = find_best_match(extracted_pred, options)
            closest_letter_idx = options.index(closest_match)
            closest_letter = idx2letter[closest_letter_idx]

            score = closest_letter.strip().lower() == answer.strip().lower()
            details = { 
                "extracted_pred": extracted_pred, 
                "closest_match": closest_match,
                "closest_letter": closest_letter,
                "book_id": question.book_id,
                "book_title": question.book_title,
            }
        else:
            # If no tags found, random guess
            closest_letter = "A"
            score = closest_letter.strip().lower() == answer.strip().lower()
            details = {
                "extracted_pred": None, 
                "closest_match": None,
                "closest_letter": None,
                "book_id": question.book_id,
                "book_title": question.book_title,
            }

        print(f"Score: {score}, Closest Letter: {closest_letter}, Answer: {answer}, Question: {question.orig_question}")
        print("----"*10)
        return score, details


class NovelEvalDataset(CapsuleDataset):
    
    class Config(CapsuleDataset.Config):
        _pass_as_config = True
        book_id: str = None
        max_questions: Optional[int] = None

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):
        self.config = config
        
        self.book = load_novelqa_dataset(config.book_id)
        
        def wrap_question(question: NovelQuestion, cur_book: Novel):
            options = (
                f"<option>{question.answer_a}</option>\n"
                f"<option>{question.answer_b}</option>\n"
                f"<option>{question.answer_c}</option>\n"
                f"<option>{question.answer_d}</option>\n"
            )
            return (
                "Please answer the question below about the following book: "
                f"ID {cur_book.book_id}"
                f"\n\n <question>\n{question.question}\n</question>"
                f"\n\n <options>\n{options}\n</options>\n"
                "Answer the question with the following format, outputting only the answer:"
                f"\n\n<answer>\n{{answer}}\n</answer>"
            )
         
        questions = []
        for question in self.book.questions:
            cur_question = NovelQuestion(
                question_id=question.question_id,
                orig_question=question.question,
                question=wrap_question(question, self.book),
                aspect=question.aspect,
                complexity=question.complexity,
                answer_a=question.answer_a,
                answer_b=question.answer_b,
                answer_c=question.answer_c,
                answer_d=question.answer_d,
                book_id=question.book_id,
                book_title=question.book_title,
                answer=question.answer,
            )
            questions.append(cur_question)
        self.questions = questions
        random.Random(42).shuffle(self.questions)

        if self.config.max_questions is not None:
            self.questions = self.questions[:self.config.max_questions]

        self.data = [
            ContextConvo(
                messages=[
                    Message(
                        role="user",
                        content=question.question,
                    ),
                    Message(
                        role="assistant",
                        content=f"<answer>{question.answer}</answer>",
                    )
                ],
                type="NovelEval",
                metadata={
                    "question_id": question.question_id,
                }
            )
            for question in self.questions
        ]

        self.tokenizer = tokenizer


