"Benchmark: https://huggingface.co/datasets/hazyresearch/evaporate/tree/main/data/Evaporate_movie_imdb"

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

from cartridges.context import StructuredContext
from cartridges.structs import Context, ContextConvo, Document, Message, Section
from cartridges.datasets import CartridgeDataset, CartridgeDatasetElementTokenLabels, CartridgeGenerateDataset, CartridgeGenerateDatasetElement, TEMPLATE
from cartridges.context import BaseContextConfig
from cartridges.utils import get_logger

logger = get_logger(__name__)


class EvaporateQuestion(BaseModel):
    question: str
    attribute: str
    doc_id: str
    question_id: str
    answer: str = ""


class Evaporate(BaseModel):
    doc_id: str
    context: str
    questions: List[EvaporateQuestion]
    title: str 


class StructuredDocumentChunk(StructuredContext):
    title: str
    content: str


class StructuredDocument(StructuredContext):
    chunks: List[StructuredDocumentChunk]

    @property
    def text(self) -> str:
        doc_str = ""
        for chunk in self.chunks:
            doc_str += f"""{chunk.content}"""
        return doc_str


def wrap_question(question: EvaporateQuestion):                
    prompt = f"Given the document, {question.question}\n\nDo not begin your answer with 'In this document' or any similar phrase, just output the answer."""
    return prompt


def load_dataset(
    doc_id: str,
    fpages: str="/home/user/data/evaporate/fda-ai-pmas/510k/",
    fmetadata: str="/home/user/data/evaporate/table.json",
) -> List[Evaporate]:

    htmls = sorted(os.listdir(fpages))
    htmls = [h for h in htmls if h.endswith(".txt")]
    metadata = json.load(open(fmetadata))

    docid = doc_id
    key = f"/data/evaporate/fda-ai-pmas/510k/{docid}"
    metadata = metadata[key]
    
    with open (f"{fpages}/{docid}", "r") as f:
        document_content = f.read()
        
    questions = []
    for i, (k, v) in enumerate(sorted(metadata.items())):
        questions.append(
            EvaporateQuestion(
                question=f"what is the '{k}'?",
                attribute=k,
                doc_id=docid,
                answer=v,
                question_id=f"{docid}_{i}",
            )
        )
        
    document = Evaporate(
        doc_id=docid,
        context=document_content,
        questions=questions,
        title='FDA Premarket Device Report',
    )

    return document


class EvaporateContextConfig(BaseContextConfig):

    doc_id : str = None
    tokenizer: str = "meta-llama/Llama-3.2-3B-Instruct"
    max_tokens_per_section: int = -1

    # pages_path: str = "/data/simran/data/evaporate/fda-ai-pmas/510k/"
    # table_path: str = "/data/simran/data/evaporate/table.json"

    def instantiate(self) -> Context:

        document = load_dataset(
            self.doc_id
            # , self.pages_path, self.table_path
        )
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)

        sections = []
        max_tokens = 131072 
        if self.max_tokens_per_section < 0: # icl 
            tokenized = len(tokenizer.encode(document.context))
            if tokenized > max_tokens:
                context = document.context[:max_tokens]
            else:
                context = document.context
            section = Section(
                desc=f"FDA Premarket Device Report: {document.title}",
                content=context,
            )
            sections.append(section)
            return Context(
                sections=sections,
                title=f"FDA Premarket Device Report: {document.title}",
            )
        
        else:
            tokenized_text = tokenizer.encode(document.context)
            tokenized = len(tokenized_text)
            assert self.max_tokens_per_section < max_tokens

            # partition the document into sections of max_tokens_per_section
            num_sections = tokenized // self.max_tokens_per_section + 1
            section_size = tokenized // num_sections
            for i in range(num_sections):
                start = i * section_size
                end = (i + 1) * section_size
                if end > tokenized:
                    end = tokenized
                section = StructuredDocumentChunk(
                    title=f"FDA Premarket Device Report: {document.title} (Page {i+1}/{num_sections})",
                    content=tokenizer.decode(tokenized_text[start:end]),
                )
                sections.append(section)

            print(f"Created {len(sections)} sections of size {self.max_tokens_per_section} tokens each")
            out_document = StructuredDocument(chunks=sections)
            print(out_document.text[:100])

            return out_document


class EvaporateMultipleChoiceGenerateDataset(CartridgeGenerateDataset):
    class Config(ObjectConfig):
        _pass_as_config = True
        doc_id: str = None
        max_questions: Optional[int] = None
        include_diagnosis: bool = True
        cot: bool = True

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):
        self.config = config
        
        self.document = load_dataset(config.doc_id)
         
        self.questions = [
            EvaporateQuestion(
                question=wrap_question(question),
                attribute=question.attribute,
                doc_id=question.doc_id,
                answer=question.answer,
                question_id=question.question_id,
            )
            for question in self.document.questions
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
    ) -> CartridgeGenerateDatasetElement:
        
        question: EvaporateQuestion = self.questions[index]

        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question.question}],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=TEMPLATE,
        )

        return CartridgeGenerateDatasetElement(
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
        # Extract the answer between <answer> and </answer> tags
        import re
        question: EvaporateQuestion = self.questions[self.question_id_to_idx[convo_id]]

        import collections
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


        def cleaned_response(response):
            response = response.strip().lower().replace('\n', ' ')
            response = response.replace("(", "").replace(")", "").replace(",", "").replace("<|eot_id|>", "")
            response = response.replace("'", "").replace('"', "").replace("$", "").replace("<|eot_id|>", "")
            return response
            

        pred_match = pred # re.search(r'<response>(.*?)</response>', pred, re.DOTALL)
        pred_match = pred # pred_match.group(1)
        if pred_match:
            # pred_match = pred_match.group(1)
            extracted_pred = cleaned_response(pred_match)
            answer = cleaned_response(answer)


            score = get_text_f1_score(extracted_pred, answer)
            details = { 
                "extracted_pred": extracted_pred, 
                "doc_id": question.doc_id,
            }
        else:
            # If no tags found, random guess
            score = 0
            extracted_pred = cleaned_response(pred)[:40]
            details = {
                "extracted_pred": None, 
                "doc_id": question.doc_id,
            }

        q_str = question.question.split('\n')[0]
        print(f"Question:\n{q_str}\n-- Pred: {extracted_pred}\n-- Answer: {answer}\n-- Score: {score}\n")
        return score, details


class EvaporateEvalDataset(CartridgeDataset):
    
    class Config(CartridgeDataset.Config):
        _pass_as_config = True
        doc_id: str = None
        max_questions: Optional[int] = None

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):
        self.config = config
        
        self.document = load_dataset(config.doc_id)
        
        questions = []
        for question in self.document.questions:
            cur_question = EvaporateQuestion(
                question=wrap_question(question),
                attribute=question.attribute,
                doc_id=question.doc_id,
                answer=question.answer,
                question_id=question.question_id,
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
                        content=f"<response>{question.answer}</response>",
                    )
                ],
                type="EvaporateEval",
                metadata={
                    "question_id": question.question_id,
                }
            )
            for question in self.questions
        ]

        self.tokenizer = tokenizer


