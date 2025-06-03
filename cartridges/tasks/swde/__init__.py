"Benchmark: https://huggingface.co/datasets/hazyresearch/evaporate/tree/main/data/swde_movie_imdb"

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

from capsules.context import StructuredContext
from capsules.generate.structs import Context, ContextConvo, Document, Message, Section
from capsules.datasets import CapsuleDataset, CapsuleDatasetElementTokenLabels, CapsuleGenerateDataset, CapsuleGenerateDatasetElement, TEMPLATE
from capsules.generate.run import BaseContextConfig
from capsules.utils import get_logger

logger = get_logger(__name__)


class SWDEQuestion(BaseModel):
    question: str
    attribute: str
    webpage_id: str
    question_id: str
    answer: str = ""

class SWDE(BaseModel):
    webpage_id: str
    context: str
    questions: List[SWDEQuestion]
    title: str 

class WebpageChunk(StructuredContext):
    title: str
    content: str

class Webpage(StructuredContext):
    chunks: List[WebpageChunk]

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


def load_swde_dataset(
    webpage_id: str,
    fpages: str="/home/simarora/code/capsules/scratch/simran/SWDE/data/evaporate/swde/movie/movie-imdb(2000)/",
    fmetadata: str="/home/simarora/code/capsules/scratch/simran/SWDE/table.json",
) -> List[SWDE]:

    htmls = sorted(os.listdir(fpages))
    htmls = [h for h in htmls if h.endswith(".htm")]
    metadata = json.load(open(fmetadata))

    docid = webpage_id
    key = f"/data/evaporate/swde/movie/movie-imdb(2000)/{docid}"
    metadata = metadata[key]
    
    with open (f"{fpages}/{docid}", "r") as f:
        webpage_content = f.read()
        
    questions = []
    for i, (k, v) in enumerate(sorted(metadata.items())):
        question = handcrafted_questions[k]
        questions.append(
            SWDEQuestion(
                # question=f"What is/are the '{k}' of this movie?",
                # question=f"What is/are the '{k}' attribute of this movie?",
                # question=f"The value of the '{k}' attribute of this movie is?",
                # question=f"Who/what/where/why/when -- the '{k}' of this movie?", # best (https://wandb.ai/hazy-research/capsules/runs/wpfewtpz) 
                # question=f"Who/what/where/when? The full '{k}' of this movie is?",

                question=question,

                # question=f"Describe the '{k}' for this movie.",
                attribute=k,
                webpage_id=docid,
                answer=v,
                question_id=f"{docid}_{i}",
            )
        )
        
    webpage = SWDE(
        webpage_id=docid,
        context=webpage_content,
        questions=questions,
        title='IMDB Movie',
    )

    return webpage


class SWDEContextConfig(BaseContextConfig):

    webpage_id : str = None
    tokenizer: str = "meta-llama/Llama-3.2-3B-Instruct"
    max_tokens_per_section: int = -1

    # pages_path: str = "/data/sabri/data/evaporate/swde/movie/movie-imdb(2000)"
    pages_path: str = "/home/simarora/code/capsules/scratch/simran/SWDE/data/evaporate/swde/movie/movie-imdb(2000)"
    table_path: str = "/home/simarora/code/capsules/scratch/simran/SWDE/table.json"

    def instantiate(self) -> Context:

        print(f"Creating webpage context with {self.max_tokens_per_section} tokens per section")
        webpage = load_swde_dataset(self.webpage_id, self.pages_path, self.table_path)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)

        sections = []
        max_tokens = 131072 
        if self.max_tokens_per_section < 0: # icl 
            tokenized = len(tokenizer.encode(webpage.context))
            if tokenized > max_tokens:
                logger.warning(f"IMDB Movie context is too long: {tokenized} tokens, truncating to {max_tokens}")
                context = webpage.context[:max_tokens]
            else:
                context = webpage.context
            section = Section(
                desc=f"IMDB Movie: {webpage.title}",
                content=context,
            )
            sections.append(section)

            return Context(
                sections=sections,
                title="IMDB Movie",
            )
        
        else:
            tokenized_text = tokenizer.encode(webpage.context)
            tokenized = len(tokenized_text)
            assert self.max_tokens_per_section < max_tokens

            # partition the webpage into sections of max_tokens_per_section
            num_sections = tokenized // self.max_tokens_per_section + 1
            section_size = tokenized // num_sections
            for i in range(num_sections):
                start = i * section_size
                end = (i + 1) * section_size
                if end > tokenized:
                    end = tokenized
                section = WebpageChunk(
                    title=f"IMDB Movie: {webpage.title} (Page {i+1}/{num_sections})",
                    content=tokenizer.decode(tokenized_text[start:end]),
                )
                sections.append(section)

            print(f"Created {len(sections)} sections of size {self.max_tokens_per_section} tokens each")

            webpage = Webpage(
                chunks=sections,
            )

            return webpage


class SWDEMultipleChoiceGenerateDataset(CapsuleGenerateDataset):
    class Config(ObjectConfig):
        _pass_as_config = True
        webpage_id: str = None
        max_questions: Optional[int] = None
        include_diagnosis: bool = True
        cot: bool = True

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):
        self.config = config
        
        self.webpage = load_swde_dataset(config.webpage_id)
        
        def wrap_question(question: SWDEQuestion):                
            return (
                f"Here's a question that tests a model's knowledge of the information in the passage: {question.question}\n\n"
                f"Think out loud briefly before you answer the question. When you are done thinking, please provide a concise answer, exactly as it appears in the html, using the following tag format:\n\n"
                f"<start> your concise answer </end>"
            )
            # return (
            #     f"Here's a question that tests a model's knowledge of the information in the passage: {question.question}\n\n"
            #     f"Please provide your final answer, exactly as it appears in the html, using the following tag format:\n\n"
            #     f"<start> your final answer </end>"
            # )
         
        self.questions = [
            SWDEQuestion(
                question=wrap_question(question),
                attribute=question.attribute,
                webpage_id=question.webpage_id,
                answer=question.answer,
                question_id=question.question_id,
            )
            for question in self.webpage.questions
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
        
        question: SWDEQuestion = self.questions[index]

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
        # Extract the answer between <answer> and </answer> tags
        import re
        question: SWDEQuestion = self.questions[self.question_id_to_idx[convo_id]]

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


        def cleaned_response(response, attribute):
            response = response.strip().lower().replace('\n', ' ')
            response = response.replace("(", "").replace(")", "").replace(",", "")
            response = response.replace("'", "").replace('"', "").replace("$", "")
            return response
            

        pred_match = re.search(r'<start>(.*?)</end>', pred, re.DOTALL)
        if pred_match:
            attribute = question.attribute
            extracted_pred = cleaned_response(pred_match.group(1), attribute)
            answer = cleaned_response(answer, attribute)


            score = get_text_f1_score(extracted_pred, answer)
            details = { 
                "extracted_pred": extracted_pred, 
                "webpage_id": question.webpage_id,
            }
        else:
            # If no tags found, random guess
            score = 0
            extracted_pred = ''
            details = {
                "extracted_pred": None, 
                "webpage_id": question.webpage_id,
            }

        q_str = question.question.split('\n')[0]
        print(f"Question:\n{q_str}\n-- Pred: {extracted_pred}\n-- Answer: {answer}\n-- Score: {score}\n")
        return score, details


class SWDEEvalDataset(CapsuleDataset):
    
    class Config(CapsuleDataset.Config):
        _pass_as_config = True
        webpage_id: str = None
        max_questions: Optional[int] = None

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):
        self.config = config
        
        self.webpage = load_swde_dataset(config.webpage_id)
        
        def wrap_question(question: SWDEQuestion, cur_webpage: SWDE):
            return (
                f"Here's a question that tests a model's knowledge of the information in the passage: {question.question}\n\n"
                f"Think out loud briefly before you answer the question. When you are done thinking, please provide a concise answer, exactly as it appears in the html, using the following tag format:\n\n"
                f"<start> your concise answer </end>"
            )
            
            # return (
            #     f"Here's a question that tests a model's knowledge of the information in the passage: {question.question}\n\n"
            #     f"Please provide your final answer, exactly as it appears in the html, using the following tag format:\n\n"
            #     f"<start> your final answer </end>"
            # )
         
        questions = []
        for question in self.webpage.questions:
            cur_question = SWDEQuestion(
                question=wrap_question(question, self.webpage),
                attribute=question.attribute,
                webpage_id=question.webpage_id,
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
                        content=f"<start>{question.answer}</end>",
                    )
                ],
                type="SWDEEval",
                metadata={
                    "question_id": question.question_id,
                }
            )
            for question in self.questions
        ]

        self.tokenizer = tokenizer


