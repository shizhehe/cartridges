from __future__ import annotations
import random
import re
import time
from typing import List, Tuple, Union
import uuid

import numpy as np
from transformers import AutoTokenizer
from pydrantic import ObjectConfig

from capsules.clients.base import ClientConfig, ClientResponse
from capsules.generate.context_convo_generators.base import ContextConvoGenerator
from capsules.generate.structs import Context, ContextConvo, Message, Document
from capsules.generate.utils import convert_sample_into_message
from capsules.utils import get_logger

logger = get_logger(__name__)
class RAGContextConvoGenerator(ContextConvoGenerator):
    class Config(ContextConvoGenerator.Config):
        assistant_client: ClientConfig
        user_client: ClientConfig
        retriever: Retriever.Config
        retrieve_top_k: int = 1

        # prompt templates used for the user (i.e. question generator)
        # the prompt templates can include a placeholder for {context}, which will be 
        # replaced with randomly sampled chunks from the full context
        user_system_prompt_template: str = "You are a helpful assistant that generates questions from a given context."
        user_prompt_template: str = "{context}"
        user_max_completion_tokens: int = 512

        # prompt templates used for the assistant (i.e. answer generator)
        # the prompt templates can include {context} and {instruction} placeholders, which will be 
        # replaced with the context and the instruction, respectively
        assistant_system_prompt_template: str = "You are a helpful assistant that answers questions from a given context."
        assistant_prompt_template: str = "{context} -- {instruction}"
        assistant_max_completion_tokens: int = 512
        assistant_top_logprobs: int = 20

        num_tokens_in_user_context: int = 1024

        tokenizer: str = "meta-llama/Llama-3.2-3B-Instruct"

    def __init__(self, config: Config, context: Context):
        self.config = config
        self.context = context
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)

        self.tokenized_documents = [
            self.tokenizer.encode(doc.to_string(), add_special_tokens=False)
            for doc in self.context.documents
        ]

        self.assistant_client = config.assistant_client.instantiate()
        self.user_client = config.user_client.instantiate()
        self.retriever = config.retriever.instantiate(
            context=self.context,
            tokenizer=self.tokenizer,
        )


    def sample_convos(self, start_idx: int, end_idx: int) -> list[ContextConvo]:
        num_samples = end_idx - start_idx

        # (1) sample a single context to provide to all the question generators
        # --- begin context sampling ---
        t0 = time.time()
        start_doc_idx = random.randint(0, len(self.context.documents) - 1)
        
        user_context = ""
        token_count = 0
        for doc_idx in range(start_doc_idx, len(self.context.documents)):
            tokenized_document = self.tokenized_documents[doc_idx]

            if token_count + len(tokenized_document) > self.config.num_tokens_in_user_context:
                user_context += self.tokenizer.decode(tokenized_document[:self.config.num_tokens_in_user_context - token_count], skip_special_tokens=True)
                break
            else:
                user_context += self.tokenizer.decode(tokenized_document, skip_special_tokens=True)
                token_count += len(tokenized_document)
        t1 = time.time()
        logger.info(f"[batch {start_idx}:{end_idx}] Sampled context in {t1 - t0:.2f} seconds")
        # --- end of context sampling ---

        # (2) generate instructions
        # --- begin question generation ---
        t0 = time.time()
        question_response: ClientResponse = self.user_client.chat(
            chats=[
                [
                    { 
                        "role": "system",
                        "content": self.config.user_system_prompt_template.format(context=user_context)
                    },
                    {
                        "role": "user",
                        "content": self.config.user_prompt_template.format(context=user_context)
                    }
                ]
            ]  * num_samples,
            max_completion_tokens=self.config.user_max_completion_tokens,
        )
        questions = [sample.text for sample in question_response.samples]
        t1 = time.time()
        logger.info(f"[batch {start_idx}:{end_idx}] Generated {num_samples} questions in {t1 - t0:.2f} seconds")
        # --- end of question generation ---

        # (3) use rag to get relevant context for each question
        # --- begin RAG retrieval ---
        t0 = time.time()
        assistant_contexts = []
        for question in questions:
            retrieved_contexts = self.retriever.retrieve(question, top_k=self.config.retrieve_top_k, as_string=True)
            assistant_context = f"<core-context>\n{user_context}\n</core-context>\n\n<retrieved-context>\n{retrieved_contexts}\n</retrieved-context>"
            assistant_contexts.append(assistant_context)
        t1 = time.time()
        logger.info(f"[batch {start_idx}:{end_idx}] Retrieved {num_samples} assistant contexts in {t1 - t0:.2f} seconds")
        # --- end of RAG retrieval ---

        # (4) generate response
        # --- begin answer generation ---
        t0 = time.time()
        answer_response: ClientResponse = self.assistant_client.chat(
            chats=[
                [
                    {
                        "role": "system",
                        "content": self.config.assistant_system_prompt_template.format(context=assistant_context, instruction=question)
                    },
                    {
                        "role": "user",
                        "content": self.config.assistant_prompt_template.format(context=assistant_context, instruction=question)
                    }
                ]
                for assistant_context, question in zip(assistant_contexts, questions)
            ],
            max_completion_tokens=self.config.assistant_max_completion_tokens,
            top_logprobs=self.config.assistant_top_logprobs,
        )
        t1 = time.time()
        logger.info(f"[batch {start_idx}:{end_idx}] Generated {num_samples} answers in {t1 - t0:.2f} seconds")
        # --- end of answer generation ---
        
        # (5) construct convo
        # --- begin convo construction ---
        logger.info(f"[batch {start_idx}:{end_idx}] Done generating {num_samples} convos")
        convos = []
        for answer_sample, assistant_context, question in zip(answer_response.samples, assistant_contexts, questions):
            answer_message = convert_sample_into_message(answer_sample)
            convos.append(
                ContextConvo(
                    id=str(uuid.uuid4()),
                    messages=[
                        Message(role="user", content=question),
                        answer_message,
                    ],
                    type="rag",
                    metadata={
                        "assistant_context": assistant_context,
                    },
                )
            )
        # --- end of convo construction ---
        return convos



class Retriever:
    class Config(ObjectConfig):
        _pass_as_config = True


class BM25Retriever(Retriever):

    class Config(Retriever.Config):
        k1: float = 1.5
        b: float = 0.75
        epsilon: float = 0.25

        max_tokens_per_chunk: int = 1000
        

    def __init__(self, config: Config, context: Context, tokenizer: AutoTokenizer):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank_bm25 is not installed. Please install it with `pip install rank-bm25`.")

        self.config = config
        self.context = context
        self.tokenizer = tokenizer
        self.documents: List[Document] = context.documents

        # Preprocess and tokenize each chunk
        chunks = []
        for document in self.documents:
            text = document.to_string()
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            chunks.extend([
                self.tokenizer.decode(tokens[i:i+self.config.max_tokens_per_chunk])
                for i in range(0, len(tokens), self.config.max_tokens_per_chunk)
            ])
        self.chunks = chunks
        self.tokenized_chunks = [self._preprocess_text(chunk) for chunk in chunks]
        
        # Create BM25 model
        self.bm25 = BM25Okapi(
            self.tokenized_chunks,
            k1=self.config.k1,
            b=self.config.b,
            epsilon=self.config.epsilon,
        )

        # import nltk
        # if not nltk.data.find('tokenizers/punkt'):
        #     nltk.download('punkt')
        # if not nltk.data.find('corpora/stopwords'):
        #     nltk.download('stopwords')


    def retrieve(self, query: str, top_k: int = 1, as_string: bool = False) -> List[Tuple[int, str]]:
        """
        Returns the most relevant chunk(s) for the given query using BM25 scoring.
        
        Args:
            query (str): The search query
            top_k (int): Number of top chunks to return
            as_string (bool): Whether to construct a string from the chunks
                in the order of the original chunks and with separators between chunks
        Returns:
            List[Tuple[int, str]]: Top k most relevant chunks with their original indices
             which is useful for reconstructing the original order of the chunks
        """
        if not self.bm25:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Preprocess the query
        tokenized_query = self._preprocess_text(query)
        
        # Get BM25 scores for all chunks
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get indices of top_k scores
        top_indices = np.argsort(scores)[-top_k:][::-1]

        chunks = [(idx, self.chunks[idx]) for idx in top_indices]
        if not as_string:
            return chunks
        
        # reconstruct  
        sorted_chunks = sorted(chunks, key=lambda x: x[0])
        reconstructed_str = ""
        SEPARATOR = "\n\n...<omitted chunks>...\n\n"
        prev_idx = None
        for idx, chunk in sorted_chunks:
            if prev_idx is not None and idx != prev_idx + 1:
                reconstructed_str += SEPARATOR
            reconstructed_str += chunk
            prev_idx = idx
        return reconstructed_str


    

        
        


        return chunks

        
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocesses text by converting to lowercase, removing special characters,
        and tokenizing into words.
        """
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords

        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        return tokens