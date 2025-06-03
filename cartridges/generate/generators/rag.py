from __future__ import annotations
import random
import time
import uuid

from transformers import AutoTokenizer

from capsules.clients.base import ClientConfig, ClientResponse
from capsules.generate.generators.base import ContextConvoGenerator
from capsules.generate.structs import Context, ContextConvo, Message
from capsules.utils import get_logger
from capsules.retrievers import Retriever, BM25Retriever
from capsules.generate.utils import convert_sample_into_message


# backwards compatability with old code before we moved the retrievers to a separate file
__all__ = ["BM25Retriever"]

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


    def sample_convos(self, batch_idx: int, num_convos: int) -> list[ContextConvo]:
        num_samples = num_convos
        start_idx = batch_idx * num_convos
        end_idx = start_idx + num_convos

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