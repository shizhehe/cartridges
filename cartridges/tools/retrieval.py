from __future__ import annotations
import abc
import re
from typing import List, Optional, Tuple

import numpy as np
from pydantic import BaseModel
from transformers import AutoTokenizer
from pydrantic import ObjectConfig

from capsules.context import StructuredContext, list_nested_contexts
from capsules.generate.structs import Context, Document
from capsules.tools.base import Tool
from capsules.retrievers import OpenAIRetriever, BM25Retriever, Retriever
from capsules.utils import get_logger

logger = get_logger(__name__)

class RetrieverTool(Tool): 
    class Config(Tool.Config):
        _pass_as_config = True
        max_tokens_per_chunk: int = 1000
        max_top_k: int = 10

    retriever: Retriever
    
    class ToolInput(BaseModel):
        query: str
        top_k: int = 1

        
    
    def __init__(self, config: Config, context: StructuredContext, tokenizer: AutoTokenizer):
        self.config = config
    
    def __call__(
        self,
        input: ToolInput,
    ) -> List[Tuple[int, str]]:
        top_k = min(input.top_k, self.config.max_top_k)
        return self.retriever.retrieve(
            query=input.query, 
            top_k=top_k,
            as_string=True,
        )

    def batch_call(
        self,
        inputs: List[ToolInput],
        current_subctxs: Optional[List[str]] = None,
    ) -> List[Tuple[int, str]]:
        try:
            queries = [input.query for input in inputs]
            top_ks = [min(input.top_k, self.config.max_top_k) for input in inputs]
            outputs = self.retriever.batch_retrieve(
                queries=queries,
                top_k=top_ks,
                as_string=True,
                current_subctxs=current_subctxs,
            )
            return [
                {
                    "success": True,
                    "input": input,
                    "tool_response": output,
                    "error": None,
                }
                for input, output in zip(inputs, outputs, strict=True)
            ]
        except Exception as e:
            logger.info(f"Error retrieving chunks: {e}")
            return [
                {
                    "success": False,
                    "input": input,
                    "tool_response": None,
                    "error": str(e),
                }
                for input in inputs
            ]
        
    def description(self) -> str:
        return f"Retrieve relevant chunks of text based on similarity with a text query you provide. The search is done using embedding similarity over chunks of maximum size {self.config.max_tokens_per_chunk} tokens."
    
    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}_chunksize{self.config.max_tokens_per_chunk}"


class BM25RetrieverTool(RetrieverTool):

    class Config(RetrieverTool.Config):
        k1: float = 1.5
        b: float = 0.75
        epsilon: float = 0.25

    def __init__(self, config: Config, context: StructuredContext, tokenizer: AutoTokenizer):
        self.config = config

        self.retriever = BM25Retriever.Config(
            k1=config.k1,
            b=config.b,
            epsilon=config.epsilon,
            max_tokens_per_chunk=config.max_tokens_per_chunk,
        ).instantiate(
            context=context,
            tokenizer=tokenizer,
        )
    

    

class OpenAIRetrieverTool(RetrieverTool):

    class Config(RetrieverTool.Config):
        embedding_model: str = "text-embedding-3-small"
        max_tokens_per_chunk: int = 1000

        cache_dir: Optional[str] = None
    
    def __init__(self, config: Config, context: StructuredContext, tokenizer: AutoTokenizer):
        self.config = config

        self.retriever = OpenAIRetriever.Config(
            embedding_model=config.embedding_model,
            max_tokens_per_chunk=config.max_tokens_per_chunk,
            cache_dir=config.cache_dir,
        ).instantiate(
            context=context,
            tokenizer=tokenizer,
        )