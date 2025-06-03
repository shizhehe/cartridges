from __future__ import annotations
import abc
import hashlib
import os
import re
import logging
import time
from typing import List, Literal, Optional, Tuple

import numpy as np
from transformers import AutoTokenizer
from pydrantic import ObjectConfig

from capsules.context import StructuredContext, list_nested_contexts
from capsules.generate.structs import Context, Document
from capsules.generate.chunk import Chunker
from capsules.utils import get_logger

logger = get_logger(__name__)

class Retriever(abc.ABC): 
    class Config(ObjectConfig):
        _pass_as_config = True
        max_tokens_per_chunk: int = 1000

        chunking_strategy: Literal["simple", "structured"] = "simple"
    
        
    def __init__(self, config: Config, context: StructuredContext | Context, tokenizer: AutoTokenizer):
        self.config = config
        self.context = context
        self.tokenizer = tokenizer
        self.chunks = self._chunk_simple(context)

    
    def _chunk_simple(self, context: StructuredContext | Context) -> List[str]:
        if hasattr(context, "documents"):
            self.documents: List[Document | StructuredContext]  = context.documents
        else:
            self.documents: List[Document | StructuredContext]  = [
                context
                for _, context in list_nested_contexts(context)
            ]

        # Preprocess and tokenize each chunk
        chunks = []
        for document in self.documents:
            text = document.to_string()
            tokens = self.tokenizer.encode(
                text, 
                add_special_tokens=False,

                # useful for suppressing the truncation error 
                max_length=999_999_999, 
                truncation=True
            )
            chunks.extend([
                self.tokenizer.decode(tokens[i:i+self.config.max_tokens_per_chunk])
                for i in range(0, len(tokens), self.config.max_tokens_per_chunk)
            ])
        return chunks
    
    def _chunk_structured(self, context: StructuredContext | Context) -> List[str]:
        pass


    
    
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
        scores = self._score(query)
        return self._construct_output(scores, top_k, as_string=as_string)
    
    def _construct_output(
        self, 
        scores: np.ndarray, 
        top_k: int, 
        as_string: bool = False,
        existing_context: Optional[str] = None,
    ) -> List[Tuple[int, str]]:
        
        if existing_context is not None:
            # filter out chunks that are already entirely in the existing context
            idx = [idx for idx, chunk in enumerate(self.chunks) if chunk not in existing_context]
            scores = scores[idx]
            chunks = [self.chunks[idx] for idx in idx]
        else:
            chunks = self.chunks
        
        # Get indices of top_k scores
        top_indices = np.argsort(scores)[-top_k:][::-1]

        chunks = [(idx, chunks[idx]) for idx in top_indices]
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

    def batch_retrieve(
        self, 
        queries: List[str], 
        top_k: List[int], 
        as_string: bool = False,
        current_subctxs: Optional[List[str]] = None,
    ) -> List[List[Tuple[int, str]]]:
        scores = self._batch_score(queries)
        if current_subctxs is None:
            current_subctxs = [None] * len(queries)

        return [
            self._construct_output(score, k, as_string=as_string, existing_context=ctx) 
            for score, k, ctx in zip(scores, top_k, current_subctxs, strict=True)
        ]
    
    @abc.abstractmethod
    def _score(self, query: str) -> np.ndarray:
        raise NotImplementedError()

    @abc.abstractmethod
    def _batch_score(self, queries: List[str]) -> List[np.ndarray]:
        raise NotImplementedError()



class BM25Retriever(Retriever):

    class Config(Retriever.Config):
        k1: float = 1.5
        b: float = 0.75
        epsilon: float = 0.25
        

    def __init__(self, config: Config, context: Context, tokenizer: AutoTokenizer):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank_bm25 is not installed. Please install it with `pip install rank-bm25`.")
        self.config = config
        super().__init__(config, context, tokenizer)

        self.tokenized_chunks = [self._preprocess_text(chunk) for chunk in self.chunks]

        t0 = time.time()
        self.bm25 = BM25Okapi(
            self.tokenized_chunks,
            k1=self.config.k1,
            b=self.config.b,
            epsilon=self.config.epsilon,
        )
        logger.info(f"BM25 indexing took {time.time() - t0} seconds")
    
    def _score(self, query: str) -> np.ndarray:
        # Preprocess the query
        tokenized_query = self._preprocess_text(query)
        
        # Get BM25 scores for all chunks
        scores = self.bm25.get_scores(tokenized_query)
        return scores
    
    def _batch_score(self, queries: List[str]) -> List[np.ndarray]:
        tokenized_queries = [self._preprocess_text(query) for query in queries]
        scores = [self.bm25.get_scores(tokenized_query) for tokenized_query in tokenized_queries]
        return scores
    
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
    


class OpenAIRetriever(Retriever):

    class Config(Retriever.Config):
        embedding_model: str = "text-embedding-3-small"
        max_tokens_per_chunk: int = 1000

        cache_dir: Optional[str] = None
        

    def __init__(self, config: Config, context: Context, tokenizer: AutoTokenizer):
        self.config = config
        super().__init__(config, context, tokenizer)
        
        self._setup_openai_client()
        self.chunk_embeddings = self._get_embeddings(self.chunks, use_cache=True)        
        

    def _setup_openai_client(self):
        """Initialize OpenAI client"""
        import os
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def _get_embeddings(self, texts: List[str], use_cache: bool= False) -> List[List[float]]:
        """Get embeddings for a list of texts using OpenAI's API"""
        embeddings = []

        if self.config.cache_dir is not None and use_cache:
            combined_str = "".join(texts)
            hash_str = hashlib.sha256(combined_str.encode()).hexdigest()
            cache_file = os.path.join(self.config.cache_dir, f"{hash_str}.npy")
            if os.path.exists(cache_file):
                data = np.load(cache_file)
                logger.info(f"Loading embeddings from cache file {cache_file}...")
                return data.tolist()
            else:
                logger.info("Cache miss. Computing embeddings from scratch...")

        # OpenAI has a limit of 2048 examples per request
        MAX_TOKENS_PER_REQUEST = 300_000
        MAX_EXAMPLES_PER_REQUEST = 2048
        batch_size = min(
            MAX_EXAMPLES_PER_REQUEST, 
            (MAX_TOKENS_PER_REQUEST // self.config.max_tokens_per_chunk) // 2 
        )

        t0 = time.time()
        for batch_start in range(0, len(texts), batch_size):
            response = self.client.embeddings.create(
                input=texts[batch_start:batch_start+batch_size],
                model=self.config.embedding_model
            )
            embeddings.extend([embedding.embedding for embedding in response.data])
        logger.info(f"Indexing with OpenAI took {time.time() - t0} seconds")

        if self.config.cache_dir is not None and use_cache:
            logger.info(f"Caching embeddings to {cache_file}...")
            os.makedirs(self.config.cache_dir, exist_ok=True)
            np.save(cache_file, np.array(embeddings))

        return embeddings

    def _score(self, query: str) -> np.ndarray:
        """
        Retrieve the most relevant chunks for the query using cosine similarity
        between embeddings
        """
        try:
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            raise ImportError("sklearn is not installed. Please install it with `pip install scikit-learn`.")

        # Get embeddings
        query_embedding = self._get_embeddings([query], use_cache=False)

        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)

        return similarities[0]

    def _batch_score(self, queries: List[str]) -> List[np.ndarray]:
        try:
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            raise ImportError("sklearn is not installed. Please install it with `pip install scikit-learn`.")

        query_embeddings = self._get_embeddings(queries, use_cache=False)
        similarities = cosine_similarity(query_embeddings, self.chunk_embeddings)
        return similarities
