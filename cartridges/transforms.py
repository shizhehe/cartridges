from abc import ABC
import re
from typing import List, Union
import numpy as np
from transformers import AutoTokenizer
from pydrantic import ObjectConfig


from capsules.clients.base import Client, ClientConfig
from capsules.generate.run import BaseContextConfig
from capsules.generate.structs import ContextConvo, Context, Document, Message
from capsules.retrievers import Retriever


class ConvoTransformConfig(ObjectConfig):
    _pass_as_config: bool = True


class ConvoTransform(ABC):

    def __init__(self, config: ConvoTransformConfig, context: Context):
        raise NotImplementedError()

    def __call__(self, convo: ContextConvo) -> ContextConvo:
        raise NotImplementedError()



class RetrieverTransform(ConvoTransform):

    class Config(ConvoTransformConfig):
        retriever: Retriever.Config
        context: BaseContextConfig
        top_k: int = 1

    def __init__(self, config: Config, tokenizer: AutoTokenizer, **kwargs):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank_bm25 is not installed. Please install it with `pip install rank-bm25`.")

        self.config = config
        self.context = self.config.context.instantiate()
        self.tokenizer = tokenizer

        self.retriever = self.config.retriever.instantiate(context=self.context, tokenizer=self.tokenizer)
    
    def __call__(self, convo: ContextConvo) -> ContextConvo:
        new_messages = [
            Message(
                content=message.content if message.role != "user" else self._augment_user_message(message),
                role=message.role,
                sample=message.sample,
            )
            for message in convo.messages
        ]
        return ContextConvo(
            id=convo.id,
            messages=new_messages,
            type=convo.type,
            metadata=convo.metadata,
        )
    
    def _augment_user_message(self, message: Message) -> Message:
        retrieved_context = self.retriever.retrieve(message.content, top_k=self.config.top_k, as_string=True)
        context_str = "<retrieved_context>" + retrieved_context + "\n</retrieved_context>"
        return context_str + "\n\n" + message.content
        
        
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


