from abc import ABC, abstractmethod
from typing import Any, List, Optional

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from cartridges.structs import Conversation
from cartridges.datasets import DatasetElement



class ModelHelper(ABC):
    """Base class providing model-specific utilities for different language models.
    
    This class serves as an interface for model-specific operations including:
    - Tokenization and chat template handling
    - Message-to-element conversion for dataset processing
    - Reasoning/thinking prompt processing and manipulation
    - Model-specific token handling and caching utilities
    
    Each model family (e.g., Qwen, Llama) should implement their own subclass
    with model-specific token IDs, chat templates, and processing logic.
    
    Registry Usage:
    Model helpers are registered in the MODEL_REGISTRY (defined in registry.py) which
    maps model names to their corresponding helper classes. This allows the codebase
    to dynamically select the appropriate helper for any supported model via
    MODEL_REGISTRY.get_model_helper(model_name). Registration happens in each model's
    __init__.py file (e.g., models/qwen/__init__.py, models/llama/__init__.py).
    """

    def __init__(
        self, 
        model_name: str,
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_chat_template(self) -> str:
        return None
    
    def get_apply_chat_template_kwargs(
        self,
        enable_thinking: bool,
    ) -> dict[str, Any]:
        return {}

    @abstractmethod
    def messages_to_element(
        messages: List[Conversation.Message],
        retokenize: bool = False,
        tokenizer: PreTrainedTokenizerFast | None = None,
        prob_drop_thinking: float = 1.0,
        metadata: dict[str, Any] = {},
    ) -> DatasetElement:
        raise NotImplementedError("ModelHelper.messages_to_element is not implemented.")
    
    def add_thinking_prompt(
        self,
        content: str,
    ) -> str:
        return content

    def get_cache_size(
        self,
        num_tokens: int,
    ) -> int:
        raise NotImplementedError("QwenHelper.get_cache_size is not implemented.")

    
