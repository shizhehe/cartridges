from functools import partial
from typing import Any, List, Optional
import random

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

from cartridges.models.helpers import ModelHelper
from cartridges.models.registry import MODEL_REGISTRY
from cartridges.structs import Conversation
from cartridges.datasets import DatasetElement
from cartridges.utils import get_logger

from cartridges.clients.base import FlatTopLogprobs

logger = get_logger(__name__)

class QwenHelper(ModelHelper):

    def messages_to_element(
        self,
        messages: List[Conversation.Message],
        retokenize: bool = False,
        tokenizer: PreTrainedTokenizerFast | None = None,
        prob_drop_thinking: float = 1.0,
        metadata: dict[str, Any] = {},
    ) -> DatasetElement:
        from cartridges.datasets import _base_convert_messages_to_element_retokenize, _base_convert_messages_to_element
        fn = _base_convert_messages_to_element_retokenize if retokenize else _base_convert_messages_to_element
        #fn = _base_convert_messages_to_element
        #logger.info(f"Forcing _base_convert_messages_to_element")
        drop_thinking_fn = partial(_qwen_drop_thinking_fn, prob_drop_thinking=prob_drop_thinking)

        return fn(
            messages,
            tokenizer=tokenizer,
            message_start_tokens={
                "user": [151644, 872,198],
                "assistant": [151644, 77091,198],
                "system": [151644, 8948, 198],
            },
            message_end_tokens={
                "user": [151645],
                "assistant": [151645],
                "system": [151645],
            },
            message_extra_end_tokens={
                "user": [198],
                "assistant": [198],
                "system": [198],
            },
            drop_thinking_fn=drop_thinking_fn,
            metadata=metadata,
        )

    def get_apply_chat_template_kwargs(
        self,
        enable_thinking: bool,
    ) -> dict[str, Any]:
        return {
            "enable_thinking": enable_thinking,
        }
    
    def add_thinking_prompt(
        self,
        content: str,
    ) -> str:
        return content    

    def tokenize_system_prompt_with_max_tokens(
        self, 
        content: str,
        max_tokens: Optional[int],
    ) -> torch.Tensor:
        END_TOKEN_IDS = [151645, 198]

        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": content}],
            include_special_tokens=True,
        )

        if max_tokens is not None and len(input_ids) > max_tokens:
            input_ids = input_ids[: max_tokens - len(END_TOKEN_IDS)] + END_TOKEN_IDS
        

        return torch.tensor(input_ids)[None, :]
    
    def get_cache_size(
        self,
        num_tokens: int,
    ) -> int:
        raise NotImplementedError("QwenHelper.get_cache_size is not implemented.")


def _qwen_drop_thinking_fn(
    message: Conversation.Message,
    prob_drop_thinking: float = 1.0
) -> Conversation.Message:
        THINKING_START_STR = "<think>"
        THINKING_END_STR = "</think>"
        THINKING_START_TOKEN = 151667
        THINKING_END_TOKEN = 151668

        if random.random() < prob_drop_thinking:
            # (1) Removing the thinking section from the str
            thinking_start = message.content.find(THINKING_START_STR)
            if thinking_start == -1:
                # TODO: if there is no thinking section, then presumably it was called
                # without thinking. We should add the empty thinking section back. 
                return message

            thinking_end = message.content.find(THINKING_END_STR, thinking_start)
            if thinking_end == -1:
                # Note: if the thinking end is not found, then presumably the message
                # ran out of space during thinking. We do not trim in this case. 
                return message
            content = message.content[:thinking_start + len(THINKING_START_STR)] + message.content[thinking_end:]
    
            # (2) Removing the thinking tokens from the token ids
            if THINKING_START_TOKEN not in message.token_ids:
                # We should never reach this point if there is no thinking token 
                # in the token ids. Something is broken if so. 
                logger.warning(f"Unexpected: no thinking token in the token ids. Please investigate.")
                return message
            
            # message.token_ids could be numpy array, so we need to convert it to list
            message.token_ids = list(message.token_ids)
            thinking_start = message.token_ids.index(THINKING_START_TOKEN)

            if THINKING_END_TOKEN not in message.token_ids:
                # We should never reach this point if there is no thinking token 
                # in the token ids. Something is broken if so. 
                logger.warning(f"Unexpected: no thinking token in the token ids. Please investigate.")
                return message
            
            thinking_end = message.token_ids.index(THINKING_END_TOKEN)
            token_ids = message.token_ids[:thinking_start + 1] + message.token_ids[thinking_end:]

            message.token_ids = np.array(token_ids)

            # (3) Remove the thinking tokens from the flattened logprobs
            if message.top_logprobs is not None:
                token_idxs = message.top_logprobs.token_idx.copy()

                # (3.1) First, decrement the token idxs for tokens after the thinking section
                mask = (token_idxs >= thinking_end) 
                thinking_length = thinking_end - thinking_start 
                token_idxs[mask] -= thinking_length
                
                # (3.2) Second, extract the thinking section tokens 
                mask = (token_idxs < thinking_start) | (token_idxs > thinking_end)
                logprobs = message.top_logprobs.logprobs[mask]
                token_id = message.top_logprobs.token_id[mask]
                token_idx = message.top_logprobs.token_idx[mask]
                
                top_logprobs = FlatTopLogprobs(
                    token_idx=token_idx,
                    token_id=token_id,
                    logprobs=logprobs,
                    shape=[thinking_length, logprobs.shape[-1]],
                )

        return Conversation.Message(
            role=message.role,
            content=content,
            token_ids=token_ids,
            top_logprobs=top_logprobs,
        )
