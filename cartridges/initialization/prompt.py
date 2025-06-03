from typing import Optional
import torch
from cartridges.structs import Context
from cartridges.kv_initialization.base import AttnConfig, KVCacheFactory, TrainableCache
from transformers import DynamicCache

from cartridges.kv_initialization.tokenization_utils import (
    tokenize_data_into_system_prompt,
)

EOS_TOKEN_ID = 128009


class PromptInitializer(KVCacheFactory):
    class Config(KVCacheFactory.Config):
        prompt: str
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.config = config


    def initalize_kv_cache(
        self,
        context: Context,
        tokenizer,
        model,
        attn_config: AttnConfig,
    ) -> TrainableCache:
        input_ids = tokenize_data_into_system_prompt(
            tokenizer,
            self.config.prompt,
            max_tokens=None
        )
        
        init_cache = DynamicCache()
        with torch.no_grad():
            model(
                input_ids.to(model.device),
                use_cache=True,
                past_key_values=init_cache,
            )

            num_tokens = input_ids.shape[-1]
            return TrainableCache(
                config=attn_config,
                num_tokens=num_tokens,
                keys=list(init_cache.key_cache),
                values=list(init_cache.value_cache),
                num_frozen_tokens=self.config.num_frozen_tokens,
            )
