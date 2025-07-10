from typing import Optional
import torch
from cartridges.cache import AttnConfig, KVCacheFactory, TrainableCache
from transformers import DynamicCache
from cartridges.context import BaseContextConfig, StructuredContext


from cartridges.initialization.tokenization_utils import (
    tokenize_data_into_system_prompt,
)

EOS_TOKEN_ID = 128009


class KVCacheInitFromFirstNTokensOfContext(KVCacheFactory):
    class Config(KVCacheFactory.Config):
        max_tokens: Optional[int]

        # if not provided, we will use the context passed to initalize_kv_cache
        # if provided, we will ignore the context passed to initalize_kv_cache
        context: Optional[BaseContextConfig] = None

    
    def __init__(self, config: Config):
        super().__init__(config)
        if config.context is not None:
            self.context = config.context.instantiate()
        else:
            self.context = None


    def initalize_kv_cache(
        self,
        tokenizer,
        model,
        attn_config: AttnConfig,
    ) -> TrainableCache:
        if self.context is not None:
            context = self.context

        input_ids = tokenize_data_into_system_prompt(
            tokenizer=tokenizer,
            content=context.to_string(),
            max_tokens=self.config.max_tokens,
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
