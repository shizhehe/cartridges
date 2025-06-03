from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
from cartridges.data.gradient import gradient_wiki_content
from cartridges.structs import Context
from cartridges.cache import AttnConfig, KVCacheFactory, TrainableCache
from cartridges.initialization.tokenization_utils import tokenize_data_into_system_prompt
from transformers import DynamicCache

class KVFromRandomVectors(KVCacheFactory):
    class Config(KVCacheFactory.Config):
        max_tokens: int

    def __init__(self, config: Config):
        self.config = config

    def initalize_kv_cache(
        self,
        context: Context,
        tokenizer,
        model,
        attn_config: AttnConfig,
    ) -> TrainableCache:
        rand_vectors = lambda: [
            torch.randn(
                1, attn_config.n_heads, self.config.max_tokens, attn_config.head_dim,
                dtype=torch.bfloat16,
            )
            for _ in range(attn_config.n_layers)
        ]

        return TrainableCache(
            attn_config,
            self.config.max_tokens,
            keys=rand_vectors(),
            values=rand_vectors(),
            num_frozen_tokens=self.config.num_frozen_tokens,
        )

class KVFromRandomText(KVCacheFactory):
    class Config(KVCacheFactory.Config):
        max_tokens: Optional[int]

    def initalize_kv_cache(
        self,
        context: Context,
        tokenizer,
        model,
        attn_config: AttnConfig,
    ) -> TrainableCache:
        gradient_wiki = gradient_wiki_content()

        input_ids = tokenize_data_into_system_prompt(
            tokenizer=tokenizer,
            content=gradient_wiki,
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
