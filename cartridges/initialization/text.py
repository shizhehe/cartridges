import os
from pathlib import Path
from typing import Literal, Optional
import torch

from cartridges.cache import AttnConfig, KVCacheFactory, TrainableCache
from cartridges.initialization.tokenization_utils import MODEL_TO_SYSTEM_PROMPT_TOKENIZER

DEFAULT_TEXT_SOURCE = os.path.join(
    os.environ["CARTRIDGES_DIR"], "cartridges/initialization/data/gradient.txt"
)

class KVFromRandomText(KVCacheFactory):
    class Config(KVCacheFactory.Config):
        max_tokens: Optional[int]
        text_source: str = DEFAULT_TEXT_SOURCE

    def initialize_kv_cache(
        self,
        tokenizer,
        model,
        attn_config: AttnConfig,
    ) -> TrainableCache:
        content = Path(self.config.text_source).read_text()

        tokenize_data_into_system_prompt = MODEL_TO_SYSTEM_PROMPT_TOKENIZER[tokenizer.name_or_path.lower()]

        input_ids = tokenize_data_into_system_prompt(
            tokenizer=tokenizer,
            content=content,
            max_tokens=self.config.max_tokens,
        ).squeeze(0)
        
        init_cache = TrainableCache(config=attn_config)
        
        with torch.no_grad():
            input_ids = input_ids.to(model.device)
            seq_ids = torch.full_like(input_ids, 0, dtype=torch.long)
            position_ids = torch.arange(input_ids.shape[-1], dtype=torch.long).to(model.device)
            model(
                input_ids=input_ids,
                seq_ids=seq_ids,
                position_ids=position_ids,
                use_cache=True,
                past_key_values=init_cache,
                mode="generate",
            )

            print(self.config.num_frozen_tokens)

            return TrainableCache(
                config=attn_config,
                init_keys=init_cache._keys,
                init_values=init_cache._values,
                num_frozen_tokens=self.config.num_frozen_tokens,
            )


