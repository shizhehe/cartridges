from typing import Optional
import torch
from capsules.generate.run import BaseContextConfig
from capsules.generate.structs import Context
from capsules.kv_initialization.base import AttnConfig, KVCacheFactory, TrainableCache
from transformers import DynamicCache

EOS_TOKEN_ID = 128009



BOS_TOKEN_ID= 128000
EOS_TOKEN_ID = 128009

START_HEADER_ID = 128006
END_HEADER_ID = 128007
SYSTEM_ID = 9125


def tokenize_data_into_system_prompt(
    tokenizer,
    contents: list[str],
    max_tokens: Optional[int],
) -> torch.Tensor:
    
    breakpoint()
    
    input_ids_separate = []
    for content in contents:
        input_ids = tokenizer.encode(content.text)
        input_ids_separate.append(input_ids)

    if (sum([len(input_ids) for input_ids in input_ids_separate]) + 5) > max_tokens:
        half_length = ( max_tokens - 5)  // 2

        print(f"Half length: {half_length}")

        first_n_ids_per_context = torch.concat(
            [torch.tensor(input_ids)[:half_length] for input_ids in input_ids_separate]
        )

        print(f"First n ids per context: {len(first_n_ids_per_context)}")

        content = tokenizer.decode(first_n_ids_per_context)

    else:

        content = "\n".join([content.text for content in contents])

    input_ids = tokenizer.apply_chat_template([
        {"role": "system", "content": content}]
    )
    assert input_ids[-1] == EOS_TOKEN_ID

    if max_tokens is not None and len(input_ids) > max_tokens:
        input_ids = input_ids[: max_tokens - 1] + [EOS_TOKEN_ID]

    assert input_ids[:4] == [BOS_TOKEN_ID, START_HEADER_ID, SYSTEM_ID, END_HEADER_ID]

    return torch.tensor(input_ids)[None, :]


class KVCacheInitFromFirstNTokensOfEachContext(KVCacheFactory):
    class Config(KVCacheFactory.Config):
        max_tokens: Optional[int]

        # if not provided, we will use the context passed to initalize_kv_cache
        # if provided, we will ignore the context passed to initalize_kv_cache
        contexts: Optional[BaseContextConfig] = None

    
    def __init__(self, config: Config):
        super().__init__(config)

    def initalize_kv_cache(
        self,
        contexts: Context,
        tokenizer,
        model,
        attn_config: AttnConfig,
    ) -> TrainableCache:
    

        input_ids = tokenize_data_into_system_prompt(
            tokenizer=tokenizer,
            contents=contexts,
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
