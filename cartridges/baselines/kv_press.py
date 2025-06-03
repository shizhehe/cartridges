import itertools
import math
import torch

import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from kvpress import (
    BasePress,
    KeyRerotationPress,
    ObservedAttentionPress,
    PerLayerCompressionPress,
    DuoAttentionPress,
    ExpectedAttentionPress,
)
from typing import List, Literal, Union, Optional
import time  # Optional: for timing generation


class ModelWithCompressedCache:
    """
    Wraps a Hugging Face CausalLM model to use a compressed KV cache
    for a fixed prefix using the kvpress library.

    Uses a manual generation loop instead of model.generate().

    Attributes:
        press (BasePress): The kvpress compression instance.
        model (AutoModelForCausalLM): The Hugging Face transformer model.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        prefix_tokens (torch.Tensor): The token IDs of the prefix sequence.
        prefix_cache (DynamicCache): The compressed KV cache state after processing
                                      the prefix_tokens.
        device (torch.device): The device the model is on.
    """

    def __init__(
        self,
        press: BasePress,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prefix_tokens: List[int],
    ):
        """
        Initializes the wrapper, processes the prefix, and stores the compressed cache.

        Args:
            press: The kvpress compression object (e.g., DuoAttentionPress).
            model: The pre-trained Hugging Face model.
            tokenizer: The tokenizer for the model.
            prefix_tokens: A list of token IDs representing the fixed prefix context.
        """
        self.press = press
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device  # Get device from model

        self.prefix_tokens = torch.tensor([prefix_tokens], device=self.device)
        self._compute_prefix_cache()

    def _should_output_attentions(self) -> bool:
        """
        Helper to determine if output_attentions=True is needed for the press method.
        """
        if isinstance(self.press, ObservedAttentionPress):
            return True
        if isinstance(
            self.press, (KeyRerotationPress, PerLayerCompressionPress)
        ) and isinstance(self.press.press, ObservedAttentionPress):
            return True
        return False

    def _compute_prefix_cache(self):
        cache = DynamicCache()
        with self.press(self.model), torch.inference_mode():  # Use inference mode
            # Process the prefix tokens to populate the cache
            self.model(
                input_ids=self.prefix_tokens,
                past_key_values=cache,
                output_attentions=self._should_output_attentions(),
                use_cache=True,  # Ensure cache is used and updated
                logits_to_keep=1,
            )
        self.prefix_cache = cache  # Store the populated/compressed cache

    def _copy_cache(self, cache_to_copy: DynamicCache) -> DynamicCache:
        """
        Creates a deep copy of the DynamicCache object's tensors.
        """
        new_cache = DynamicCache()
        if hasattr(cache_to_copy, "key_cache") and cache_to_copy.key_cache is not None:
            new_cache.key_cache = [k.clone() for k in cache_to_copy.key_cache]
        if (
            hasattr(cache_to_copy, "value_cache")
            and cache_to_copy.value_cache is not None
        ):
            new_cache.value_cache = [v.clone() for v in cache_to_copy.value_cache]

        # if hasattr(cache_to_copy, "seen_tokens"):
        #     new_cache.seen_tokens = cache_to_copy.seen_tokens

        assert not hasattr(new_cache, "_quantized_key_cache")

        return new_cache

    def generate(self, input_ids: List[int], max_new_tokens: int, **kwargs) -> str:
        """
        Generates text starting from the end of the compressed prefix,
        using the provided input_ids as the prompt continuation, via a manual
        token-by-token greedy decoding loop.

        Args:
            input_ids: A list of token IDs representing the new input (e.g., the question).
            max_new_tokens: The maximum number of new tokens to generate.
            **kwargs: Currently unused in this manual loop, but kept for signature consistency.
                      Could be extended for temperature, top_k etc.

        Returns:
            The generated text string, decoded by the tokenizer.
        """
        start_time = time.time()
        # Ensure input_ids is a tensor on the correct device, with batch dim [1, seq_len]
        input_tensor = torch.tensor([input_ids], device=self.device)

        # Make a copy of the prefix cache to modify during generation

        generated_token_ids: List[int] = []

        # Determine EOS token(s)
        eos_token_id_set = set()
        if self.model.generation_config.eos_token_id is not None:
            assert isinstance(self.model.generation_config.eos_token_id, list)
            eos_token_id_set.update(self.model.generation_config.eos_token_id)

        cache = self._copy_cache(self.prefix_cache)

        print("computing first token")
        # 1. Process the initial input_ids to update the cache
        outputs = self.model(
            input_ids=input_tensor,
            past_key_values=cache,
            use_cache=True,
            num_logits_to_keep=1,
        )
        # Get logits for the *next* token prediction (shape: [1, vocab_size])
        next_token_logits = outputs.logits[0, -1]

        # Start the generation loop (greedy decoding)
        for _ in tqdm.tqdm(range(max_new_tokens)):
            # 2. Select the next token ID (greedy approach)
            next_token_id = torch.argmax(next_token_logits, dim=-1)  # Shape: [1]

            # 3. Store the generated token ID
            token_id_item = next_token_id.item()  # Get the integer value
            generated_token_ids.append(token_id_item)

            # 4. Check for EOS condition
            if token_id_item in eos_token_id_set:
                # print(f"\nEOS token {token_id_item} detected. Stopping.")
                break

            # 5. Prepare input for the *next* iteration (just the last generated token)
            # Shape needs to be [1, 1]
            next_input_tensor = next_token_id.unsqueeze(-1).unsqueeze(-1)

            # 6. Run the model again with the single new token and updated cache
            outputs = self.model(
                input_ids=next_input_tensor, past_key_values=cache, use_cache=True
            )
            next_token_logits = outputs.logits[0, -1]

        # Decode the collected token IDs
        answer = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        return answer

    def kv_size_bytes(self):
        kv_numel = 0
        for tensor in itertools.chain(
            self.prefix_cache.key_cache,
            self.prefix_cache.value_cache,
        ):
            kv_numel += tensor.numel()
            assert tensor.dtype == torch.bfloat16
        kv_size = 2 * kv_numel  # 2 for bf16

        if isinstance(self.press, DuoAttentionPress):
            # with duo attention, the kv cache isn't actually modified
            return math.floor(kv_size * (1 - self.press.compression_ratio_))
        else:
            # otherwise, the kv cache is actually modified
            return kv_size


def make_press(
    press_type: Literal["duo", "duo_on_the_fly", "expected_attention"], ratio: float
) -> BasePress:
    if press_type == "duo":
        return DuoAttentionPress(head_compression_ratio=ratio)
    elif press_type == "duo_on_the_fly":
        return DuoAttentionPress(
            head_compression_ratio=ratio,
            on_the_fly_scoring=True,
        )
    elif press_type == "expected_attention":
        return ExpectedAttentionPress(compression_ratio=ratio)
    else:
        raise ValueError(
            f"Invalid kv_compression: {press_type}. Must be 'duo', 'duo_on_the_fly', or 'expected_attention'."
        )
