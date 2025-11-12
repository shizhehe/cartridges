from collections import defaultdict
import os
from typing import Any, Dict, List, Optional
from transformers import DynamicCache, AutoTokenizer
import torch
from tqdm import tqdm

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.utils import get_logger

logger = get_logger(__name__)


def flex_generate(
    model,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    seq_ids: torch.Tensor,
    position_ids: torch.Tensor,
    cache: Optional[TrainableCache] = None,
    stop_token_ids: Optional[List[int]] = None,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    show_progress: bool = False,
    is_peft: bool = False,
    max_repetitions: int = 5,
) -> Dict[int, List[int]]:
    """Autoregressive generation with FlexAttention (e.g. FlexLlamaModel, FlexQwen3Model).
    
    Args:
        model: The model to use for generation
        input_ids: (N,) tensor of input ids where N is the total number of tokens across 
            the sequences.
        seq_ids: (N,) tensor specifying the membership of each token to a sequence
        position_ids: (N,) tensor of position of a token within it's sequence
        stop_token_ids: By default, will use the end of text id from the tokenizer.
        tokenizer: tokenizer to use for decoding
        max_new_tokens: maximum number of new tokens to generate.
        temperature: temperature for sampling
        show_progress: whether to show a progress bar during generation
        max_repetitions: maximum number of consecutive identical tokens before stopping generation
    
    This implementation relies on the PackedCache above.
    """
    print(tokenizer.decode(input_ids))
    
    # Debug: Check model state in generation function
    logger.info(f"flex_generate called with model type: {type(model)}")
    logger.info(f"is_peft flag: {is_peft}")
    logger.info(f"cache provided: {cache is not None}")
    logger.info(f"temperature: {temperature}")
    logger.info(f"max_new_tokens: {max_new_tokens}")
    
    if is_peft:
        from peft import PeftModel
        logger.info(f"Using PEFT model: {isinstance(model, PeftModel)}")
        if hasattr(model, 'active_adapters'):
            logger.info(f"PEFT active adapters: {model.active_adapters}")
        if hasattr(model, 'peft_config'):
            logger.info(f"PEFT config: {model.peft_config}")
            
        # Check LoRA B matrix initialization right before generation
        logger.info("Checking LoRA B matrix weights before generation...")
        non_zero_count = 0
        total_b_matrices = 0
        for name, param in model.named_parameters():
            if 'lora_B' in name and param.requires_grad:
                max_val = param.data.abs().max().item()
                norm_val = param.data.norm().item()
                total_b_matrices += 1
                if max_val > 1e-6:
                    logger.warning(f"NON-ZERO LoRA B matrix {name}: max={max_val:.6f}, norm={norm_val:.6f}")
                    non_zero_count += 1
                else:
                    logger.debug(f"Zero LoRA B matrix {name}: max={max_val:.6f}, norm={norm_val:.6f}")
        
        logger.info(f"LoRA B matrices: {non_zero_count}/{total_b_matrices} are non-zero")
        if non_zero_count == 0:
            logger.info("✓ All LoRA B matrices are zero - adapter should behave like base model")
        else:
            logger.warning(f"⚠ {non_zero_count} LoRA B matrices are non-zero - adapter may affect generation")
            
    device = input_ids.device
    model.eval()
    if stop_token_ids is None:
        stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    
    if cache is None and not is_peft:
        cache = TrainableCache(
            config=AttnConfig(
                n_layers=model.config.num_hidden_layers,
                n_heads=model.config.num_key_value_heads,
                head_dim=model.config.head_dim,
            ),
        )
        
    logger.info(f"Processing sequence with {len(input_ids)} tokens")
    logger.info(f"Input tokens: {input_ids.tolist()}")
    logger.info(f"Input text: {repr(tokenizer.decode(input_ids, skip_special_tokens=True))}")
    
    # Test with a simple known input for comparison
    test_text = "The capital of France is "
    test_tokens = tokenizer.encode(test_text, return_tensors="pt").to(device).flatten()
    test_seq_ids = torch.zeros(len(test_tokens), dtype=torch.long, device=device)
    test_position_ids = torch.arange(len(test_tokens), dtype=torch.long, device=device)
    
    logger.info(f"Test comparison - '{test_text}': {test_tokens.tolist()}")
    logger.info(f"Test comparison decoded: {repr(tokenizer.decode(test_tokens, skip_special_tokens=True))}")
    
    # Test with base model directly (bypass LoRA and flex_generate)
    if is_peft:
        logger.info("Testing base model directly...")
        try:
            # Get the actual base FlexQwen3 model
            base_model = model.base_model.model  # PeftModel -> LoraModel -> FlexQwen3ForCausalLM
            logger.info(f"Base model type: {type(base_model)}")
            
            with torch.no_grad():
                # Test with direct base model access (bypass PEFT entirely)
                try:
                    # Direct forward pass with base FlexQwen3 model
                    base_output = base_model(
                        input_ids=test_tokens,
                        seq_ids=test_seq_ids,
                        position_ids=test_position_ids,
                        attention_mask=None,
                        past_key_values=None,
                        use_cache=True,
                        mode="generate",
                    )
                    base_logits = base_output.logits[0, -1, :]
                    base_next_token = base_logits.argmax().item()
                    base_next_word = tokenizer.decode(base_next_token)
                    logger.info(f"Base model (direct access) prediction: {base_next_token} ({repr(base_next_word)})")
                    
                except Exception as e:
                    logger.error(f"Base model test failed: {e}")
                    
        except Exception as e:
            logger.error(f"Could not access base model: {e}")
    
    # Quick test generation on both inputs
    with torch.no_grad():
        # Test on simple input
        test_output = model(
            input_ids=test_tokens,
            seq_ids=test_seq_ids,
            position_ids=test_position_ids,
            attention_mask=None,
            past_key_values=None,
            use_cache=True,
            mode="generate",
        )
        test_logits = test_output.logits[0, -1, :]
        test_next_token = test_logits.argmax().item()
        test_next_word = tokenizer.decode(test_next_token)
        logger.info(f"Test input next token prediction: {test_next_token} ({repr(test_next_word)})")
        
        # Test on actual input
        actual_output = model(
            input_ids=input_ids,
            seq_ids=seq_ids,
            position_ids=position_ids,
            attention_mask=None,
            past_key_values=None,
            use_cache=True,
            mode="generate",
        )
        actual_logits = actual_output.logits[0, -1, :]
        actual_next_token = actual_logits.argmax().item()
        actual_next_word = tokenizer.decode(actual_next_token)
        logger.info(f"Actual input next token prediction: {actual_next_token} ({repr(actual_next_word)})")
    
    # Initialize generated sequences
    generated_tokens: Dict[int, List[int]] = defaultdict(list)
    
    # Track repetitions per sequence
    repetition_counts: Dict[int, int] = defaultdict(int)
    last_tokens: Dict[int, Optional[int]] = defaultdict(lambda: None)
    
    # Current state
    current_input_ids = input_ids
    current_seq_ids = seq_ids
    current_position_ids = position_ids 

    past_key_values = None
    
    progress_range = tqdm(range(max_new_tokens), desc="Generating", disable=not show_progress, leave=False)
    for step in progress_range:
        # Forward pass - update cache with current seq_ids before the forward pass
        with torch.no_grad():
            # FlexQwen3 handles causal masking via FlexAttention and seq_ids
            # No need for explicit attention mask since we don't have padded tokens
            attention_mask = None
            
            outputs = model(
                input_ids=current_input_ids,
                seq_ids=current_seq_ids,
                position_ids=current_position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values if is_peft else cache,
                use_cache=True,
                mode="generate",
            )

            past_key_values = outputs.past_key_values
        
        # Get logits for the last token of each sequence
        logits = outputs.logits  # (1, seq_len, vocab_size)
        last_logits = logits[0, -len(current_input_ids):, :]  # Get logits for current tokens
        
        # Sample next tokens for each sequence
        next_tokens = []
        next_seq_ids = []
        next_position_ids = []
        
        # Process single sequence (already filtered in training script)
        # Get the last token's logits (use last token in sequence)
        token_logits = last_logits[-1]  # Last token in the batch
        
        # Apply temperature
        if temperature > 0:
            token_logits = token_logits / temperature
            next_token = torch.multinomial(torch.softmax(token_logits, dim=-1), 1).item()
        else:
            next_token = token_logits.argmax().item()
        
        # Use the actual sequence ID from input
        seq_id_int = current_seq_ids[0].item()
        
        # Check for repetition
        if last_tokens[seq_id_int] == next_token:
            repetition_counts[seq_id_int] += 1
        else:
            repetition_counts[seq_id_int] = 0
            last_tokens[seq_id_int] = next_token
        
        # Check if sequence should continue
        should_stop = (next_token in stop_token_ids or 
                      repetition_counts[seq_id_int] >= max_repetitions)
        
        if should_stop:
            if repetition_counts[seq_id_int] >= max_repetitions:
                logger.info(f"Stopping sequence {seq_id_int} due to {max_repetitions} consecutive repetitions of token {next_token} ({tokenizer.decode([next_token])})")
            progress_range.close()
            break
        
        # Continue generation
        next_tokens = [next_token]
        next_seq_ids = [seq_id_int]  # Use actual sequence ID
        next_position_ids = [current_position_ids[-1] + 1]  # Increment last position
        generated_tokens[seq_id_int].append(next_token)
        
        # Prepare inputs for next iteration
        current_input_ids = torch.tensor(next_tokens, device=device, dtype=torch.long)
        current_seq_ids = torch.tensor(next_seq_ids, device=device, dtype=torch.long)
        current_position_ids = torch.tensor(next_position_ids, device=device, dtype=torch.long)
        
    # SE (07/26): Very important to clear the cache after generation, otherwise, during
    # training, the keys and values from the last generation will be included
    # This issue is silent when training on a single GPU, but becomes apparent when
    # training on multiple GPUs. We get a crash on flex attention I guess because the 
    # cache sizes differ between GPUs.
    if not is_peft:
        cache.clear()
    
    return generated_tokens
    


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    from cartridges.utils.wandb import load_model_and_cache_from_wandb


    # Define command line argument parser
    parser = argparse.ArgumentParser(description="Select model type")
    parser.add_argument("--model", default="llama", help="Choose between 'llama' and 'qwen' models")
    args = parser.parse_args()

    # Import the appropriate model based on the command line argument
    if args.model == "llama":
        from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        model = FlexLlamaForCausalLM.from_pretrained(model_name).to("cuda").to(torch.bfloat16)
        cache = None
    elif args.model == "qwen":
        from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
        model_name = "Qwen/Qwen3-4B"
        model = FlexQwen3ForCausalLM.from_pretrained(model_name).to("cuda").to(torch.bfloat16)
        cache = None

    elif args.model.startswith("hazy-research"):
        cache_and_model = load_model_and_cache_from_wandb(
            wandb_run_id="hazy-research/cartridges/ehij7vlt",
            step=29,
        )
        model_name = cache_and_model.model.name_or_path
        cache = cache_and_model.cache.to("cuda").to(torch.bfloat16)
        model = cache_and_model.model.to("cuda").to(torch.bfloat16)
    else:
        raise ValueError(f"Model {args.model} not supported")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    convos = [
        [
            {"role": "user", "content": "What is the capital of the moon?"},
        ],
        [
            {"role": "user", "content": "Who is the patient?"},
        ],
        # [
        #     {"role": "user", "content": "Why is the sky blue?"},
        # ],
    ]

    input_ids, seq_ids, position_ids = [], [], []
    for idx, convo in enumerate(convos):
        curr_input_ids = tokenizer.apply_chat_template(
            convo, 
            tokenize=True, 
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
        # Flatten the input_ids and create corresponding seq_ids and position_ids
        flat_input_ids = curr_input_ids.flatten()
        curr_seq_ids = torch.full((flat_input_ids.shape[0],), idx, dtype=torch.long, device="cuda")
        curr_position_ids = torch.arange(flat_input_ids.shape[0], device="cuda")
        
        input_ids.append(flat_input_ids)
        seq_ids.append(curr_seq_ids)
        position_ids.append(curr_position_ids)
    
    input_ids = torch.cat(input_ids, dim=0)
    seq_ids = torch.cat(seq_ids, dim=0)
    position_ids = torch.cat(position_ids, dim=0)

    print("Starting generation...")
    print(f"Input shapes: input_ids={input_ids.shape}, seq_ids={seq_ids.shape}, position_ids={position_ids.shape}")

    if cache is None:
        cache = TrainableCache(
            max_seq_len=1024,
            config=AttnConfig(
                n_layers=model.config.num_hidden_layers,
                n_heads=model.config.num_key_value_heads,
                head_dim=model.config.head_dim,
            ),
            device=str(input_ids.device),
        )
    
    output = flex_generate(
        model=model,
        input_ids=input_ids,
        seq_ids=seq_ids,
        position_ids=position_ids,
        tokenizer=tokenizer,
        max_new_tokens=128,  # Reduce for testing
        show_progress=True,
        cache=cache,
        max_repetitions=5, 
    )
    print("Generated tokens:", output)
    
    # Decode the output
    for seq_idx, tokens in output.items():
        if tokens:
            decoded = tokenizer.decode(tokens)
            print(decoded)
            print(f"Sequence {seq_idx}: {decoded}")