import os
from cartridges.datasets import TEMPLATE
from transformers import AutoTokenizer
import wandb
from cartridges.train import TrainConfig, CacheAndModel
from cartridges.cache import TrainableCache
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List, Optional, Union, TYPE_CHECKING
from peft import PeftModel


def load_model_and_cache_from_wandb(
    wandb_run_id: str,
    step: int,
    device: str = "cuda",
) -> tuple[CacheAndModel, AutoTokenizer]:
    is_ddp = "LOCAL_RANK" in os.environ
    is_rank_zero = (not is_ddp) or (dist.get_rank() == 0)

    train_config = TrainConfig.from_wandb(wandb_run_id, strict=False)

    model = train_config.model.instantiate().to(device)
    tokenizer = AutoTokenizer.from_pretrained(train_config.tokenizer)

    if is_rank_zero:
        out = wandb.restore(
            f"cache-step{step}.pt", run_path=wandb_run_id, root=train_config.run_dir
        )
    
    dist.barrier()
    cache = TrainableCache.from_pretrained(
        os.path.join(train_config.run_dir, f"cache-step{step}.pt"), 
        device=device
    )

    return CacheAndModel(cache=cache, model=model), tokenizer


def generate(
    input_ids: torch.Tensor,
    cache_and_model: Union[CacheAndModel, DDP, PeftModel],
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 32,
):
    
    cache = None
    
    if isinstance(cache_and_model, PeftModel):
        cache = None
    elif isinstance(cache_and_model, DDP):
        module: CacheAndModel = cache_and_model.module
        cache = module.cache if hasattr(module, 'cache') else None
    else:
        if hasattr(cache_and_model, 'cache'):
            cache = cache_and_model.cache
        else:
            cache = None
        
    # Initialize generation loop
    generated_tokens = []

    with torch.inference_mode():
        # Get the device from the model
        device = next(cache_and_model.parameters()).device
        input_ids = input_ids.to(device)
        past_key_values = None  # For tracking KV cache in standard generation
        
        for _ in range(max_new_tokens):
            # Get model outputs
            if cache is None:
                # Standard autoregressive generation for PEFT models
                outputs = cache_and_model(
                    input_ids=(input_ids if len(generated_tokens) == 0 else input_ids[:, -1:]),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                # Update KV cache for next iteration
                past_key_values = outputs.past_key_values
            else:
                # Custom cache-based generation
                outputs = cache_and_model(
                    input_ids=(input_ids if len(generated_tokens) == 0 else input_ids[:, -1:]),
                )
                assert past_key_values is None

            # Get next token prediction
            next_token = outputs.logits[:, -1:].argmax(dim=-1)
            # Stop if EOS token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token)

            # Update input_ids for next iteration
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
        # Clean up cache after generation
        if cache is not None:
            # SE(05/19): there used to be a code path that would manually clear, but it was not being used so we removed
            assert hasattr(cache, "clear")  
            cache.clear()

    if len(generated_tokens) == 0:
        return "[invalid]"
    return tokenizer.decode(torch.cat(generated_tokens, dim=-1)[0])


def get_loss(
    input_ids: torch.Tensor,
    cache_and_model, #Union['CacheAndModel', DDP, PeftModel],
    tokenizer: AutoTokenizer,
    answer_ids: torch.Tensor,
):
    original_seq_len = input_ids.shape[1]


    # Move tensors to device
    device = next(cache_and_model.parameters()).device
    input_ids = input_ids.to(device)
    answer_ids = answer_ids.to(device)

    # Perplexity calculation
    # Concatenate input and answer
    full_input = torch.cat([input_ids, answer_ids], dim=1)

    with torch.inference_mode():
        outputs = cache_and_model(full_input)
        logits = outputs.logits  # shape: [batch, seq_len, vocab]

        # Find where the answer starts in the full input
        answer_start = input_ids.shape[1] + 4
        
        # Predict the answer tokens only (i.e., target is answer_ids)
        shift_logits = logits[:, answer_start - 1:-1, :]  # predict from last input token onward
        shift_labels = answer_ids[:, 4:]

        from torch.nn import functional as F
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction='none',
        )
        total_loss = loss.sum()
        num_tokens = loss.numel()

    # Clean up cache after generation
    cache = cache_and_model.cache 
    if cache is not None:
        if hasattr(cache, "clear"):
            cache.clear()
        else:
            for layer_idx in range(len(cache.key_cache)):
                cache.key_cache[layer_idx] = cache.key_cache[layer_idx][
                    :, :, :original_seq_len
                ]
                cache.value_cache[layer_idx] = cache.value_cache[layer_idx][
                    :, :, :original_seq_len
                ]
    else:
        print("ahhhhhhhh cache is none")
        # assert 0, "Cache is None, but we are trying to clear it"


    return total_loss, num_tokens



def generate_samples(
    input_ids: torch.Tensor,  # shape (batch_size, seq_len)
    cache_and_model: Union[CacheAndModel, DDP, PeftModel],
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 32,
    num_samples: int = 16,
    temperature: float = 0.0,
    use_stop_token: bool = True,
):
    
    cache: Optional[TrainableCache] = None
    original_seq_len = input_ids.shape[1]
    
    if isinstance(cache_and_model, PeftModel):
        cache = None
        raise NotImplementedError("PEFT models not supported for batch generation")
    elif isinstance(cache_and_model, DDP):
        module: CacheAndModel = cache_and_model.module
        cache: TrainableCache = module.cache if hasattr(module, 'cache') else None
    else:
        if hasattr(cache_and_model, 'cache'):
            cache = cache_and_model.cache
            original_seq_len = cache.get_seq_length()
        else:
            breakpoint()
            raise ValueError("Unexpected")
    
    input_ids = input_ids.repeat(num_samples, 1)
    sample_idxs = torch.arange(num_samples)  # needed since the shape of input_ids will change
    # Initialize generation loop
    generated_tokens = [[] for _ in range(num_samples)]

    with torch.inference_mode():
        # Get the device from the model
        device = next(cache_and_model.parameters()).device
        input_ids = input_ids.to(device)
        past_key_values = None  # For tracking KV cache in standard generation
        
        for _ in range(max_new_tokens):
            # Get model outputs
            if cache is None:
                assert False
                # Standard autoregressive generation for PEFT models
                outputs = cache_and_model(
                    input_ids=(input_ids if len(generated_tokens) == 0 else input_ids[:, -1:]),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                # Update KV cache for next iteration
                past_key_values = outputs.past_key_values
            else:
                # Custom cache-based generation
                outputs = cache_and_model(
                    input_ids=(input_ids if len(generated_tokens[0]) == 0 else input_ids[:, -1:]),
                )
                assert past_key_values is None
            
            
            # Get next token prediction
            next_sample_idxs, next_input_ids = [], []
            next_key_cache = [[] for _ in range(len(cache.key_cache))]
            next_value_cache = [[] for _ in range(len(cache.value_cache))]
            for input_idx, sample_idx in enumerate(sample_idxs):
                if temperature == 0.0:
                    next_token = outputs.logits[input_idx, -1:].argmax(dim=-1)
                else:
                    scaled_logits = outputs.logits[input_idx, -1:] / temperature
                    probs = scaled_logits.softmax(dim=-1)
                    next_token = probs.multinomial(num_samples=1, replacement=True)[0]
                generated_tokens[sample_idx].append(next_token)
                
                # Stop if EOS token is generated
                if not use_stop_token or next_token.item() != tokenizer.eos_token_id:
                    next_sample_idxs.append(sample_idx)
                    next_input_ids.append(torch.cat([input_ids[input_idx], next_token], dim=-1))
                    
                    for layer_idx in range(len(cache.key_cache)):
                        next_key_cache[layer_idx].append(cache.key_cache[layer_idx][input_idx])
                        next_value_cache[layer_idx].append(cache.value_cache[layer_idx][input_idx])
            
            if len(next_sample_idxs) == 0:
                break
            
            sample_idxs = torch.tensor(next_sample_idxs)
            input_ids = torch.stack(next_input_ids, dim=0)
            cache.key_cache = [torch.stack(next_key_cache[layer_idx], dim=0) for layer_idx in range(len(next_key_cache))]
            cache.value_cache = [torch.stack(next_value_cache[layer_idx], dim=0) for layer_idx in range(len(next_value_cache))]

        # Clean up cache after generation
        if cache is not None:
            if hasattr(cache, "clear"):
                cache.clear()
            else:
                for layer_idx in range(len(cache.key_cache)):
                    cache.key_cache[layer_idx] = cache.key_cache[layer_idx][
                        :, :, :original_seq_len
                    ]
                    cache.value_cache[layer_idx] = cache.value_cache[layer_idx][
                        :, :, :original_seq_len
                    ]

    if len(generated_tokens) == 0:
        return "[invalid]"
    return [tokenizer.decode(torch.tensor(toks)) for toks in generated_tokens]



def generate_batch(
    input_ids: List[torch.Tensor],  # shape (batch_size, seq_len)
    cache_and_model: Union[CacheAndModel, DDP, PeftModel],
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 32,
    num_samples: int = 16,
    temperature: float = 0.0,
    use_stop_token: bool = True,
):
    is_ddp = "LOCAL_RANK" in os.environ
    is_rank_zero = (not is_ddp) or (dist.get_rank() == 0)
    
    cache: Optional[TrainableCache] = None
    num_inputs = len(input_ids)
    
    if isinstance(cache_and_model, DDP):
        is_peft = False
        module: CacheAndModel = cache_and_model.module
        cache: TrainableCache = module.cache if hasattr(module, 'cache') else None
    else:
        if hasattr(cache_and_model, 'cache'):
            cache = cache_and_model.cache
            original_seq_len = cache.get_seq_length()
        else:
            cache = None
    is_peft = cache is None
    
    input_ids = [x for x in input_ids for _ in range(num_samples)]  # we want to repeat each input num_samples, not interleave
    num_generations = num_inputs * num_samples
    seq_lens = [input_ids.shape[0] for input_ids in input_ids]
    orig_idxs = torch.arange(num_generations)  # needed since the shape of input_ids will change
    generated_tokens = [[] for _ in range(num_generations)]
    with torch.inference_mode():
        # Get the device from the model
        device = next(cache_and_model.parameters()).device
        input_ids = [x.to(device) for x in input_ids]
        curr_input_ids = torch.stack([x[:min(seq_lens)] for x in input_ids], dim=0)
        curr_orig_idxs = orig_idxs
        
        for curr_token_idx in range(min(seq_lens), max(seq_lens) + max_new_tokens + 1):
            # Get model outputs
            if curr_token_idx == min(seq_lens): # prefill on first token
                x = curr_input_ids
            else: # decode after first token
                x = curr_input_ids[:, -1:]  

            if is_peft:
                # Standard autoregressive generation for PEFT models
                outputs = cache_and_model(
                    input_ids=x,
                    past_key_values=cache,
                    use_cache=True,
                )
                # Update KV cache for next iteration
                cache = outputs.past_key_values
            else:
                # Custom cache-based generation
                outputs = cache_and_model(input_ids=x)
            
            
            # Get next token prediction
            next_orig_idxs, next_input_ids = [], []
            next_key_cache = [[] for _ in range(len(cache.key_cache))]
            next_value_cache = [[] for _ in range(len(cache.value_cache))]
            for input_idx, orig_idx in enumerate(curr_orig_idxs):
                in_prefill = seq_lens[orig_idx] > curr_input_ids.shape[1]
                if in_prefill:
                    # this input has not yet been fully prefilled yet, so we need
                    next_token = input_ids[orig_idx][curr_input_ids.shape[1]].unsqueeze(0)

                else:
                    if temperature == 0.0:
                        next_token = outputs.logits[input_idx, -1:].argmax(dim=-1)
                    else:
                        scaled_logits = outputs.logits[input_idx, -1:] / temperature
                        probs = scaled_logits.softmax(dim=-1)
                        next_token = probs.multinomial(num_samples=1, replacement=True)[0]
                    generated_tokens[orig_idx].append(next_token)
                
                # construct next_* lists based on the stop conditions for the sequence
                if (
                    (
                        in_prefill or
                        not use_stop_token or 
                        next_token.item() != tokenizer.eos_token_id
                    ) and curr_token_idx < seq_lens[orig_idx] + max_new_tokens - 1 
                ):
                    next_orig_idxs.append(orig_idx)
                    next_input_ids.append(torch.cat([curr_input_ids[input_idx], next_token], dim=-1))
                    
                    for layer_idx in range(len(cache.key_cache)):
                        next_key_cache[layer_idx].append(cache.key_cache[layer_idx][input_idx])
                        next_value_cache[layer_idx].append(cache.value_cache[layer_idx][input_idx])
                
            # # print memory consumption
            # if is_rank_zero:
            #     print(torch.cuda.memory_allocated() / 1024**2, "MB")
            
            if len(next_orig_idxs) == 0:
                break
            
            curr_orig_idxs = torch.tensor(next_orig_idxs)
            curr_input_ids = torch.stack(next_input_ids, dim=0)
            cache.key_cache = [torch.stack(next_key_cache[layer_idx], dim=0) for layer_idx in range(len(next_key_cache))]
            cache.value_cache = [torch.stack(next_value_cache[layer_idx], dim=0) for layer_idx in range(len(next_value_cache))]

        # Clean up cache after generation
        if is_peft:
            cache = None
        else: 
            assert hasattr(cache, "clear")
            cache.clear()

    if len(generated_tokens) == 0:
        return "[invalid]"
    return [tokenizer.decode(torch.tensor(toks)) for toks in generated_tokens]


if __name__ == "__main__":
    cache_and_model, tokenizer = load_model_and_cache_from_wandb(   
        wandb_run_id="hazy-research/Cartridges/ghm1tny6",
        step=64,
        device="cuda"
    )
    

    messages = [
        {"role": "user", "content": "What is the capital of the moon?"},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True,
        return_tensors="pt",
        chat_template=TEMPLATE,
    ).to("cuda")
    output = generate(input_ids, cache_and_model, tokenizer, max_new_tokens=32)
    print(output)

    breakpoint()