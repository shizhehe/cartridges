from functools import lru_cache
from typing import List

import torch
import torch.nn.functional as F

from tabulate import tabulate
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_mask,
    flex_attention,
)

from triton.testing import do_bench


torch.set_default_device("cuda")
torch.manual_seed(0)

torch._dynamo.config.cache_size_limit = 1000

# Compile the flex_attention function
flex_attention = torch.compile(flex_attention, dynamic=False)

# For better performance, you can use:
# flex_attention = torch.compile(_flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")

data_type = torch.float16

# The kernels will utilize block sparsity to increase performance
print(f"Using the default sparsity block size: {_DEFAULT_SPARSE_BLOCK_SIZE}")


@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask


def calculate_tflops(flops: float, time_ms: float, multiplier: int) -> float:
    return multiplier * flops * (1e3 / time_ms) / 1e12


def print_header(text):
    width = 91
    print("╔" + "═" * (width - 2) + "╗")
    print(f"║ {text.center(width - 4)} ║")
    print("╚" + "═" * (width - 2) + "╝")


def test_mask(
    num_heads: int = 16,
    head_dim: int = 64,
    seq_lens: List[int] = [128, 512, 768, 2048, 1024, 512],
    prefix_len: int = 1024,
    skip_correctness: bool = False,
    print_mask: bool = True,
    device: str = "cuda",
    include_padding: bool = False,
):
    batch_size = len(seq_lens)
    max_seq_len = max(seq_lens)
    total_seq_len = sum(seq_lens)


    # Creating the block mask
    # --- begin create block mask ---
    seq_id = torch.cat(
        [torch.full((prefix_len,), -1, dtype=torch.long)] + 
        [torch.full((seq_len,), idx, dtype=torch.long) for idx, seq_len in enumerate(seq_lens)]
    ).to(device)
    def mask_mod(_, _h, q_idx, kv_idx):
        return (kv_idx < prefix_len) | ((seq_id[q_idx + prefix_len] == seq_id[kv_idx]) & (q_idx + prefix_len >= kv_idx))

    block_mask = create_block_mask_cached(mask_mod, 1, 1, total_seq_len, total_seq_len + prefix_len, device=device)
    sdpa_mask_fn = mask_mod 
    mask = create_mask(sdpa_mask_fn, 1, 1, total_seq_len, total_seq_len + prefix_len, device=device)

    seq_lens_tensor = torch.tensor(seq_lens, device=device)
    def padded_mask_mod(batch_idx, _h, q_idx, kv_idx):
        # return (kv_idx < prefix_len) | ((q_idx + prefix_len >= kv_idx)  & q_idx < seq_lens_tensor[batch_idx])
        return (kv_idx < prefix_len) | ((q_idx + prefix_len >= kv_idx))

    padded_mask = create_mask(
        lambda b_idx, _h, q_idx, kv_idx: (q_idx < seq_lens_tensor[b_idx]) & ((kv_idx < prefix_len) | (q_idx + prefix_len >= kv_idx)),
        B=batch_size, H=1, Q_LEN=max_seq_len, KV_LEN=prefix_len + max_seq_len, device=device
    )
    # --- end create block mask ---
    
    # Print the block mask
    # --- begin print block mask ---
    if print_mask:
        print(seq_lens)
        print(block_mask.to_string(grid_size=-1))
    # --- end print block mask ---

    tensor_kwargs = dict(device=device, dtype=data_type, requires_grad=True)
    qkv_packed = [
        torch.randn(1, num_heads, total_seq_len, head_dim, **tensor_kwargs),
        torch.randn(1, num_heads, total_seq_len + prefix_len, head_dim, **tensor_kwargs),
        torch.randn(1, num_heads, total_seq_len + prefix_len, head_dim, **tensor_kwargs),
    ]
    grad_out_packed = torch.randn(1, num_heads, total_seq_len, head_dim, device=device, dtype=torch.float16)


    # Create padded version by reshaping packed data
    qkv_padded = []
    for qkv_tensor, name in zip(qkv_packed, ["query", "key", "value"]):
        offset = 0 if name == "query" else prefix_len
        padded_tensor = torch.zeros(batch_size, num_heads, max_seq_len + offset, head_dim, **tensor_kwargs)
        start_idx = offset
        with torch.no_grad():  # needed to avoid "a view of leaf Variable that requires..."
            for batch_idx, seq_len in enumerate(seq_lens):
                padded_tensor[batch_idx, :, offset:offset + seq_len, :] = qkv_tensor[0, :, start_idx:start_idx + seq_len, :]
                start_idx += seq_len
            if name != "query":
                # copy the prefix into the padded tensor
                padded_tensor[:, :, :prefix_len, :] = qkv_tensor[:, :, :prefix_len, :]
        padded_tensor.requires_grad_(True)
        qkv_padded.append(padded_tensor)
    grad_out_padded = torch.zeros(batch_size, num_heads, max_seq_len, head_dim, device=device, dtype=torch.float16)
    start_idx = 0
    for b_idx, seq_len in enumerate(seq_lens):
        grad_out_padded[b_idx, :, :seq_len, :] = grad_out_packed[0, :, start_idx:start_idx + seq_len, :]
        start_idx += seq_len

    

    # --- Begin compute FLOPs ---
    if block_mask is not None:
        density = (100 - block_mask.sparsity()) / 100
    else:
        density = 1.0
    flops = density * num_heads * head_dim * total_seq_len ** 2
    # --- End compute FLOPs ---

    # Prepare functions to benchmark
    # --- begin prepare ---
    functions_to_bench = [
        {
            "fn": lambda: F.scaled_dot_product_attention(*qkv_packed, attn_mask=mask),
            "name": "F.sdpa + mask",
            "gradOut": grad_out_packed,
            "flops": flops,
        },
        {
            "fn": lambda: flex_attention(*qkv_packed, block_mask=block_mask),
            "name": "flexattention",
            "gradOut": grad_out_packed,
            "flops": flops,
        },
        {
            "fn": lambda: F.scaled_dot_product_attention(*qkv_padded, attn_mask=padded_mask),
            "name": "padded causal FA2",
            "gradOut": grad_out_padded,
            "flops": flops,
        },
    ]
    # --- end prepare ---


    #  Benchmark functions 
    # --- begin benchmark ---
    times = []
    for attn in functions_to_bench:
        fwd_time = do_bench(attn["fn"])
        fwd_out = attn["fn"]()
        
        bwd_time = do_bench(lambda: fwd_out.backward(attn["gradOut"], retain_graph=True))
        
        times.append((attn["name"], fwd_time, bwd_time))

        del fwd_out
        torch.cuda.empty_cache()

    # --- end benchmark ---

    # Print the results
    # --- begin print results ---
    results = []
    for (label, fw_time, bw_time) in times:
        result = [
            label,
            f"{fw_time:.4f} ms",
            f"{calculate_tflops(attn["flops"], fw_time, 4):.2f} TF/s",
            f"{bw_time:.4f} ms",
            f"{calculate_tflops(attn["flops"], bw_time, 10):.2f} TF/s",
        ]
        results.append(result)

    # Print the results in a tabulated format
    # --- begin print results ---
    print(
        tabulate(
            results,
            headers=["Operation", "FW Time", "FW FLOPS", "BW Time", "BW FLOPS"],
            tablefmt="grid",
        )
    )
    # --- end print results ---


    # Check the correctness of the implementations
    # --- begin correctness check ---
    print_header(f"{mask_mod.__name__}".replace("_", " ").title())
    if not skip_correctness:
        # For correctness, compare flex attention with F.sdpa using the same mask
        for tensor in qkv_packed:
            tensor.grad = None

        # Use flex attention
        flex_out = flex_attention(*qkv_packed, block_mask=block_mask)
        flex_out.backward(grad_out_packed, retain_graph=True)
        flex_grads = [tensor.grad.clone() for tensor in qkv_packed]
        
        for tensor in qkv_packed:
            tensor.grad = None

        # Use F.sdpa with the same mask for comparison
        sdpa_out = F.scaled_dot_product_attention(*qkv_packed, attn_mask=mask)
        sdpa_out.backward(grad_out_packed)
        sdpa_grads = [tensor.grad for tensor in qkv_packed]

        # Use with padding 
        pad_out = F.scaled_dot_product_attention(*qkv_padded, attn_mask=padded_mask)
        pad_out.backward(grad_out_padded)
        # SE (07/20): TODO implement grad check for padding
        pad_grads = [tensor.grad for tensor in qkv_padded]
        pad_out = torch.cat(
            [pad_out[batch_idx, :, :seq_len, :] for batch_idx, seq_len in enumerate(seq_lens)],
            dim=1
        ).unsqueeze(0)
        
        # Compare outputs and gradients
        torch.testing.assert_close(flex_out, sdpa_out, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(flex_out, pad_out, atol=1e-3, rtol=1e-3)
            
        for flex_grad, sdpa_grad, pad_grad in zip(flex_grads, sdpa_grads, pad_grads):
            torch.testing.assert_close(flex_grad, sdpa_grad, atol=1e-3, rtol=1e-3)
        print("Correctness check passed ✅")
    # --- end correctness check ---


if __name__ == "__main__":
    import random

    random.seed(0)

    test_mask(
        skip_correctness=False, 
        seq_lens=[128, 512, 768, 1024, 512],
        prefix_len=8192
    )

    