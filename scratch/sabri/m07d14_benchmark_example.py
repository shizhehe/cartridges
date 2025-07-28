from functools import lru_cache
from typing import Optional, List

import torch
import torch.nn.functional as F

from tabulate import tabulate
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_mask,
    flex_attention,
    _score_mod_signature,
    _mask_mod_signature,
)

from triton.testing import do_bench

from attn_gym.masks.document_mask import length_to_offsets
from attn_gym.masks import (
    causal_mask,
    generate_sliding_window,
    generate_prefix_lm_mask,
    generate_doc_mask_mod,
)
from attn_gym.mods import generate_alibi_bias, generate_tanh_softcap


AVAILABLE_EXAMPLES = {
    "causal": lambda: test_mask(mask_mod=causal_mask),
    "alibi": lambda: test_mask(score_mod=generate_alibi_bias(16), skip_correctness=True),
    "sliding_window": lambda: test_mask(mask_mod=generate_sliding_window(window_size=1024)),
    "prefix_lm": lambda: test_mask(mask_mod=generate_prefix_lm_mask(prefix_length=1024)),
    "document": lambda: run_document_masking(max_seq_len=32768, num_docs=12),
    "document_padding": lambda: run_document_masking_with_padding(max_seq_len=32768, num_docs=12),
    "softcap": lambda: test_mask(
        score_mod=generate_tanh_softcap(30, approx=False), skip_correctness=True
    ),
    "softcap_approx": lambda: test_mask(
        score_mod=generate_tanh_softcap(30, approx=True), skip_correctness=True
    ),
}


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
    mask_mod: Optional[_mask_mod_signature],
    H: int = 16,
    S: int = 8192,
    D: int = 64,
    seq_lens: List[int] = [128, 512, 768, 2048, 1024, 512],
    skip_correctness: bool = False,
    print_mask: bool = True,
    device: str = "cuda",
    include_padding: bool = False,
):
    # 
    batch_size = len(seq_lens)
    max_seq_len = max(seq_lens)
    total_seq_len = sum(seq_lens)

    seq_id = torch.cat(
        [torch.full((seq_len,), idx, dtype=torch.long) for idx, seq_len in enumerate(seq_lens)]
    ).to(device)
    def mask_mod(_, _h, q_idx, kv_idx):
        return (seq_id[q_idx] == seq_id[kv_idx]) & (q_idx >= kv_idx)
    
    block_mask = create_block_mask_cached(mask_mod, 1, 1, total_seq_len, total_seq_len, device=device)
    sdpa_mask_fn = mask_mod 
    mask = create_mask(sdpa_mask_fn, 1, 1, total_seq_len, total_seq_len, device=device)

    qkv_packed = [
        torch.randn(1, H, total_seq_len, D, device=device, dtype=data_type, requires_grad=True)
        for _ in range(3)
    ]

    # Create padded version by reshaping packed data
    qkv_padded = []
    for qkv_tensor in qkv_packed:
        padded_tensor = torch.zeros(batch_size, H, max_seq_len, D, device=device, dtype=data_type, requires_grad=True)
        start_idx = 0
        with torch.no_grad():
            for batch_idx, seq_len in enumerate(seq_lens):
                padded_tensor[batch_idx, :, :seq_len, :] = qkv_tensor[0, :, start_idx:start_idx + seq_len, :]
                start_idx += seq_len
        padded_tensor.requires_grad_(True)
        qkv_padded.append(padded_tensor)
    gradOut_padded = torch.randn(batch_size, H, max_seq_len, D, device=device, dtype=torch.float16)
    gradOut_packed = torch.randn(1, H, total_seq_len, D, device=device, dtype=torch.float16)

    causal_fa2 = lambda: F.scaled_dot_product_attention(*qkv_packed, is_causal=True)
    sdpa_mask = lambda: F.scaled_dot_product_attention(*qkv_packed, attn_mask=mask)
    flex_attention_call = lambda: flex_attention(*qkv_packed, block_mask=block_mask)
    padded_causal_fa2 = lambda: F.scaled_dot_product_attention(*qkv_padded, is_causal=True)

    results = []
    if block_mask is not None:
        density = (100 - block_mask.sparsity()) / 100
    else:
        density = 1.0
    causal_fav2_flops = 0.5 * batch_size * H * D * S * S
    flops = density * batch_size * H * D * S * S

    times = []
    functions_to_bench = [
        {
            "fn": causal_fa2,
            "name": "causal FA2",
            "gradOut": gradOut_packed,
            "flops": causal_fav2_flops,
        },
        {
            "fn": sdpa_mask,
            "name": "F.sdpa + mask",
            "gradOut": gradOut_packed,
            "flops": flops,
        },
        {
            "fn": flex_attention_call,
            "name": "flexattention",
            "gradOut": gradOut_packed,
            "flops": flops,
        },
        {
            "fn": padded_causal_fa2,
            "name": "padded causal FA2",
            "gradOut": gradOut_padded,
            "flops": flops,
        },
    ]
    
    for attn in functions_to_bench:
        fwd_time = do_bench(attn["fn"])
        fwd_out = attn["fn"]()
        
        bwd_time = do_bench(lambda: fwd_out.backward(attn["gradOut"], retain_graph=True))
        
        times.append((attn["name"], fwd_time, bwd_time))

        del fwd_out
        torch.cuda.empty_cache()

    print_header(
        f"{mask_mod.__name__}".replace(
            "_", " "
        ).title()
    )
    # Inline correctness check
    if not skip_correctness:
        padded_outs = []
        flex_outs = []

        for tensor in qkv_padded:
            tensor.grad = None

        out1 = padded_causal_fa2()
        padded_outs.append(out1)
        out1.backward(gradOut_padded)
        padded_outs += [tensor.grad for tensor in qkv_padded]

        for tensor in qkv_packed:
            tensor.grad = None

        out2 = flex_attention_call()
        flex_outs.append(out2)
        out2.backward(gradOut_packed)
        flex_outs += [tensor.grad for tensor in qkv_packed]
        
        # Compare padded output with flex attention output
        # Concatenate sequence data to match packed format
        padded_out_concat = torch.cat([padded_outs[0][i, :, :seq_lens[i], :] for i in range(len(seq_lens))], dim=1)
        padded_out_concat = padded_out_concat.unsqueeze(0)  # Add batch dimension to match flex output
        
        torch.testing.assert_close(flex_outs[0], padded_out_concat, atol=1e-1, rtol=1e-2)
        print("Correctness check passed ✅")



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
    print(
        tabulate(
            results,
            headers=[
                "Operation",
                "FW Time",
                "FW FLOPS",
                "BW Time",
                "BW FLOPS",
            ],
            tablefmt="grid",
        )
    )

    # Optionally print the block mask
    if print_mask:
        print("\nBlock Mask:\n", block_mask)


def run_document_masking(max_seq_len: int, num_docs: int):
    import random

    random.seed(0)

    def generate_random_lengths(total_length, num_documents):
        # Initialize all lengths to 1 to ensure each document has at least one token
        lengths = [1] * num_documents
        remaining_length = total_length - num_documents

        # Randomly distribute the remaining length
        for _ in range(remaining_length):
            index = random.randint(0, num_documents - 1)
            lengths[index] += 1

        return lengths

    lengths = generate_random_lengths(max_seq_len, num_docs)
    offsets = length_to_offsets(lengths, "cuda")
    document_causal_mask = generate_doc_mask_mod(causal_mask, offsets)
    test_mask(mask_mod=document_causal_mask, S=32768)


def run_document_masking_with_padding(max_seq_len: int, num_docs: int):
    import random

    random.seed(0)

    def generate_random_lengths(total_length, num_documents):
        # Initialize all lengths to 1 to ensure each document has at least one token
        lengths = [1] * num_documents
        remaining_length = total_length - num_documents

        # Randomly distribute the remaining length
        for _ in range(remaining_length):
            index = random.randint(0, num_documents - 1)
            lengths[index] += 1

        return lengths

    lengths = generate_random_lengths(max_seq_len, num_docs)
    offsets = length_to_offsets(lengths, "cuda")
    document_causal_mask = generate_doc_mask_mod(causal_mask, offsets)
    test_mask(mask_mod=document_causal_mask, S=32768, include_padding=True)


def main(examples: List[str] = ["all"]):
    """Run the benchmark with the given examples.

    Args:
        examples: List of examples to run. If "all" is specified, all examples will be run.
    """

    if "all" in examples:
        ex_to_run = list(AVAILABLE_EXAMPLES.keys())
        ex_to_run = ["document"]
    else:
        ex_to_run = examples

    for ex in ex_to_run:
        if ex in AVAILABLE_EXAMPLES:
            AVAILABLE_EXAMPLES[ex]()
            torch.cuda.empty_cache()
        else:
            print(f"Warning: Unknown example key '{ex}'. Skipping.")


if __name__ == "__main__":
    try:
        from jsonargparse import ArgumentParser
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    parser = ArgumentParser(description="Run specific examples or all examples.")
    parser.add_argument(
        "--examples",
        type=str,
        nargs="+",
        default=["all"],
        help="List of examples to run. Use space to separate multiple examples. "
        "Available options: "
        + ", ".join(sorted(AVAILABLE_EXAMPLES.keys()))
        + ", or 'all' to run all examples.",
    )

    args = parser.parse_args()
    main(**vars(args))