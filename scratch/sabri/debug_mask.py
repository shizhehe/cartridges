import torch
from torch.nn.attention.flex_attention import create_block_mask

device = "cuda"

# Simple case: single sequence of length 4
seq_len = 4
seq_ids = torch.zeros(seq_len, dtype=torch.long, device=device)  # All belong to sequence 0

print(f"seq_ids: {seq_ids}")

# The current mask function from FlexQwen3Model
def current_mask_func(_, _h, q_idx, kv_idx):
    prefix_len = 0
    kv_seq_ids = seq_ids  # No prefix for this test
    result = (kv_idx < prefix_len) | ((seq_ids[q_idx] == kv_seq_ids[kv_idx]) & (q_idx + prefix_len >= kv_idx))
    return result

# What it should be for causal attention
def correct_mask_func(_, _h, q_idx, kv_idx):
    prefix_len = 0
    kv_seq_ids = seq_ids  # No prefix for this test
    result = (kv_idx < prefix_len) | ((seq_ids[q_idx] == kv_seq_ids[kv_idx]) & (q_idx >= kv_idx))
    return result

print("Current mask function results:")
for q in range(seq_len):
    for kv in range(seq_len):
        result = current_mask_func(None, None, q, kv)
        print(f"q={q}, kv={kv}: {result}")

print("\nCorrect mask function results:")
for q in range(seq_len):
    for kv in range(seq_len):
        result = correct_mask_func(None, None, q, kv)
        print(f"q={q}, kv={kv}: {result}")

# Test with block masks
current_block_mask = create_block_mask(current_mask_func, 1, 1, seq_len, seq_len, device=device)
correct_block_mask = create_block_mask(correct_mask_func, 1, 1, seq_len, seq_len, device=device)

print(f"\nCurrent block mask shape: {current_block_mask.shape if hasattr(current_block_mask, 'shape') else 'N/A'}")
print(f"Correct block mask shape: {correct_block_mask.shape if hasattr(correct_block_mask, 'shape') else 'N/A'}")