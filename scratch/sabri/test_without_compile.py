import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from cartridges.models.qwen.modeling_qwen3 import (
    FlexQwen3Model,
)
from cartridges.models.qwen.configuration_qwen3 import Qwen3Config

from transformers import Qwen3Model, Qwen3Config

# Temporarily disable the torch.compile
import cartridges.models.qwen.modeling_qwen3 as modeling_qwen3
modeling_qwen3.flex_attention = flex_attention  # Use uncompiled version

device = "cuda"

def small_qwen_config():
    """Create a small Qwen3 config for testing."""
    return Qwen3Config(
        vocab_size=1000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=64,
        max_position_embeddings=512,
        use_cache=False,
        layer_types=["full_attention", "full_attention"],
    )

seq_lens = [64]
batch_size = len(seq_lens)
max_seq_len = max(seq_lens)
total_seq_len = sum(seq_lens)

seq_id = torch.cat(
    [torch.full((seq_len,), idx, dtype=torch.long, device=device) for idx, seq_len in enumerate(seq_lens)]
)
input_ids = torch.randint(0, 1000, (1, total_seq_len)).to(device)

padded_input_ids = torch.full((batch_size, max_seq_len), 0, dtype=torch.long, device=device)
start_idx = 0   
for i, seq_len in enumerate(seq_lens):
    padded_input_ids[i, :seq_len] = input_ids[0, start_idx:start_idx+seq_len]
    start_idx += seq_len

config = small_qwen_config()
model = FlexQwen3Model(config).to(device)
ref_model = Qwen3Model(config).to(device)
ref_model.load_state_dict(model.state_dict())

print("Running FlexQwen3Model (without torch.compile)...")
out = model(input_ids, seq_ids=seq_id).last_hidden_state

print("Running reference Qwen3Model...")
ref_out_padded = ref_model(padded_input_ids).last_hidden_state

ref_out = torch.cat(
    [ref_out_padded[batch_idx, :seq_len] for batch_idx, seq_len in enumerate(seq_lens)],
    dim=0
).unsqueeze(0)

print(f"Outputs equal? {torch.allclose(out, ref_out, atol=1e-3, rtol=1e-3)}")
print(f"Max absolute difference: {(out - ref_out).abs().max()}")
print(f"FlexQwen3Model output mean: {out.mean()}, std: {out.std()}")
print(f"Reference output mean: {ref_out.mean()}, std: {ref_out.std()}")