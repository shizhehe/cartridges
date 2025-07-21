import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask

from cartridges.models.qwen.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3Model,
    Qwen3ForCausalLM,
    repeat_kv,
    apply_rotary_pos_emb,
)
from cartridges.models.qwen.configuration_qwen3 import Qwen3Config


@pytest.fixture
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


@pytest.fixture
def sample_inputs():
    """Create sample inputs for testing."""
    batch_size = 2
    seq_len = 8
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "seq_ids": torch.arange(seq_len),
        "position_ids": torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1),
    }


class TestQwen3Attention:
    """Test the Qwen3Attention module with flex attention."""
    
    def test_attention_vs_padded_eager_attention(self, small_qwen_config, sample_inputs):
        """Test that flex attention gives similar results to padded eager attention."""
        torch.manual_seed(42)
        
        # Create attention layer
        attention = Qwen3Attention(small_qwen_config, layer_idx=0)
        attention.eval()
        
        batch_size = sample_inputs["batch_size"]
        seq_len = sample_inputs["seq_len"]
        hidden_size = small_qwen_config.hidden_size
        
        # Create sample hidden states
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Create position embeddings (mock cos/sin tensors)
        cos = torch.cos(torch.arange(seq_len).float().unsqueeze(-1) * torch.arange(attention.head_dim).float() / 10000)
        sin = torch.sin(torch.arange(seq_len).float().unsqueeze(-1) * torch.arange(attention.head_dim).float() / 10000)
        cos = cos.unsqueeze(0).repeat(batch_size, 1, 1)
        sin = sin.unsqueeze(0).repeat(batch_size, 1, 1)
        position_embeddings = (cos, sin)
        
        # Create block mask for flex attention (causal mask)
        def causal_mask_func(_, _h, q_idx, kv_idx):
            return q_idx >= kv_idx
        
        block_mask = create_block_mask(causal_mask_func, 1, 1, seq_len, seq_len, device=hidden_states.device)
        
        # Get flex attention output
        with torch.no_grad():
            flex_output, _ = attention(hidden_states, position_embeddings, block_mask)
        
        # Create eager attention baseline by manually implementing the attention
        with torch.no_grad():
            # Project to Q, K, V
            q_states = attention.q_norm(attention.q_proj(hidden_states).view(batch_size, seq_len, -1, attention.head_dim)).transpose(1, 2)
            k_states = attention.k_norm(attention.k_proj(hidden_states).view(batch_size, seq_len, -1, attention.head_dim)).transpose(1, 2)
            v_states = attention.v_proj(hidden_states).view(batch_size, seq_len, -1, attention.head_dim).transpose(1, 2)
            
            # Apply rotary embeddings
            q_states, k_states = apply_rotary_pos_emb(q_states, k_states, cos, sin)
            
            # Create causal mask for eager attention
            causal_mask = torch.full((seq_len, seq_len), float('-inf'), dtype=q_states.dtype, device=q_states.device)
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, attention.config.num_attention_heads, -1, -1)
            
            # Manual eager attention
            k_states_repeated = repeat_kv(k_states, attention.num_key_value_groups)
            v_states_repeated = repeat_kv(v_states, attention.num_key_value_groups)
            
            attn_weights = torch.matmul(q_states, k_states_repeated.transpose(2, 3)) * attention.scaling
            attn_weights = attn_weights + causal_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_states.dtype)
            
            eager_attn_output = torch.matmul(attn_weights, v_states_repeated)
            eager_attn_output = eager_attn_output.transpose(1, 2).contiguous()
            eager_attn_output = eager_attn_output.reshape(batch_size, seq_len, -1)
            eager_output = attention.o_proj(eager_attn_output)
        
        # Check that outputs are close (allowing for some numerical differences)
        torch.testing.assert_close(flex_output, eager_output, atol=1e-4, rtol=1e-3)
    
    def test_attention_output_shape(self, small_qwen_config, sample_inputs):
        """Test that attention output has correct shape."""
        attention = Qwen3Attention(small_qwen_config, layer_idx=0)
        
        batch_size = sample_inputs["batch_size"]
        seq_len = sample_inputs["seq_len"]
        hidden_size = small_qwen_config.hidden_size
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        cos = torch.randn(batch_size, seq_len, attention.head_dim)
        sin = torch.randn(batch_size, seq_len, attention.head_dim)
        position_embeddings = (cos, sin)
        
        def causal_mask_func(_, _h, q_idx, kv_idx):
            return q_idx >= kv_idx
        
        block_mask = create_block_mask(causal_mask_func, 1, 1, seq_len, seq_len, device=hidden_states.device)
        
        output, _ = attention(hidden_states, position_embeddings, block_mask)
        
        assert output.shape == (batch_size, seq_len, hidden_size)
