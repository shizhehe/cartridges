#!/usr/bin/env python3

import torch
from cartridges.generation import PackedCache

def test_packed_cache():
    """Test PackedCache functionality with sequence contiguity"""
    cache = PackedCache()
    
    # Test initialization
    assert cache._seq_ids is None
    print("✓ PackedCache initialization works")
    
    # Test seq_ids setting and concatenation
    seq_ids_1 = torch.tensor([0, 0, 1])  # First batch: seq 0 (2 tokens), seq 1 (1 token)
    seq_ids_2 = torch.tensor([0, 1, 1])  # Second batch: seq 0 (1 token), seq 1 (2 tokens)
    
    cache.set_seq_ids(seq_ids_1)
    assert torch.equal(cache.seq_ids(), seq_ids_1)
    print("✓ Initial seq_ids setting works")
    
    cache.set_seq_ids(seq_ids_2)
    expected = torch.tensor([0, 0, 1, 0, 1, 1])  # Concatenated
    assert torch.equal(cache.seq_ids(), expected)
    print("✓ Seq_ids concatenation works")
    
    # Test cache update with dummy keys/values
    batch_size, num_heads, seq_len, head_dim = 1, 8, 3, 64
    keys_1 = torch.randn(batch_size, num_heads, seq_len, head_dim)
    values_1 = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # First update
    returned_keys, returned_values = cache.update(keys_1, values_1, layer_idx=0, cache_kwargs={})
    assert returned_keys.shape == (1, 8, 3, 64)
    assert cache.get_seq_length() == 3
    print("✓ First cache update works")
    
    # Second update - should concatenate
    keys_2 = torch.randn(batch_size, num_heads, seq_len, head_dim)
    values_2 = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    returned_keys, returned_values = cache.update(keys_2, values_2, layer_idx=0, cache_kwargs={})
    assert returned_keys.shape == (1, 8, 6, 64)  # Concatenated along seq dimension
    assert cache.get_seq_length() == 6
    print("✓ Cache concatenation works")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_packed_cache()