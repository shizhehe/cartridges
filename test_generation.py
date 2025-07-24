#!/usr/bin/env python3

import torch
from cartridges.generation import PackedCache

def test_packed_cache():
    """Test PackedCache basic functionality"""
    cache = PackedCache()
    
    # Test initialization
    assert cache._seq_ids is None
    print("✓ PackedCache initialization works")
    
    # Test seq_ids method
    seq_ids = torch.tensor([0, 0, 1, 1, 2])
    cache._seq_ids = seq_ids
    assert torch.equal(cache.seq_ids(), seq_ids)
    print("✓ PackedCache seq_ids method works")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_packed_cache()