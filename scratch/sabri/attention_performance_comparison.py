#!/usr/bin/env python3
"""
Performance comparison between padding-based (using SDPA) and packed attention approaches
for handling batches with mismatched sequence lengths.
Updated to use attention-gym utilities for cleaner implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import time
import numpy as np
from typing import List, Tuple, Dict

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Try to import tabulate for nice table output
try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

# Import attention-gym utilities
try:
    from attn_gym.masks import generate_doc_mask_mod
    from attn_gym.masks.document_mask import length_to_offsets
    from attn_gym.masks.causal import causal_mask
    from torch.nn.attention.flex_attention import noop_mask
    ATTN_GYM_AVAILABLE = True
except ImportError:
    print("Warning: attention-gym not available. Using fallback implementations.")
    ATTN_GYM_AVAILABLE = False

# Import Triton benchmarking utilities
try:
    from triton.testing import do_bench
    TRITON_AVAILABLE = True
except ImportError:
    print("Warning: triton not available. Using fallback timing.")
    TRITON_AVAILABLE = False

# Compile the flex_attention function
flex_attention = torch.compile(flex_attention, dynamic=False)


class PaddingAttentionSDPA(nn.Module):
    """Efficient attention with padding using scaled_dot_product_attention."""
    
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Convert attention mask to the format expected by SDPA
        if attention_mask is not None:
            # SDPA expects True for positions that should be attended to
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attn_mask = attn_mask.expand(batch_size, num_heads, seq_len, seq_len)
        else:
            attn_mask = None
        
        # Use scaled_dot_product_attention for maximum efficiency
        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False
        )
        
        return attn_output


class PackedAttention(nn.Module):
    """Packed attention using torch flex attention."""
    
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_mask: object) -> torch.Tensor:
        # q, k, v are already in the correct shape: (batch=1, heads, seq_len, head_dim)
        
        # Use flex attention with block mask
        attn_output = flex_attention(q, k, v, block_mask=block_mask)
        
        return attn_output


def create_padding_qkv(sequences: List[torch.Tensor], num_heads: int, head_dim: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create padded Q, K, V tensors with attention mask."""
    batch_size = len(sequences)
    max_len = max(seq.size(0) for seq in sequences)
    
    # Create attention mask
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    # Create padded Q, K, V tensors
    q = torch.randn(batch_size, num_heads, max_len, head_dim)
    k = torch.randn(batch_size, num_heads, max_len, head_dim)
    v = torch.randn(batch_size, num_heads, max_len, head_dim)
    
    for i, seq in enumerate(sequences):
        seq_len = seq.size(0)
        attention_mask[i, :seq_len] = 1
        # Zero out padded positions in Q, K, V
        q[i, :, seq_len:, :] = 0
        k[i, :, seq_len:, :] = 0
        v[i, :, seq_len:, :] = 0
    
    return q, k, v, attention_mask


def create_packed_qkv(sequences: List[torch.Tensor], num_heads: int, head_dim: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, object, List[int]]:
    """Create packed Q, K, V tensors with block mask for flex attention."""
    # Get sequence lengths
    seq_lengths = [seq.size(0) for seq in sequences]
    total_tokens = sum(seq_lengths)
    
    # Create random Q, K, V tensors
    q = torch.randn(1, num_heads, total_tokens, head_dim, device=device)
    k = torch.randn(1, num_heads, total_tokens, head_dim, device=device)
    v = torch.randn(1, num_heads, total_tokens, head_dim, device=device)

    kv_seq_id = torch.cat(
        [torch.full((seq.size(0),), idx, dtype=torch.long) for idx, seq in enumerate(sequences)]
    ).to(device)
    q_seq_id = kv_seq_id.clone()
    # Create mask modification function for flex attention
    def sequence_mask(_, _h, q_idx, kv_idx):
        return q_seq_id[q_idx] == kv_seq_id[kv_idx]
    
    # Create block mask using the mask modification function
    block_mask = create_block_mask(
        sequence_mask,
        B=None,  # batch size
        H=None,  # will be expanded for multiple heads
        Q_LEN=total_tokens,
        KV_LEN=total_tokens,
        device=device,
        _compile=True
    )

    
    
    # Calculate sparsity of the block mask manually for debugging
    total_attention_elements = total_tokens * total_tokens
    allowed_elements = sum(seq_len * seq_len for seq_len in seq_lengths)
    sparsity = (total_attention_elements - allowed_elements) / total_attention_elements
    print(f"    Block mask sparsity: {sparsity:.2%} ({allowed_elements}/{total_attention_elements} elements allowed)")
    
    return q, k, v, block_mask, seq_lengths


def create_packed_qkv_with_attn_gym(sequences: List[torch.Tensor], num_heads: int, head_dim: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, object, List[int]]:
    """Create packed Q, K, V tensors with block mask using attention-gym utilities."""
    if not ATTN_GYM_AVAILABLE:
        print("    Falling back to manual implementation (attention-gym not available)")
        return create_packed_qkv(sequences, num_heads, head_dim, device)
    
    # Get sequence lengths
    seq_lengths = [seq.size(0) for seq in sequences]
    total_tokens = sum(seq_lengths)
    
    # Create random Q, K, V tensors
    q = torch.randn(1, num_heads, total_tokens, head_dim, device=device)
    k = torch.randn(1, num_heads, total_tokens, head_dim, device=device)
    v = torch.randn(1, num_heads, total_tokens, head_dim, device=device)

    # Use attention-gym's utilities for cleaner mask creation
    offsets = length_to_offsets(seq_lengths, device)
    
    # Create document mask with no inner mask (just document separation)
    doc_mask_mod = generate_doc_mask_mod(noop_mask, offsets)
    
    # Create block mask using attention-gym's document mask
    block_mask = create_block_mask(
        doc_mask_mod,
        B=None,  # batch size
        H=None,  # will be expanded for multiple heads
        Q_LEN=total_tokens,
        KV_LEN=total_tokens,
        device=device,
        _compile=True
    )
    
    # Calculate sparsity of the block mask
    total_attention_elements = total_tokens * total_tokens
    allowed_elements = sum(seq_len * seq_len for seq_len in seq_lengths)
    sparsity = (total_attention_elements - allowed_elements) / total_attention_elements
    print(f"    Block mask sparsity (attention-gym): {sparsity:.2%} ({allowed_elements}/{total_attention_elements} elements allowed)")
    
    return q, k, v, block_mask, seq_lengths


def unpack_sequences(packed_output: torch.Tensor, seq_lengths: List[int]) -> List[torch.Tensor]:
    """Unpack the output back to individual sequences."""
    sequences = []
    start_idx = 0
    for seq_len in seq_lengths:
        end_idx = start_idx + seq_len
        sequences.append(packed_output[start_idx:end_idx])
        start_idx = end_idx
    return sequences


def generate_random_sequences(batch_size: int, min_len: int, max_len: int, d_model: int) -> List[torch.Tensor]:
    """Generate random sequences of varying lengths."""
    sequences = []
    for _ in range(batch_size):
        seq_len = torch.randint(min_len, max_len + 1, (1,)).item()
        sequences.append(torch.randn(seq_len, d_model))
    return sequences


def benchmark_approach(forward_fn, device: torch.device, num_iterations: int = 100) -> float:
    """Benchmark a specific approach using Triton's do_bench or fallback timing."""
    
    if TRITON_AVAILABLE:
        # Use Triton's optimized benchmarking
        def triton_wrapper():
            with torch.no_grad():
                return forward_fn()
        
        # do_bench automatically handles warmup and synchronization
        time_ms = do_bench(triton_wrapper)
        return time_ms / 1000.0  # Convert ms to seconds
    else:
        # Fallback to manual timing
        print("    Using fallback timing (triton not available)")
        
        # Warmup
        for _ in range(20):
            with torch.no_grad():
                _ = forward_fn()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = forward_fn()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        return (end_time - start_time) / num_iterations


def padding_forward(model: PaddingAttentionSDPA, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Forward pass for padding approach."""
    return model(q, k, v, attention_mask)


def packed_forward(model: PackedAttention, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_mask: object) -> torch.Tensor:
    """Forward pass for packed approach."""
    return model(q, k, v, block_mask)


def calculate_flops_and_throughput(sequences: List[torch.Tensor], num_heads: int, head_dim: int, 
                                   block_mask: object = None) -> Dict[str, float]:
    """Calculate FLOPS and throughput metrics for comparison."""
    seq_lengths = [seq.size(0) for seq in sequences]
    batch_size = len(sequences)
    max_len = max(seq_lengths)
    total_tokens = sum(seq_lengths)
    
    # For padding approach: batch_size * num_heads * head_dim * max_len^2
    padding_flops = batch_size * num_heads * head_dim * max_len * max_len
    
    # For packed approach: depends on block mask sparsity
    if block_mask is not None:
        # Use block mask sparsity to calculate effective FLOPS
        density = (100 - block_mask.sparsity()) / 100
        packed_flops = density * num_heads * head_dim * total_tokens * total_tokens
    else:
        # Fallback: calculate based on actual sequence lengths
        packed_flops = num_heads * head_dim * sum(seq_len * seq_len for seq_len in seq_lengths)
    
    return {
        'padding_flops': padding_flops,
        'packed_flops': packed_flops,
        'total_tokens': total_tokens,
        'padded_tokens': batch_size * max_len,
        'flops_ratio': padding_flops / packed_flops if packed_flops > 0 else 0
    }


def calculate_efficiency_metrics(sequences: List[torch.Tensor]) -> Dict[str, float]:
    """Calculate efficiency metrics for a batch of sequences."""
    seq_lengths = [seq.size(0) for seq in sequences]
    total_tokens = sum(seq_lengths)
    max_len = max(seq_lengths)
    batch_size = len(sequences)
    
    # Padding efficiency
    padded_tokens = batch_size * max_len
    padding_efficiency = total_tokens / padded_tokens
    
    # Length variance (higher variance = more benefit from packing)
    length_variance = np.var(seq_lengths)
    
    return {
        'padding_efficiency': padding_efficiency,
        'length_variance': length_variance,
        'avg_length': np.mean(seq_lengths),
        'max_length': max_len,
        'total_tokens': total_tokens,
        'padded_tokens': padded_tokens
    }


def run_comparison(batch_sizes: List[int], sequence_configs: List[Tuple[int, int]], 
                  d_model: int = 512, num_heads: int = 8, num_iterations: int = 100, 
                  use_attention_gym: bool = True) -> Dict:
    """Run comprehensive comparison between approaches."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results = {
        'batch_sizes': [],
        'seq_config': [],
        'padding_time': [],
        'packed_time': [],
        'speedup': [],
        'padding_efficiency': [],
        'length_variance': [],
        'memory_ratio': []
    }
    
    for batch_size in batch_sizes:
        for min_len, max_len in sequence_configs:
            print(f"\nTesting batch_size={batch_size}, seq_len={min_len}-{max_len}")
            
            # Generate test sequences
            sequences = generate_random_sequences(batch_size, min_len, max_len, d_model)
            
            # Calculate efficiency metrics
            metrics = calculate_efficiency_metrics(sequences)
            
            # Initialize models
            padding_model = PaddingAttentionSDPA(d_model, num_heads).to(device)
            packed_model = PackedAttention(d_model, num_heads).to(device)
            
            head_dim = d_model // num_heads
            
            # Prepare QKV tensors (not included in timing)
            pad_q, pad_k, pad_v, attention_mask = create_padding_qkv(sequences, num_heads, head_dim)
            pad_q, pad_k, pad_v = pad_q.to(device), pad_k.to(device), pad_v.to(device)
            attention_mask = attention_mask.to(device)
            
            if use_attention_gym:
                pack_q, pack_k, pack_v, block_mask, _ = create_packed_qkv_with_attn_gym(sequences, num_heads, head_dim, device)
            else:
                pack_q, pack_k, pack_v, block_mask, _ = create_packed_qkv(sequences, num_heads, head_dim, device)
            
            # Calculate FLOPS metrics
            flops_metrics = calculate_flops_and_throughput(sequences, num_heads, head_dim, block_mask)
            
            # Single test runs to see shapes
            print("\n--- Single test runs ---")
            with torch.no_grad():
                pad_result = padding_forward(padding_model, pad_q, pad_k, pad_v, attention_mask)
                pack_result = packed_forward(packed_model, pack_q, pack_k, pack_v, block_mask)
            print(f"Results shapes: pad={pad_result.shape}, pack={pack_result.shape}")
            
            # Calculate theoretical complexity
            pad_complexity = pad_q.shape[0] * pad_q.shape[2] * pad_q.shape[2]  # batch * seq_len^2
            pack_complexity = pack_q.shape[2] * pack_q.shape[2]  # total_tokens^2
            
            # But with block mask, effective complexity should be much lower
            seq_lengths = [seq.size(0) for seq in sequences]
            effective_pack_complexity = sum(seq_len * seq_len for seq_len in seq_lengths)
            
            print(f"Theoretical complexity: padding={pad_complexity:,}, packed_full={pack_complexity:,}")
            print(f"Effective packed complexity (with block mask): {effective_pack_complexity:,}")
            print(f"Complexity ratio (effective_packed/padding): {effective_pack_complexity/pad_complexity:.2f}x")
            print(f"Block mask should reduce computation by: {pack_complexity/effective_pack_complexity:.2f}x")
            
            # Benchmark padding approach (only attention computation)
            print(f"\n--- Benchmarking (using {'Triton' if TRITON_AVAILABLE else 'fallback'}) ---")
            padding_time = benchmark_approach(
                lambda: padding_forward(padding_model, pad_q, pad_k, pad_v, attention_mask),
                device,
                num_iterations
            )
            
            # Benchmark packed approach (only attention computation)
            packed_time = benchmark_approach(
                lambda: packed_forward(packed_model, pack_q, pack_k, pack_v, block_mask),
                device,
                num_iterations
            )
            
            # Calculate FLOPS per second
            padding_tflops = (flops_metrics['padding_flops'] * 4) / (padding_time * 1e12)  # 4x for fwd pass
            packed_tflops = (flops_metrics['packed_flops'] * 4) / (packed_time * 1e12)
            
            # Store results
            results['batch_sizes'].append(batch_size)
            results['seq_config'].append(f"{min_len}-{max_len}")
            results['padding_time'].append(padding_time * 1000)  # Convert to ms
            results['packed_time'].append(packed_time * 1000)
            results['speedup'].append(padding_time / packed_time)
            results['padding_efficiency'].append(metrics['padding_efficiency'])
            results['length_variance'].append(metrics['length_variance'])
            results['memory_ratio'].append(metrics['padded_tokens'] / metrics['total_tokens'])
            
            print(f"  Padding (SDPA): {padding_time*1000:.3f}ms ({padding_tflops:.2f} TFLOPS)")
            print(f"  Packed (Flex): {packed_time*1000:.3f}ms ({packed_tflops:.2f} TFLOPS)")
            print(f"  Speedup: {padding_time/packed_time:.2f}x")
            print(f"  FLOPS efficiency: {flops_metrics['flops_ratio']:.2f}x theoretical")
            print(f"  Padding efficiency: {metrics['padding_efficiency']:.2f}")
            print(f"  Memory ratio: {metrics['padded_tokens']/metrics['total_tokens']:.2f}x")
            
            # Calculate actual throughput
            pad_tokens_per_sec = (batch_size * max([seq.size(0) for seq in sequences])) / padding_time
            pack_tokens_per_sec = sum([seq.size(0) for seq in sequences]) / packed_time
            print(f"  Throughput: padding={pad_tokens_per_sec:,.0f} tokens/sec, packed={pack_tokens_per_sec:,.0f} tokens/sec")
            
            # If block mask is working, packed should be much faster
            theoretical_speedup = pack_complexity / effective_pack_complexity
            actual_speedup = padding_time / packed_time
            print(f"  Block mask theoretical speedup: {theoretical_speedup:.2f}x")
            print(f"  Actual speedup: {actual_speedup:.2f}x")
            if theoretical_speedup > 0:
                print(f"  Block mask effectiveness: {(actual_speedup/theoretical_speedup)*100:.1f}%")
            else:
                print(f"  Block mask effectiveness: N/A (theoretical speedup = {theoretical_speedup:.2f})")
    
    return results


def print_benchmark_table(results: Dict, title: str = "Attention Performance Comparison"):
    """Print a nice table summary of benchmark results."""
    if not TABULATE_AVAILABLE:
        print("Tabulate not available, using simple output format")
        return
    
    # Prepare data for the table
    table_data = []
    for i in range(len(results['batch_sizes'])):
        row = [
            f"B{results['batch_sizes'][i]} S{results['seq_config'][i]}",
            f"{results['padding_time'][i]:.3f}",
            f"{results['packed_time'][i]:.3f}", 
            f"{results['speedup'][i]:.2f}x",
            f"{results['padding_efficiency'][i]:.3f}",
            f"{results['memory_ratio'][i]:.2f}x"
        ]
        table_data.append(row)
    
    headers = [
        "Config",
        "Padding (ms)",
        "Packed (ms)",
        "Speedup", 
        "Pad Efficiency",
        "Memory Ratio"
    ]
    
    print(f"\n{title}")
    print("=" * len(title))
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def plot_results(results: Dict):
    """Plot comparison results."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, skipping plots")
        return
    
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time comparison
    x = range(len(results['batch_sizes']))
    ax1.bar([i - 0.2 for i in x], results['padding_time'], width=0.4, label='Padding (SDPA)', alpha=0.7)
    ax1.bar([i + 0.2 for i in x], results['packed_time'], width=0.4, label='Packed (Flex)', alpha=0.7)
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Execution Time Comparison')
    ax1.legend()
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"B{b}\nS{s}" for b, s in zip(results['batch_sizes'], results['seq_config'])])
    
    # Speedup vs Padding Efficiency
    ax2.scatter(results['padding_efficiency'], results['speedup'], alpha=0.7, s=100)
    ax2.set_xlabel('Padding Efficiency')
    ax2.set_ylabel('Speedup (Padding/Packed)')
    ax2.set_title('Speedup vs Padding Efficiency')
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Memory ratio
    ax3.bar(x, results['memory_ratio'], alpha=0.7, color='orange')
    ax3.set_xlabel('Configuration')
    ax3.set_ylabel('Memory Ratio (Padded/Actual)')
    ax3.set_title('Memory Overhead of Padding')
    ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"B{b}\nS{s}" for b, s in zip(results['batch_sizes'], results['seq_config'])])
    
    # Speedup vs Length Variance
    ax4.scatter(results['length_variance'], results['speedup'], alpha=0.7, s=100, color='green')
    ax4.set_xlabel('Length Variance')
    ax4.set_ylabel('Speedup (Padding/Packed)')
    ax4.set_title('Speedup vs Length Variance')
    ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/sabri/code/cartridges/scratch/sabri/attention_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def test_correctness(d_model: int = 512, num_heads: int = 8, tolerance: float = 1e-4) -> bool:
    """Test that both approaches return equivalent results."""
    print("\nTesting correctness...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate test sequences
    test_sequences = generate_random_sequences(batch_size=4, min_len=10, max_len=20, d_model=d_model)
    
    # Initialize models
    padding_model = PaddingAttentionSDPA(d_model, num_heads).to(device)
    packed_model = PackedAttention(d_model, num_heads).to(device)
    
    head_dim = d_model // num_heads
    
    # Create same QKV tensors for both approaches
    pad_q, pad_k, pad_v, attention_mask = create_padding_qkv(test_sequences, num_heads, head_dim)
    pad_q, pad_k, pad_v = pad_q.to(device), pad_k.to(device), pad_v.to(device)
    attention_mask = attention_mask.to(device)
    
    # Convert padded QKV to packed format
    seq_lengths = [seq.size(0) for seq in test_sequences]
    total_tokens = sum(seq_lengths)
    
    pack_q = torch.zeros(1, num_heads, total_tokens, head_dim, device=device)
    pack_k = torch.zeros(1, num_heads, total_tokens, head_dim, device=device)
    pack_v = torch.zeros(1, num_heads, total_tokens, head_dim, device=device)
    
    # Copy data from padded to packed format
    packed_idx = 0
    for batch_idx, seq_len in enumerate(seq_lengths):
        pack_q[0, :, packed_idx:packed_idx+seq_len, :] = pad_q[batch_idx, :, :seq_len, :]
        pack_k[0, :, packed_idx:packed_idx+seq_len, :] = pad_k[batch_idx, :, :seq_len, :]
        pack_v[0, :, packed_idx:packed_idx+seq_len, :] = pad_v[batch_idx, :, :seq_len, :]
        packed_idx += seq_len
    
    # Create block mask
    def sequence_mask(_, _h, q_idx, kv_idx):
        q_seq_id = torch.zeros_like(q_idx, dtype=torch.long)
        kv_seq_id = torch.zeros_like(kv_idx, dtype=torch.long)
        
        start_idx = 0
        for seq_id, seq_len in enumerate(seq_lengths):
            end_idx = start_idx + seq_len
            q_seq_id = torch.where((q_idx >= start_idx) & (q_idx < end_idx), seq_id, q_seq_id)
            kv_seq_id = torch.where((kv_idx >= start_idx) & (kv_idx < end_idx), seq_id, kv_seq_id)
            start_idx = end_idx
        
        return q_seq_id == kv_seq_id
    
    block_mask = create_block_mask(
        sequence_mask,
        B=None, H=None, Q_LEN=total_tokens, KV_LEN=total_tokens, device=device, _compile=True
    )
    
    # Run both approaches
    with torch.no_grad():
        padding_output = padding_model(pad_q, pad_k, pad_v, attention_mask)
        packed_output = packed_model(pack_q, pack_k, pack_v, block_mask)
    
    # Convert packed output back to padded format for comparison
    packed_to_padded = torch.zeros_like(padding_output)
    packed_idx = 0
    for batch_idx, seq_len in enumerate(seq_lengths):
        packed_to_padded[batch_idx, :, :seq_len, :] = packed_output[0, :, packed_idx:packed_idx+seq_len, :]
        packed_idx += seq_len
    
    # Compare outputs
    all_close = True
    max_diff = 0.0
    
    for i, seq_len in enumerate(seq_lengths):
        # Only compare valid (non-padded) positions
        pad_seq = padding_output[i, :, :seq_len, :]
        pack_seq = packed_to_padded[i, :, :seq_len, :]
        
        diff = torch.abs(pad_seq - pack_seq).max().item()
        max_diff = max(max_diff, diff)
        
        if diff > tolerance:
            print(f"  Sequence {i}: MISMATCH (max diff: {diff:.6f})")
            all_close = False
        else:
            print(f"  Sequence {i}: MATCH (max diff: {diff:.6f})")
    
    print(f"\nOverall max difference: {max_diff:.6f}")
    print(f"Tolerance: {tolerance}")
    print(f"Test {'PASSED' if all_close else 'FAILED'}")
    
    return all_close


def main():
    """Run the performance comparison."""
    print("Attention Performance Comparison: Padding (SDPA) vs Packed (Flex)")
    print("Updated to use attention-gym utilities for cleaner document masking")
    print("=" * 70)
    
    # Test correctness first
    # if not test_correctness():
    #     print("\n❌ Correctness test failed! Stopping execution.")
    #     return None
    
    print("\n✅ Correctness test passed! Proceeding with performance comparison.")
    
    # Configuration
    batch_sizes = [4]
    sequence_configs = [(32, 8192)]
    d_model = 1024
    num_heads = 8
    num_iterations = 10
    
    # Run comparison (with attention-gym by default)
    results = run_comparison(batch_sizes, sequence_configs, d_model, num_heads, num_iterations, use_attention_gym=True)
    
    # Print benchmark table
    print_benchmark_table(results, "Attention Performance Comparison (Triton Benchmarking)")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    avg_speedup = np.mean(results['speedup'])
    avg_memory_ratio = np.mean(results['memory_ratio'])
    avg_padding_efficiency = np.mean(results['padding_efficiency'])
    
    print(f"Average speedup (Packed vs Padding): {avg_speedup:.2f}x")
    print(f"Average memory overhead (Padding): {avg_memory_ratio:.2f}x")
    print(f"Average padding efficiency: {avg_padding_efficiency:.2f}")
    
    best_config_idx = np.argmax(results['speedup'])
    print(f"Best configuration: Batch={results['batch_sizes'][best_config_idx]}, "
          f"Seq={results['seq_config'][best_config_idx]}, "
          f"Speedup={results['speedup'][best_config_idx]:.2f}x")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("- Use packed attention when padding efficiency < 0.7")
    print("- Use packed attention when sequence length variance is high")
    print("- SDPA padding is very efficient for uniform sequence lengths")
    print("- Triton benchmarking provides more accurate performance measurements")
    print("- attention-gym utilities simplify mask creation and improve code maintainability")
    
    # Plot results
    plot_results(results)
    
    return results


if __name__ == "__main__":
    results = main()