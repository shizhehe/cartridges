#!/usr/bin/env python3
"""
Analyze logprobs distributions from synthesize.py output.

Usage:
    python m07d06_analyze_logprobs.py

Example:
    python m07d06_analyze_logprobs.py --dataset_path /home/sabri/code-memory/outputs/2025-07-06-13-37-13-gmail_synthesis/gmail_synthesis-0/artifact/dataset.pkl
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional
import pandas as pd
from pydrantic import RunConfig
import pydrantic
from transformers import AutoTokenizer
def load_dataset(dataset_path: str):
    """Load the dataset from pickle file."""
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    return data['rows']

def extract_logprobs(training_examples):
    """Extract all logprobs from training examples."""
    all_logprobs = []
    example_stats = []
    all_topk_logprobs = []  # Store all top-k logprobs for each token
    
    for i, example in enumerate(training_examples):
        example_logprobs = []
        for message in example.messages:
            if message.top_logprobs is not None:
                # Extract the top logprobs (first column is the actual chosen token)
                chosen_logprobs = message.top_logprobs.logprobs[:, 0]
                example_logprobs.extend(chosen_logprobs.tolist())
                all_logprobs.extend(chosen_logprobs.tolist())
                
                # Store all top-k logprobs for probability mass analysis
                for token_idx in range(message.top_logprobs.logprobs.shape[0]):
                    token_topk_logprobs = message.top_logprobs.logprobs[token_idx, :]
                    all_topk_logprobs.append(token_topk_logprobs)
        
        if example_logprobs:
            example_stats.append({
                'example_idx': i,
                'num_tokens': len(example_logprobs),
                'mean_logprob': np.mean(example_logprobs),
                'std_logprob': np.std(example_logprobs),
                'min_logprob': np.min(example_logprobs),
                'max_logprob': np.max(example_logprobs),
                'median_logprob': np.median(example_logprobs)
            })
    
    return all_logprobs, example_stats, all_topk_logprobs

def analyze_logprobs_distribution(logprobs: List[float], example_stats: List[dict]):
    """Analyze the distribution of logprobs."""
    logprobs_array = np.array(logprobs)
    
    print("=== LOGPROBS DISTRIBUTION ANALYSIS ===")
    print(f"Total tokens with logprobs: {len(logprobs)}")
    print(f"Examples with logprobs: {len(example_stats)}")
    print()
    
    print("=== OVERALL STATISTICS ===")
    print(f"Mean logprob: {np.mean(logprobs_array):.4f}")
    print(f"Median logprob: {np.median(logprobs_array):.4f}")
    print(f"Std logprob: {np.std(logprobs_array):.4f}")
    print(f"Min logprob: {np.min(logprobs_array):.4f}")
    print(f"Max logprob: {np.max(logprobs_array):.4f}")
    print()
    
    print("=== PERCENTILES ===")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        print(f"{p:2d}th percentile: {np.percentile(logprobs_array, p):.4f}")
    print()
    
    # Convert to probabilities for additional analysis
    probs = np.exp(logprobs_array)
    print("=== PROBABILITY STATISTICS ===")
    print(f"Mean probability: {np.mean(probs):.4f}")
    print(f"Median probability: {np.median(probs):.4f}")
    print(f"Std probability: {np.std(probs):.4f}")
    print()
    
    # Analyze very low probability tokens
    very_low_prob_threshold = 0.01
    low_prob_tokens = probs < very_low_prob_threshold
    print(f"Tokens with prob < {very_low_prob_threshold}: {np.sum(low_prob_tokens)} ({100*np.mean(low_prob_tokens):.1f}%)")
    
    # Analyze high confidence tokens
    high_conf_threshold = 0.5
    high_conf_tokens = probs > high_conf_threshold
    print(f"Tokens with prob > {high_conf_threshold}: {np.sum(high_conf_tokens)} ({100*np.mean(high_conf_tokens):.1f}%)")
    print()
    
    return logprobs_array, probs

def analyze_topk_for_probability_mass(all_topk_logprobs: List[np.ndarray], target_mass: float = 0.97):
    """Analyze how many top-k tokens are needed to capture target probability mass."""
    topk_needed = []
    
    for token_logprobs in all_topk_logprobs:
        # Convert to probabilities and sort in descending order
        probs = np.exp(token_logprobs)
        probs_sorted = np.sort(probs)[::-1]  # Sort descending
        
        # Calculate cumulative probability mass
        cumulative_probs = np.cumsum(probs_sorted)
        
        # Find how many tokens needed to reach target mass
        k_needed = np.argmax(cumulative_probs >= target_mass) + 1
        
        # Handle edge case where we never reach target mass
        if cumulative_probs[-1] < target_mass:
            k_needed = len(probs_sorted)
        
        topk_needed.append(k_needed)
    
    return topk_needed

def visualize_topk_samples(training_examples, tokenizer_name: str = "Qwen/Qwen2.5-0.5B", num_samples: int = 3, output_dir: Optional[str] = None):
    """Visualize a few samples with top-k logprobs and decoded tokens."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer {tokenizer_name}: {e}")
        print("Falling back to Qwen/Qwen2.5-0.5B")
        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
        except Exception as e2:
            print(f"Error loading fallback tokenizer: {e2}")
            return
    
    samples_found = 0
    output_text = []
    
    for example_idx, example in enumerate(training_examples):
        if samples_found >= num_samples:
            break
            
        for msg_idx, message in enumerate(example.messages):
            if message.top_logprobs is not None and samples_found < num_samples:
                # Get a few interesting tokens (first 5 and some random ones)
                num_tokens = min(10, message.top_logprobs.logprobs.shape[0])
                token_indices = list(range(min(5, num_tokens)))
                if num_tokens > 5:
                    # Add some random tokens from the rest
                    import random
                    remaining_indices = list(range(5, message.top_logprobs.logprobs.shape[0]))
                    token_indices.extend(random.sample(remaining_indices, min(5, len(remaining_indices))))
                
                output_text.append(f"\n{'='*80}")
                output_text.append(f"SAMPLE {samples_found + 1} - Example {example_idx}, Message {msg_idx} ({message.role})")
                output_text.append(f"{'='*80}")
                output_text.append(f"Message content preview: {message.content[:200]}...")
                output_text.append(f"Total tokens in message: {message.top_logprobs.logprobs.shape[0]}")
                output_text.append(f"Top-k size: {message.top_logprobs.logprobs.shape[1]}")
                output_text.append("")
                
                for token_pos in token_indices:
                    token_logprobs = message.top_logprobs.logprobs[token_pos, :]
                    token_ids = message.top_logprobs.token_ids[token_pos, :]
                    
                    # Convert to probabilities
                    probs = np.exp(token_logprobs)
                    
                    # Sort by probability (descending)
                    sorted_indices = np.argsort(probs)[::-1]
                    
                    output_text.append(f"TOKEN POSITION {token_pos}:")
                    output_text.append(f"  Chosen token (rank 0): ID={token_ids[0]}, logprob={token_logprobs[0]:.4f}, prob={probs[0]:.4f}")
                    
                    # Decode the chosen token
                    try:
                        chosen_text = tokenizer.decode([token_ids[0]], skip_special_tokens=False)
                        output_text.append(f"  Chosen token text: '{chosen_text}'")
                    except Exception as e:
                        output_text.append(f"  Chosen token text: [decode error: {e}]")
                    
                    output_text.append(f"  Top-{min(5, len(token_ids))} alternatives:")
                    for rank in range(min(5, len(token_ids))):
                        idx = sorted_indices[rank]
                        token_id = token_ids[idx]
                        logprob = token_logprobs[idx]
                        prob = probs[idx]
                        
                        try:
                            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                            output_text.append(f"    Rank {rank}: ID={token_id}, logprob={logprob:.4f}, prob={prob:.4f}, text='{token_text}'")
                        except Exception as e:
                            output_text.append(f"    Rank {rank}: ID={token_id}, logprob={logprob:.4f}, prob={prob:.4f}, text=[decode error]")
                    
                    # Calculate cumulative probability mass
                    cumulative_prob = np.cumsum(np.sort(probs)[::-1])
                    tokens_for_97 = np.argmax(cumulative_prob >= 0.97) + 1
                    output_text.append(f"  Tokens needed for 97% mass: {tokens_for_97}")
                    output_text.append("")
                
                samples_found += 1
                break
    
    # Print to console
    for line in output_text:
        print(line)
    
    # Save to file if output_dir is specified
    if output_dir:
        output_path = Path(output_dir) / "topk_samples_visualization.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_text))
        print(f"\nSaved detailed token analysis to {output_path}")

def plot_distributions(logprobs_array: np.ndarray, probs: np.ndarray, topk_needed: List[int] = None, output_dir: Optional[str] = None):
    """Create distribution plots."""
    if topk_needed is not None:
        # Create 2x3 grid to include top-k histogram
        _, axes = plt.subplots(2, 3, figsize=(20, 10))
        
        # Top-k needed histogram
        axes[0, 2].hist(topk_needed, bins=min(50, max(topk_needed) - min(topk_needed) + 1), 
                       alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('Top-K Tokens Needed for 97% Probability Mass')
        axes[0, 2].set_xlabel('Number of Top-K Tokens')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].axvline(np.mean(topk_needed), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(topk_needed):.1f}')
        axes[0, 2].axvline(np.median(topk_needed), color='green', linestyle='--', 
                          label=f'Median: {np.median(topk_needed):.1f}')
        axes[0, 2].legend()
        
        # Box plot of top-k needed
        axes[1, 2].boxplot(topk_needed, vert=True)
        axes[1, 2].set_title('Box Plot of Top-K Tokens Needed')
        axes[1, 2].set_ylabel('Number of Tokens')
        
        # Add statistics text
        stats_text = f"""Statistics for Top-K Analysis:
Mean: {np.mean(topk_needed):.2f}
Median: {np.median(topk_needed):.2f}
Std: {np.std(topk_needed):.2f}
Min: {np.min(topk_needed)}
Max: {np.max(topk_needed)}
95th percentile: {np.percentile(topk_needed, 95):.0f}"""
        axes[1, 2].text(0.02, 0.98, stats_text, transform=axes[1, 2].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        _, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Logprobs histogram
    axes[0, 0].hist(logprobs_array, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Distribution of Log Probabilities')
    axes[0, 0].set_xlabel('Log Probability')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(logprobs_array), color='red', linestyle='--', label=f'Mean: {np.mean(logprobs_array):.3f}')
    axes[0, 0].legend()
    
    # Probabilities histogram
    axes[0, 1].hist(probs, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Distribution of Probabilities')
    axes[0, 1].set_xlabel('Probability')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(np.mean(probs), color='red', linestyle='--', label=f'Mean: {np.mean(probs):.3f}')
    axes[0, 1].legend()
    
    # Log-scale probability histogram
    axes[1, 0].hist(probs, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Distribution of Probabilities (Log Scale)')
    axes[1, 0].set_xlabel('Probability')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_yscale('log')
    
    # Box plot of logprobs
    axes[1, 1].boxplot(logprobs_array, vert=True)
    axes[1, 1].set_title('Box Plot of Log Probabilities')
    axes[1, 1].set_ylabel('Log Probability')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / "logprobs_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved distribution plot to {output_path}")
    else:
        plt.show()

def analyze_per_example_stats(example_stats: List[dict]):
    """Analyze per-example statistics."""
    if not example_stats:
        print("No examples with logprobs found.")
        return
    
    df = pd.DataFrame(example_stats)
    
    print("=== PER-EXAMPLE STATISTICS ===")
    print(f"Examples analyzed: {len(df)}")
    print()
    
    print("Token counts per example:")
    print(f"  Mean: {df['num_tokens'].mean():.1f}")
    print(f"  Median: {df['num_tokens'].median():.1f}")
    print(f"  Min: {df['num_tokens'].min()}")
    print(f"  Max: {df['num_tokens'].max()}")
    print()
    
    print("Mean logprob per example:")
    print(f"  Mean: {df['mean_logprob'].mean():.4f}")
    print(f"  Std: {df['mean_logprob'].std():.4f}")
    print(f"  Min: {df['mean_logprob'].min():.4f}")
    print(f"  Max: {df['mean_logprob'].max():.4f}")
    print()
    
    # Find examples with very low or high confidence
    low_conf_examples = df[df['mean_logprob'] < df['mean_logprob'].quantile(0.1)]
    high_conf_examples = df[df['mean_logprob'] > df['mean_logprob'].quantile(0.9)]
    
    print(f"Examples with lowest confidence (bottom 10%): {len(low_conf_examples)}")
    if len(low_conf_examples) > 0:
        print(f"  Mean logprob range: {low_conf_examples['mean_logprob'].min():.4f} to {low_conf_examples['mean_logprob'].max():.4f}")
    
    print(f"Examples with highest confidence (top 10%): {len(high_conf_examples)}")
    if len(high_conf_examples) > 0:
        print(f"  Mean logprob range: {high_conf_examples['mean_logprob'].min():.4f} to {high_conf_examples['mean_logprob'].max():.4f}")

class AnalyzeLogprobsConfig(RunConfig):
    """Configuration for analyzing logprobs distributions."""
    
    dataset_path: str
    save_plots: bool = True
    visualize_samples: bool = True
    tokenizer_name: str = "Qwen/Qwen3-4B"
    num_visualization_samples: int = 3
    
    def run(self):
        """Run the logprobs analysis."""
        # Load dataset
        print(f"Loading dataset from {self.dataset_path}")
        try:
            training_examples = load_dataset(self.dataset_path)
            print(f"Loaded {len(training_examples)} training examples")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return
        
        # Extract logprobs
        print("Extracting logprobs...")
        all_logprobs, example_stats, all_topk_logprobs = extract_logprobs(training_examples)
        
        if not all_logprobs:
            print("No logprobs found in the dataset!")
            return
        
        # Analyze distributions
        logprobs_array, probs = analyze_logprobs_distribution(all_logprobs, example_stats)
        
        # Per-example analysis
        analyze_per_example_stats(example_stats)
        
        # Analyze top-k for probability mass
        topk_needed = None
        if all_topk_logprobs:
            print("\nAnalyzing top-k tokens needed for 97% probability mass...")
            topk_needed = analyze_topk_for_probability_mass(all_topk_logprobs, target_mass=0.97)
            
            print(f"Top-K Analysis (97% probability mass):")
            print(f"  Mean tokens needed: {np.mean(topk_needed):.2f}")
            print(f"  Median tokens needed: {np.median(topk_needed):.0f}")
            print(f"  95th percentile: {np.percentile(topk_needed, 95):.0f}")
            print(f"  Max tokens needed: {np.max(topk_needed)}")
        
        # Visualize samples with decoded tokens
        if self.visualize_samples:
            print("\nVisualizing sample tokens with top-k logprobs...")
            output_dir = self.run_dir if self.run_dir else "."
            visualize_topk_samples(training_examples, 
                                 tokenizer_name=self.tokenizer_name,
                                 num_samples=self.num_visualization_samples,
                                 output_dir=output_dir)
        
        # Create plots
        if self.save_plots:
            print("\nCreating distribution plots...")
            output_dir = self.run_dir if self.run_dir else "."
            plot_distributions(logprobs_array, probs, topk_needed, output_dir)
        
        print("\nAnalysis complete!")

if __name__ == "__main__":
    pydrantic.main(AnalyzeLogprobsConfig(
        dataset_path="/home/sabri/code-memory/outputs/2025-07-06-13-37-13-gmail_synthesis/gmail_synthesis-0/artifact/dataset.pkl",
        output_dir=os.environ["CODEMEM_OUTPUT_DIR"],
    ))