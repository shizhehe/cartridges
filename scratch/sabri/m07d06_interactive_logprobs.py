#!/usr/bin/env python3
"""
Interactive terminal program for exploring logprobs in synthesize.py output.

Usage:
    python m07d06_interactive_logprobs.py --dataset_path /path/to/dataset.pkl
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from pydrantic import RunConfig
import pydrantic
from transformers import AutoTokenizer
import textwrap
import random


def load_dataset(dataset_path: str):
    """Load the dataset from pickle file."""
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    return data['rows']


class InteractiveLogprobsExplorer:
    def __init__(self, training_examples, tokenizer_name: str = "Qwen/Qwen2.5-0.5B"):
        self.training_examples = training_examples
        self.current_example_idx = 0
        self.current_message_idx = 0
        self.current_token_idx = 0
        self.current_page = 0
        self.tokens_per_page = 5
        
        # Filter examples that have logprobs
        self.examples_with_logprobs = []
        for idx, example in enumerate(training_examples):
            for msg_idx, message in enumerate(example.messages):
                if message.top_logprobs is not None:
                    self.examples_with_logprobs.append((idx, msg_idx))
        
        if not self.examples_with_logprobs:
            raise ValueError("No examples with logprobs found in dataset!")
        
        self.current_pos = 0  # Position in examples_with_logprobs
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            print(f"✓ Loaded tokenizer: {tokenizer_name}")
        except Exception as e:
            print(f"⚠ Error loading tokenizer {tokenizer_name}: {e}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
                print("✓ Loaded fallback tokenizer: Qwen/Qwen2.5-0.5B")
            except Exception as e2:
                print(f"✗ Error loading fallback tokenizer: {e2}")
                self.tokenizer = None
        
        print(f"Found {len(self.examples_with_logprobs)} messages with logprobs across {len(training_examples)} examples")
    
    def get_current_example_and_message(self) -> Tuple[int, int, object, object]:
        """Get current example and message."""
        if not self.examples_with_logprobs:
            return None, None, None, None
        
        example_idx, message_idx = self.examples_with_logprobs[self.current_pos]
        example = self.training_examples[example_idx]
        message = example.messages[message_idx]
        return example_idx, message_idx, example, message
    
    def decode_token(self, token_id: int) -> str:
        """Decode a single token ID to text."""
        if self.tokenizer is None:
            return f"[ID:{token_id}]"
        
        try:
            return self.tokenizer.decode([token_id], skip_special_tokens=False)
        except Exception as e:
            return f"[ID:{token_id},ERR:{str(e)[:20]}]"
    
    def show_current_context(self):
        """Show context about current example and message."""
        example_idx, message_idx, example, message = self.get_current_example_and_message()
        if example is None:
            print("No examples available!")
            return
        
        print(f"\n{'='*80}")
        print(f"EXAMPLE {example_idx + 1}/{len(self.training_examples)} | MESSAGE {message_idx + 1}/{len(example.messages)} | POS {self.current_pos + 1}/{len(self.examples_with_logprobs)}")
        print(f"{'='*80}")
        
        print(f"Example Type: {example.type}")
        print(f"Message Role: {message.role}")
        print(f"Message Length: {len(message.content)} chars")
        
        if message.top_logprobs is not None:
            print(f"Tokens with logprobs: {message.top_logprobs.logprobs.shape[0]}")
            print(f"Top-k size: {message.top_logprobs.logprobs.shape[1]}")
        
        print(f"\nMessage Content:")
        print("-" * 40)
        wrapped_content = textwrap.fill(message.content, width=80)
        print(wrapped_content[:500] + ("..." if len(wrapped_content) > 500 else ""))
        print("-" * 40)
    
    def show_token_details(self, token_idx: int):
        """Show detailed logprobs for a specific token."""
        example_idx, message_idx, example, message = self.get_current_example_and_message()
        if message is None or message.top_logprobs is None:
            print("No logprobs available!")
            return
        
        if token_idx >= message.top_logprobs.logprobs.shape[0]:
            print(f"Token index {token_idx} out of range (max: {message.top_logprobs.logprobs.shape[0] - 1})")
            return
        
        token_logprobs = message.top_logprobs.logprobs[token_idx, :]
        token_ids = message.top_logprobs.token_ids[token_idx, :]
        
        # Convert to probabilities and sort
        probs = np.exp(token_logprobs)
        sorted_indices = np.argsort(probs)[::-1]
        
        print(f"\n🔍 TOKEN {token_idx + 1}/{message.top_logprobs.logprobs.shape[0]}")
        print(f"{'='*60}")
        
        print(f"Chosen token (rank 0):")
        chosen_text = self.decode_token(token_ids[0])
        print(f"  Text: '{chosen_text}'")
        print(f"  ID: {token_ids[0]}")
        print(f"  Logprob: {token_logprobs[0]:.4f}")
        print(f"  Probability: {probs[0]:.4f}")
        
        print(f"\nTop-10 alternatives:")
        for rank in range(min(10, len(token_ids))):
            idx = sorted_indices[rank]
            token_id = token_ids[idx]
            logprob = token_logprobs[idx]
            prob = probs[idx]
            token_text = self.decode_token(token_id)
            
            marker = "→" if rank == 0 else " "
            print(f"  {marker} {rank:2d}: '{token_text:>15}' | ID:{token_id:>6} | logprob:{logprob:>7.3f} | prob:{prob:>7.4f}")
        
        # Calculate cumulative probability mass
        cumulative_prob = np.cumsum(np.sort(probs)[::-1])
        tokens_for_95 = np.argmax(cumulative_prob >= 0.95) + 1
        tokens_for_97 = np.argmax(cumulative_prob >= 0.97) + 1
        tokens_for_99 = np.argmax(cumulative_prob >= 0.99) + 1
        
        print(f"\nProbability mass analysis:")
        print(f"  Tokens for 95% mass: {tokens_for_95}")
        print(f"  Tokens for 97% mass: {tokens_for_97}")
        print(f"  Tokens for 99% mass: {tokens_for_99}")
        print(f"  Total entropy: {-np.sum(probs * np.log(probs + 1e-10)):.3f}")
    
    def show_token_page(self, page_num: int = None):
        """Show a page of tokens with their logprobs."""
        _, _, _, message = self.get_current_example_and_message()
        if message is None or message.top_logprobs is None:
            print("No logprobs available!")
            return
        
        if page_num is not None:
            self.current_page = page_num
        
        total_tokens = message.top_logprobs.logprobs.shape[0]
        total_pages = (total_tokens + self.tokens_per_page - 1) // self.tokens_per_page
        
        # Bounds checking
        if self.current_page < 0:
            self.current_page = 0
        elif self.current_page >= total_pages:
            self.current_page = total_pages - 1
        
        start_idx = self.current_page * self.tokens_per_page
        end_idx = min(start_idx + self.tokens_per_page, total_tokens)
        
        print(f"\n📄 TOKEN PAGE {self.current_page + 1}/{total_pages} (tokens {start_idx + 1}-{end_idx} of {total_tokens})")
        print("=" * 80)
        
        for i in range(start_idx, end_idx):
            token_logprobs = message.top_logprobs.logprobs[i, :]
            token_ids = message.top_logprobs.token_ids[i, :]
            
            # Get top 3 alternatives
            probs = np.exp(token_logprobs)
            sorted_indices = np.argsort(probs)[::-1]
            
            chosen_text = self.decode_token(token_ids[0])
            chosen_prob = probs[0]
            
            print(f"\n🔹 Token {i+1:3d}: '{chosen_text:>15}' | prob:{chosen_prob:>7.4f} | logprob:{token_logprobs[0]:>7.3f}")
            
            # Show top 3 alternatives
            for rank in range(min(3, len(token_ids))):
                idx = sorted_indices[rank]
                token_id = token_ids[idx]
                prob = probs[idx]
                token_text = self.decode_token(token_id)
                
                if rank == 0:
                    marker = "  → "
                else:
                    marker = f"  {rank+1}: "
                
                print(f"{marker}'{token_text:>15}' | prob:{prob:>7.4f}")
        
        print(f"\n📊 Page {self.current_page + 1}/{total_pages} | Use 'pn'/'pp' for next/prev page, 'pg <num>' to jump")
    
    def show_help(self):
        """Show help message."""
        print(f"\n{'='*60}")
        print("INTERACTIVE LOGPROBS EXPLORER - COMMANDS")
        print(f"{'='*60}")
        print("Navigation:")
        print("  n / next     - Next example with logprobs")
        print("  p / prev     - Previous example with logprobs")
        print("  r / random   - Jump to random example")
        print("  g <num>      - Go to example number")
        print("")
        print("Token Pages:")
        print("  page         - Show current page of tokens")
        print("  pn / pnext   - Next page of tokens")
        print("  pp / pprev   - Previous page of tokens")
        print("  pg <num>     - Go to specific page")
        print("  tpp <num>    - Set tokens per page (default: 5)")
        print("")
        print("Token exploration:")
        print("  t <num>      - Show token details at position")
        print("  rt           - Show random token from current message")
        print("  scan         - Quick scan through all tokens")
        print("  first        - Show first 5 tokens")
        print("  last         - Show last 5 tokens")
        print("")
        print("Display:")
        print("  s / show     - Show current context")
        print("  h / help     - Show this help")
        print("  q / quit     - Exit program")
        print(f"{'='*60}")
    
    def run(self):
        """Run the interactive explorer."""
        print("🚀 Interactive Logprobs Explorer")
        print("Type 'h' for help, 'q' to quit")
        
        self.show_current_context()
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if not command:
                    continue
                
                if command in ['q', 'quit', 'exit']:
                    print("Goodbye! 👋")
                    break
                
                elif command in ['h', 'help']:
                    self.show_help()
                
                elif command in ['s', 'show']:
                    self.show_current_context()
                
                elif command in ['n', 'next']:
                    if self.current_pos < len(self.examples_with_logprobs) - 1:
                        self.current_pos += 1
                        self.current_page = 0  # Reset to first page
                        self.show_current_context()
                    else:
                        print("Already at last example!")
                
                elif command in ['p', 'prev']:
                    if self.current_pos > 0:
                        self.current_pos -= 1
                        self.current_page = 0  # Reset to first page
                        self.show_current_context()
                    else:
                        print("Already at first example!")
                
                elif command in ['r', 'random']:
                    self.current_pos = random.randint(0, len(self.examples_with_logprobs) - 1)
                    self.current_page = 0  # Reset to first page
                    self.show_current_context()
                
                elif command.startswith('g '):
                    try:
                        target = int(command.split()[1]) - 1
                        if 0 <= target < len(self.examples_with_logprobs):
                            self.current_pos = target
                            self.current_page = 0  # Reset to first page
                            self.show_current_context()
                        else:
                            print(f"Invalid position! Valid range: 1-{len(self.examples_with_logprobs)}")
                    except (ValueError, IndexError):
                        print("Usage: g <number>")
                
                elif command in ['page']:
                    self.show_token_page()
                
                elif command in ['pn', 'pnext']:
                    self.current_page += 1
                    self.show_token_page()
                
                elif command in ['pp', 'pprev']:
                    self.current_page -= 1
                    self.show_token_page()
                
                elif command.startswith('pg '):
                    try:
                        target_page = int(command.split()[1]) - 1
                        self.show_token_page(target_page)
                    except (ValueError, IndexError):
                        print("Usage: pg <page_number>")
                
                elif command.startswith('tpp '):
                    try:
                        new_tpp = int(command.split()[1])
                        if 1 <= new_tpp <= 20:
                            self.tokens_per_page = new_tpp
                            self.current_page = 0  # Reset to first page
                            print(f"Tokens per page set to {new_tpp}")
                        else:
                            print("Tokens per page must be between 1 and 20")
                    except (ValueError, IndexError):
                        print("Usage: tpp <number>")
                
                elif command.startswith('t '):
                    try:
                        token_idx = int(command.split()[1]) - 1
                        self.show_token_details(token_idx)
                    except (ValueError, IndexError):
                        print("Usage: t <token_number>")
                
                elif command in ['rt']:
                    example_idx, message_idx, example, message = self.get_current_example_and_message()
                    if message and message.top_logprobs is not None:
                        random_token = random.randint(0, message.top_logprobs.logprobs.shape[0] - 1)
                        self.show_token_details(random_token)
                    else:
                        print("No logprobs available!")
                
                elif command in ['scan']:
                    self.scan_tokens()
                
                elif command in ['first']:
                    self.show_first_last_tokens(True)
                
                elif command in ['last']:
                    self.show_first_last_tokens(False)
                
                else:
                    print(f"Unknown command: {command}. Type 'h' for help.")
            
            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                break
            except EOFError:
                print("\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def scan_tokens(self):
        """Quick scan through tokens showing basic info."""
        example_idx, message_idx, example, message = self.get_current_example_and_message()
        if message is None or message.top_logprobs is None:
            print("No logprobs available!")
            return
        
        print(f"\n📊 QUICK SCAN - {message.top_logprobs.logprobs.shape[0]} tokens")
        print("-" * 80)
        
        for i in range(min(20, message.top_logprobs.logprobs.shape[0])):
            token_logprobs = message.top_logprobs.logprobs[i, :]
            token_ids = message.top_logprobs.token_ids[i, :]
            
            chosen_text = self.decode_token(token_ids[0])
            chosen_prob = np.exp(token_logprobs[0])
            
            print(f"  {i+1:3d}: '{chosen_text:>15}' | prob:{chosen_prob:.4f} | logprob:{token_logprobs[0]:.3f}")
        
        if message.top_logprobs.logprobs.shape[0] > 20:
            print(f"  ... and {message.top_logprobs.logprobs.shape[0] - 20} more tokens")
    
    def show_first_last_tokens(self, show_first: bool):
        """Show first or last 5 tokens."""
        example_idx, message_idx, example, message = self.get_current_example_and_message()
        if message is None or message.top_logprobs is None:
            print("No logprobs available!")
            return
        
        num_tokens = message.top_logprobs.logprobs.shape[0]
        
        if show_first:
            indices = list(range(min(5, num_tokens)))
            print(f"\n🔝 FIRST 5 TOKENS")
        else:
            indices = list(range(max(0, num_tokens - 5), num_tokens))
            print(f"\n🔚 LAST 5 TOKENS")
        
        print("-" * 60)
        for i in indices:
            token_logprobs = message.top_logprobs.logprobs[i, :]
            token_ids = message.top_logprobs.token_ids[i, :]
            
            chosen_text = self.decode_token(token_ids[0])
            chosen_prob = np.exp(token_logprobs[0])
            
            print(f"  {i+1:3d}: '{chosen_text:>15}' | prob:{chosen_prob:.4f} | logprob:{token_logprobs[0]:.3f}")


class InteractiveLogprobsConfig(RunConfig):
    """Configuration for interactive logprobs explorer."""
    
    dataset_path: str
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B"
    
    def run(self):
        """Run the interactive explorer."""
        print(f"Loading dataset from {self.dataset_path}")
        try:
            training_examples = load_dataset(self.dataset_path)
            print(f"Loaded {len(training_examples)} training examples")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return
        
        try:
            explorer = InteractiveLogprobsExplorer(training_examples, self.tokenizer_name)
            explorer.run()
        except Exception as e:
            print(f"Error starting explorer: {e}")


if __name__ == "__main__":
    pydrantic.main(InteractiveLogprobsConfig(
        dataset_path="/home/sabri/code-memory/outputs/2025-07-06-13-37-13-gmail_synthesis/gmail_synthesis-0/artifact/dataset.pkl",
        output_dir=os.environ["CODEMEM_OUTPUT_DIR"],
    ))