import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import tiktoken
import statistics

class TokenizationPerformanceComparator:
    def __init__(self, model_name: str = "gpt-4"):
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.test_texts = [
            "This is a short test text." * 100,
            "This is a longer test text that contains more words and should take more time to tokenize." * 100,
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat." * 100,
            "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet at least once, making it useful for testing text processing algorithms." * 100,
            "In the field of natural language processing, tokenization is a fundamental step that involves breaking down text into smaller units called tokens. These tokens can be words, subwords, or characters depending on the tokenization strategy employed." * 100,
        ] * 100  # Multiply to create a larger dataset for meaningful performance comparison

    def tokenize_serial(self, texts: List[str]) -> List[List[int]]:
        """Tokenize texts serially (one after another)"""
        results = []
        for text in texts:
            tokens = self.tokenizer.encode(text)
            results.append(tokens)
        return results

    def tokenize_threaded(self, texts: List[str], max_workers: int = 4) -> List[List[int]]:
        """Tokenize texts using threading"""
        def tokenize_single(text):
            return self.tokenizer.encode(text)
        
        results = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(tokenize_single, text): i 
                             for i, text in enumerate(texts)}
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                results[index] = future.result()
        
        return results

    async def tokenize_async(self, texts: List[str]) -> List[List[int]]:
        """Tokenize texts using async/await"""
        async def tokenize_single_async(text):
            # Simulate async operation by running in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.tokenizer.encode, text)
        
        tasks = [tokenize_single_async(text) for text in texts]
        return await asyncio.gather(*tasks)

    def tokenize_batch(self, texts: List[str], batch_size: int = 50) -> List[List[int]]:
        """Tokenize texts in batches"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self.tokenizer.encode_batch(batch)
            results.extend(batch_results)
        return results

    def measure_performance(self, method_name: str, method_func, *args, **kwargs) -> dict:
        """Measure the performance of a tokenization method"""
        start_time = time.perf_counter()
        
        if asyncio.iscoroutinefunction(method_func):
            result = asyncio.run(method_func(*args, **kwargs))
        else:
            result = method_func(*args, **kwargs)
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Verify all results are the same length
        total_tokens = sum(len(tokens) for tokens in result)
        
        return {
            'method': method_name,
            'execution_time': execution_time,
            'total_texts': len(self.test_texts),
            'total_tokens': total_tokens,
            'tokens_per_second': total_tokens / execution_time,
            'texts_per_second': len(self.test_texts) / execution_time
        }

    def run_comparison(self, num_runs: int = 3) -> dict:
        """Run performance comparison across all methods"""
        methods = [
            ('Serial', self.tokenize_serial),
            ('Threaded (4 workers)', lambda texts: self.tokenize_threaded(texts, max_workers=4)),
            ('Threaded (8 workers)', lambda texts: self.tokenize_threaded(texts, max_workers=8)),
            ('Async', self.tokenize_async),
            ('Batch (25)', lambda texts: self.tokenize_batch(texts, batch_size=25)),
            ('Batch (50)', lambda texts: self.tokenize_batch(texts, batch_size=50)),
            ('Batch (100)', lambda texts: self.tokenize_batch(texts, batch_size=100)),
        ]
        
        results = {}
        
        print(f"Running tokenization performance comparison with {len(self.test_texts)} texts...")
        print(f"Each method will be run {num_runs} times and averaged.\n")
        
        for method_name, method_func in methods:
            print(f"Testing {method_name}...")
            
            run_results = []
            for run in range(num_runs):
                result = self.measure_performance(method_name, method_func, self.test_texts)
                run_results.append(result)
            
            # Calculate averages
            avg_time = statistics.mean(r['execution_time'] for r in run_results)
            avg_tokens_per_sec = statistics.mean(r['tokens_per_second'] for r in run_results)
            avg_texts_per_sec = statistics.mean(r['texts_per_second'] for r in run_results)
            
            results[method_name] = {
                'avg_execution_time': avg_time,
                'avg_tokens_per_second': avg_tokens_per_sec,
                'avg_texts_per_second': avg_texts_per_sec,
                'individual_runs': run_results
            }
            
            print(f"  Average time: {avg_time:.4f}s")
            print(f"  Average tokens/sec: {avg_tokens_per_sec:.2f}")
            print(f"  Average texts/sec: {avg_texts_per_sec:.2f}\n")
        
        return results

    def print_summary(self, results: dict):
        """Print a summary of the performance comparison"""
        print("=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        
        # Sort by execution time
        sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_execution_time'])
        
        print(f"{'Method':<20} {'Time (s)':<10} {'Tokens/s':<12} {'Texts/s':<10}")
        print("-" * 60)
        
        baseline_time = sorted_results[0][1]['avg_execution_time']
        
        for method_name, result in sorted_results:
            speedup = baseline_time / result['avg_execution_time']
            print(f"{method_name:<20} {result['avg_execution_time']:<10.4f} "
                  f"{result['avg_tokens_per_second']:<12.2f} "
                  f"{result['avg_texts_per_second']:<10.2f} "
                  f"({speedup:.2f}x)")

def main():
    comparator = TokenizationPerformanceComparator()
    results = comparator.run_comparison(num_runs=3)
    comparator.print_summary(results)

if __name__ == "__main__":
    main()