import pydrantic

from capsules.baselines.mtob_kv_cache_compression import MtobKVCacheCompressionBaseline
from capsules.clients.together import TogetherClient


if __name__ == "__main__":
    config = MtobKVCacheCompressionBaseline(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        kv_compression="expected_attention",
        kv_compression_ratio=0.9,
        direction="ke",
        grammar_book="medium",
        include_wordlist=False,
        include_sentences=True,
        temperature=0.0,
        max_completion_tokens=64,
        num_samples=1,
        prompt_type="generic",
    )
    pydrantic.main([config])
