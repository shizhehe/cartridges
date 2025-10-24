from capsules.configs.paper.mtob.evals.kv_cache_compression.common_settings import (
    get_configs,
)
import pydrantic


configs = get_configs(
    name="llama_8b_ke",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    direction="ke",
    attention_type="key_diff",
)

if __name__ == "__main__":
    pydrantic.main(configs)
