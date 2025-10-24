from capsules.configs.paper.mtob.evals.kv_cache_compression.common_settings import (
    get_configs,
)
import pydrantic


configs = get_configs(
    name="llama_8b_ek",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    direction="ek",
    attention_type="expected_attention",
)

if __name__ == "__main__":
    pydrantic.main(configs)
