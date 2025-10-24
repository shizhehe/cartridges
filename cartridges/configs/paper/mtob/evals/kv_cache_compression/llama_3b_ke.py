from capsules.configs.paper.mtob.evals.kv_cache_compression.common_settings import (
    get_configs,
)
import pydrantic


configs = get_configs(
    name="llama_3b_ke",
    model="meta-llama/Meta-Llama-3.2-3B-Instruct",
    direction="ke",
)

if __name__ == "__main__":
    pydrantic.main(configs)
