import json
import torch
import collections
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from capsules.kv_initialization.strategies.first_n_tokens_per_context import ( KVCacheInitFromFirstNTokensOfEachContext )
from capsules.kv_initialization.strategies.first_n_tokens import ( KVCacheInitFromFirstNTokensOfContext )
from capsules.tasks.finance import FinanceBenchContextConfig, FinanceBenchDocumentStructuredConfigCompose
from capsules.kv_initialization.base import AttnConfig, KVCacheFactory, TrainableCache
from capsules.train import TrainConfig, CacheAndModel

from capsules.configs.paper.financebench.compose.eval_utils import ( QA_PAIRS, run_query_set )

device = "cuda" if torch.cuda.is_available() else "cpu"


print("Loading model and tokenizer")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct").to(torch.bfloat16).to(device)


# ICL Baseline
num_tokens = 131702

def run_icl_baseline():

    pair_to_results = collections.defaultdict(dict)

    for pair, inputs in QA_PAIRS.items():

        print(f"Setting up cache for {pair}")
        company1, company2 = pair.split("_")

        config=FinanceBenchDocumentStructuredConfigCompose(
            doc_names=[f"{company1}_2022_10K", f"{company2}_2022_10K"], 
        )
        contexts = config.instantiate()

        comparative_messages = inputs["questions"]
        comparative_answers = inputs["answers"]

        if not comparative_answers: 
            print(f"Skipping {pair} as there are no answers\n")
            continue

        # kv_cache_initializer = KVCacheInitFromFirstNTokensOfEachContext.Config(
        #     max_tokens = num_tokens,
        # )

        kv_cache_initializer = KVCacheInitFromFirstNTokensOfContext.Config(
            max_tokens = num_tokens,
        )

        print(f"Initializing cache for {company1} and {company2} with {num_tokens} tokens")
        icl_cache: (
            TrainableCache
        ) = kv_cache_initializer.instantiate().initalize_kv_cache(
            context=contexts,  
            tokenizer=tokenizer,
            model=model,
            attn_config=AttnConfig(
                n_layers=model.config.num_hidden_layers,
                n_heads=model.config.num_key_value_heads,
                head_dim=model.config.hidden_size // model.config.num_attention_heads,
            ),
        )

        cache_mdoel = CacheAndModel(cache=icl_cache, model=model)
        cache_size = icl_cache.get_seq_length()

        responses, score = run_query_set(tokenizer, cache_mdoel, comparative_messages, comparative_answers)
        print(f"Running query set for {company1} and {company2}; Cache size: {cache_size}; Score: {score}\n")


        pair_to_results[pair] = {
            "cache_size": cache_size,
            "score": score,
        }
    
    # Save the results to a JSON file
    with open("icl_baseline_results12.json", "w") as f:
        json.dump(pair_to_results, f, indent=4)

if __name__ == "__main__":
    run_icl_baseline()



