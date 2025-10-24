
CONFIGS = {

    "meta-llama/Llama-3.2-3B-Instruct": {
        "architectures": [
            "LlamaForCausalLM"
        ],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 128000,
        "eos_token_id": 128001,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 3072,
        "initializer_range": 0.02,
        "intermediate_size": 8192,
        "max_position_embeddings": 131072,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": 24,
        "num_hidden_layers": 28,
        "num_key_value_heads": 8,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": {
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        },
        "rope_theta": 500000.0,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.45.0.dev0",
        "use_cache": True,
        "vocab_size": 128256
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "architectures": [
            "LlamaForCausalLM"
        ],
        "attention_dropout": 0.0,
        "bos_token_id": 128000,
        "eos_token_id": 128001,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 14336,
        "max_position_embeddings": 131072,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": {
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        },
        "rope_theta": 500000.0,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.43.0.dev0",
        "vocab_size": 128256
    }
}




def get_llama_cache_size(
    model_name: str,
    num_tokens: int,
) -> int:
    cfg = CONFIGS[model_name]

    if "head_dim" in cfg:
        head_dim = cfg["head_dim"]
    else:
        head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]

    return (
        cfg["num_hidden_layers"] * 
        cfg["num_key_value_heads"] * 
        head_dim * 
        2 *  # for key and value
        2 * # for bfloat16
        num_tokens
    )
    

MODEL_TO_CACHE_SIZE_FN = {
    "meta-llama/Llama-3.2-3B-Instruct".lower(): lambda num_tokens: get_llama_cache_size(
        "meta-llama/Llama-3.2-3B-Instruct", num_tokens=num_tokens,
    ),
    "meta-llama/Llama-3.1-8B-Instruct".lower(): lambda num_tokens: get_llama_cache_size(
        "meta-llama/Llama-3.1-8B-Instruct", num_tokens=num_tokens,
    ),
}
    