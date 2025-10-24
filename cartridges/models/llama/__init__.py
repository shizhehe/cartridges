from cartridges.models.llama.helpers import Llama3Helper
from cartridges.models.registry import MODEL_REGISTRY
from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM

for model_name in [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]:
    MODEL_REGISTRY.register(
        model_name=model_name,
        helper_cls=Llama3Helper,
        model_cls=FlexLlamaForCausalLM,
    )