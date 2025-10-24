from cartridges.models.qwen.helpers import QwenHelper
from cartridges.models.registry import MODEL_REGISTRY
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

for model_name in [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-4B-Instruct-2507",
]:
    MODEL_REGISTRY.register(
        model_name=model_name,
        helper_cls=QwenHelper,
        model_cls=FlexQwen3ForCausalLM,
    )