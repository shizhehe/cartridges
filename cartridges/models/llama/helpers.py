from typing import Any, List, Optional

import torch
from transformers import PreTrainedTokenizerFast

from cartridges.models.helpers import ModelHelper
from cartridges.structs import Conversation
from cartridges.datasets import DatasetElement

class Llama3Helper(ModelHelper):

    def get_chat_template(self) -> str:
        return CARTRIDGES_LLAMA3_CHAT_TEMPLATE

    def messages_to_element(
        self,
        messages: List[Conversation.Message],
        retokenize: bool = False,
        tokenizer: PreTrainedTokenizerFast | None = None,
        prob_drop_thinking: float = 1.0,
    ) -> DatasetElement:
        from cartridges.datasets import _base_convert_messages_to_element_retokenize, _base_convert_messages_to_element
        fn = _base_convert_messages_to_element_retokenize if retokenize else _base_convert_messages_to_element

        return fn(
            messages,
            tokenizer=tokenizer,
            message_start_tokens={
                # "<|start_header_id|>", "user", "<|end_header_id|>", "\n\n"
                "user": [128006, 882, 128007, 271],
                # "<|start_header_id|>", "assistant", "<|end_header_id|>", "\n\n"
                "assistant": [128006, 78191, 128007, 271],
                # "<|start_header_id|>", "system", "<|end_header_id|>", "\n\n"
                "system": [128006, 9125, 128007, 271],
            },
            message_end_tokens={
                # "<|eot_id|>"
                "user": [128009],
                "assistant": [128009],
                "system": [128009],
            },
            message_extra_end_tokens={
                "user": [],
                "assistant": [],
                "system": [],
            },

            # TODO(SE): Look into what is happening here. 
            drop_thinking_fn=lambda x: x,
        )

    def get_apply_chat_template_kwargs(
        self,
        enable_thinking: bool,
    ) -> dict[str, Any]:
        return {}
    
    def tokenize_system_prompt_with_max_tokens(
        self, 
        content: str,
        max_tokens: Optional[int],
    ) -> torch.Tensor:
        BOS_TOKEN_ID= 128000
        EOS_TOKEN_ID = 128009

        START_HEADER_ID = 128006
        END_HEADER_ID = 128007
        SYSTEM_ID = 9125

        input_ids = self.tokenizer.apply_chat_template([{"role": "system", "content": content}])
        assert input_ids[-1] == EOS_TOKEN_ID

        if max_tokens is not None and len(input_ids) > max_tokens:
            input_ids = input_ids[: max_tokens - 1] + [EOS_TOKEN_ID]

        assert input_ids[:4] == [BOS_TOKEN_ID, START_HEADER_ID, SYSTEM_ID, END_HEADER_ID]

        return torch.tensor(input_ids)[None, :]


    def get_cache_size(
        self,
        num_tokens: int,
    ) -> int:
        return get_llama_cache_size(self.model_name, num_tokens=num_tokens)


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


CARTRIDGES_LLAMA3_CHAT_TEMPLATE = """
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- if strftime_now is defined %}
        {%- set date_string = strftime_now("%d %b %Y") %}
    {%- else %}
        {%- set date_string = "26 Jul 2024" %}
    {%- endif %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
{%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
        {{- '{"name": "' + tool_call.name + '", ' }}
        {{- '"parameters": ' }}
        {{- tool_call.arguments | tojson }}
        {{- "}" }}
        {{- "<|eot_id|>" }}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"""

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