from typing import Optional
import torch



def llama_tokenize_data_into_system_prompt(
    tokenizer,
    content: str,
    max_tokens: Optional[int],
) -> torch.Tensor:
    BOS_TOKEN_ID= 128000
    EOS_TOKEN_ID = 128009

    START_HEADER_ID = 128006
    END_HEADER_ID = 128007
    SYSTEM_ID = 9125

    input_ids = tokenizer.apply_chat_template([{"role": "system", "content": content}])
    assert input_ids[-1] == EOS_TOKEN_ID

    if max_tokens is not None and len(input_ids) > max_tokens:
        input_ids = input_ids[: max_tokens - 1] + [EOS_TOKEN_ID]

    assert input_ids[:4] == [BOS_TOKEN_ID, START_HEADER_ID, SYSTEM_ID, END_HEADER_ID]

    return torch.tensor(input_ids)[None, :]


def qwen_tokenize_data_into_system_prompt(
    tokenizer,
    content: str,
    max_tokens: Optional[int],
) -> torch.Tensor:
    END_TOKEN_IDS = [151645, 198]

    input_ids = tokenizer.apply_chat_template([{"role": "system", "content": content}])

    if max_tokens is not None and len(input_ids) > max_tokens:
        input_ids = input_ids[: max_tokens - 1] + END_TOKEN_IDS

    return torch.tensor(input_ids)[None, :]


MODEL_TO_SYSTEM_PROMPT_TOKENIZER = {
    "meta-llama/Llama-3.2-1B-Instruct": llama_tokenize_data_into_system_prompt,
    "Qwen/Qwen3-4b": qwen_tokenize_data_into_system_prompt,
}
