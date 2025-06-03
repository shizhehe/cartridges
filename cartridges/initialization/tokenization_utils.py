from typing import Optional
import torch


BOS_TOKEN_ID= 128000
EOS_TOKEN_ID = 128009

START_HEADER_ID = 128006
END_HEADER_ID = 128007
SYSTEM_ID = 9125


def tokenize_data_into_system_prompt(
    tokenizer,
    content: str,
    max_tokens: Optional[int],
) -> torch.Tensor:
    input_ids = tokenizer.apply_chat_template([{"role": "system", "content": content}])
    assert input_ids[-1] == EOS_TOKEN_ID

    if max_tokens is not None and len(input_ids) > max_tokens:
        input_ids = input_ids[: max_tokens - 1] + [EOS_TOKEN_ID]

    assert input_ids[:4] == [BOS_TOKEN_ID, START_HEADER_ID, SYSTEM_ID, END_HEADER_ID]

    return torch.tensor(input_ids)[None, :]
