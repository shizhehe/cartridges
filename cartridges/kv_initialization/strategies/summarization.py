from dataclasses import asdict, dataclass
import os
import re

import torch
from capsules.clients.base import Client
from transformers import AutoTokenizer

from capsules.generate.structs import Context
from capsules.kv_initialization.base import (
    AttnConfig,
    KVCacheFactoryWithStateSaving,
    TrainableCache,
)

from pydrantic import ObjectConfig
from transformers import DynamicCache

from capsules.kv_initialization.tokenization_utils import (
    tokenize_data_into_system_prompt,
)
from capsules.utils import get_logger

logger = get_logger(__name__)


@dataclass
class SummarizationResults:
    summary_components: list[str]
    messages: list[dict]

    @property
    def summary(self):
        return "\n".join(self.summary_components)


# Added original_tokens parameter
def get_initial_summary_prompt(document: str, target_tokens: int, original_tokens: int) -> str:
    """
    Generate a prompt to compress a long document into a detailed, information-dense summary.

    Args:
        document (str): The long document text to compress.
        target_tokens (int): The target token count for the compressed summary.
        original_tokens (int): The token count of the original document.

    Returns:
        str: The generated prompt instructing the model to produce a compressed summary.
    """

    # Added information about original document length and compression goal
    return f"""# Document Compression Instructions

You are tasked with compressing the following long document.
- Original Document Length: approximately **{original_tokens} tokens**.
- Target Summary Length: approximately **{target_tokens} tokens**.

Your goal is to create an information-dense summary that captures as much specific detail and structure from the original document as possible, compressing it significantly while strictly adhering to the target length.

Please follow these guidelines precisely:

1.  **Maximize Information Density:** Include the most critical facts, figures, named entities, concepts, and structural elements from the document. Prioritize information that is specific and essential for understanding the original content.
2.  **Priority Ordering:** Begin the summary with the absolute most crucial information. Subsequently add details in decreasing order of importance.
3.  **Strict Length Control:** Your final summary MUST be very close to **{target_tokens} tokens**. Do not significantly exceed this limit.
4.  **Preserve Key Details:** Actively incorporate specific names, numbers, dates, relationships, and the core arguments or findings presented in the document. The goal is high-fidelity compression, not just a vague overview.
5.  **Planning First (Mandatory):** Before writing the summary, you MUST outline your plan. Analyze the document, identify the key information blocks, and list the core points you will include in order of importance. Explain *why* these points are chosen based on the goal of compressing {original_tokens} tokens down to {target_tokens}.
6.  **Final Output Format:** After your planning/thinking process, output the complete compressed summary enclosed ONLY between `<summary>` and `</summary>` tags. Do not include any other text outside these tags in the final output section.

# Document to Compress

<document>
{document}
</document>

# Your Task

1.  **Think Step-by-Step:** Write down your analysis of the document and your detailed plan for compression, considering the original length ({original_tokens} tokens) and the target ({target_tokens} tokens). Include the prioritized list of information.
2.  **Generate Compressed Summary:** After your plan, write the final compressed summary, ensuring it meets the detail and length requirements, enclosed in `<summary>` tags.
"""


# Need to modify summarize_document to pass current token count and needed tokens
def get_write_more_prompt(current_tokens: int, target_tokens: int, tokens_needed: int) -> str:
    """
    Generate a prompt to expand an existing summary with more details.

    Args:
        current_tokens (int): The current token count of the summary generated so far.
        target_tokens (int): The final desired token count for the summary.
        tokens_needed (int): The approximate number of additional tokens required.

    Returns:
        str: The generated prompt instructing the model to add more details.
    """
    return f"""# Summary Expansion Instructions

The summary generated so far is too short.
- Current summary length: approximately {current_tokens} tokens.
- Final target length: approximately {target_tokens} tokens.
- Additional tokens needed: approximately **{tokens_needed} tokens**.

Your task is to provide **additional data** from the original document to incorporate into the summary. This additional data should be approximately **{tokens_needed} tokens** long.

Please adhere to the following guidelines:

1.  **Identify Missing Information:** Review the original document and the summary implicitly generated so far (based on conversation history). Identify the *next most important* specific details, facts, figures, or concepts from the original document that are *not yet included*.
2.  **Prioritize:** Select the most critical missing information to add first.
3.  **Target Length for Addition:** The content you generate between the tags below should be approximately **{tokens_needed} tokens** long.
4.  **No Repetition:** CRITICAL: Do **not** repeat information that is likely already in the summary generated in previous turns. Focus only on *new* details from the original document.
5.  **Planning First (Mandatory):** Before writing the additional data, outline your plan. List the specific pieces of information you intend to add and why they are the next most important.
6.  **Output Format:** After your planning/thinking process, output *only* the additional data to be appended, enclosed between `<additional_data>` and `</additional_data>` tags. Do not re-write the full summary.

# Current Task

1.  **Think Step-by-Step:** Write down your analysis of what important information is still missing and your plan for what to add.
2.  **Generate Additional Data:** After your plan, write the additional data, ensuring it meets the detail, non-repetition, and length requirements for the *addition*, enclosed in `<additional_data>` tags.
"""


TOKEN_BUFFER_FOR_THINKING = 5000
MIN_SUMMARY_LEN = 50


def extract_from_tags(output: str, tag: str, choose_last: bool = False) -> str | None:
    matches = list(re.finditer(f"<{tag}>(.*?)</{tag}>", output, re.DOTALL))
    if (not choose_last and len(matches) != 1) or len(matches) == 0:
        return None
    return matches[-1].group(1).strip()


def summarize_document(
    document: str,
    target_tokens: int,
    client: Client,
    tokenizer: AutoTokenizer,
) -> SummarizationResults:
    # Calculate original document token count
    original_tokens = len(tokenizer.encode(document))
    logger.info(f"Original document token count: {original_tokens}") # Optional logging


    # Initialize with the first summary request
    messages = [
        {"role": "user", "content": get_initial_summary_prompt(document, target_tokens, original_tokens)}
    ]

    breakpoint()    
    # Get initial summary from the model
    response = client.chat(
        [messages],
        temperature=0.0,
        max_completion_tokens=target_tokens + TOKEN_BUFFER_FOR_THINKING,
    )
    breakpoint()
    text = response.samples[0].text
    messages.append({"role": "assistant", "content": text})

    # Extract the summary from the response
    summary = extract_from_tags(text, "summary")
    if summary is None:
        raise ValueError(f"Malformatted response (no summary tag):\n{text}")

    results = SummarizationResults([summary], messages)
    tokens = lambda: len(tokenizer.encode(results.summary))

    # Continue requesting more content until we reach target token count
    while tokens() < target_tokens:
        # Calculate how many more tokens we need
        current_token_count = tokens() # Calculate current length *before* the call
        tokens_needed = max(target_tokens - current_token_count, MIN_SUMMARY_LEN)
        messages.append(
            {"role": "user", "content": get_write_more_prompt(current_token_count, target_tokens, tokens_needed)}
        )

        response = client.chat(
            [messages],
            temperature=0.0,
            max_completion_tokens=target_tokens + TOKEN_BUFFER_FOR_THINKING,
        )
        text = response.samples[0].text
        messages.append({"role": "assistant", "content": text})

        additional_content = extract_from_tags(text, "additional_data")
        if additional_content is None:
            raise ValueError(f"Malformatted response (no content tags):\n{text}")

        results.summary_components.append(additional_content)

    return results


class KVCacheInitFromSummary(KVCacheFactoryWithStateSaving):
    class Config(KVCacheFactoryWithStateSaving.Config):
        client: ObjectConfig
        num_tokens: int

    def initalize_kv_cache_impl(
        self,
        context: Context,
        tokenizer,
        model,
        attn_config: AttnConfig,
    ) -> tuple[TrainableCache, dict]:
        is_ddp = "LOCAL_RANK" in os.environ
        if is_ddp:
            raise NotImplementedError("DDP not handled :(")

        logger.info("***Summarizing document for KV initalization - not reading from summary cache***")

        summary = summarize_document(
            document=context.to_string(),
            target_tokens=self.config.num_tokens,
            client=self.config.client.instantiate(),
            tokenizer=tokenizer,
        )

        breakpoint()

        input_ids = tokenize_data_into_system_prompt(
            tokenizer, summary.summary, self.config.num_tokens
        )

        init_cache = DynamicCache()
        with torch.no_grad():
            model(
                input_ids=input_ids.to(model.device),
                use_cache=True,
                past_key_values=init_cache,
            )

            return TrainableCache(
                config=attn_config,
                num_tokens=self.config.num_tokens,
                keys=list(init_cache.key_cache),
                values=list(init_cache.value_cache),
                num_frozen_tokens=self.config.num_frozen_tokens,
            ), asdict(summary)
