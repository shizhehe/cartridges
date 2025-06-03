import abc
from pydrantic import ObjectConfig

from cartridges.clients.tokasaurus_batch import CartridgesConvoWithLogprobs
from cartridges.structs import Context, TrainingExample, TrainingExample
from cartridges.context import StructuredContext

import numpy as np


class ConvoSynthesizer(abc.ABC):

    class Config(ObjectConfig):
        _pass_as_config: bool = True

    def __init__(self, config: Config, context: Context | StructuredContext):
        self.config = config
        self.context = context

    @abc.abstractmethod
    def sample_convos(
        self,
        batch_idx: int,
        num_convos: int,
        total_batches: int,
    ) -> list[TrainingExample]:
        raise NotImplementedError()


def get_subcontext(batch_idx: int, total_batches: int, subcontexts: list[str]) -> str:
    """
    Maps a batch index to a subcontext based on an even distribution of batches.

    The batches are divided among the subcontexts as evenly as possible. If total_batches is not
    evenly divisible by the number of subcontexts, the first (total_batches % num_subcontexts) subcontexts
    receive one extra batch.

    Arguments:
        batch_idx (int): The index of the current batch.
        total_batches (int): The total number of batches.
        subcontexts (str): A sequence (or string) of subcontexts.

    Returns:
        int: The index of the subcontext corresponding to the batch index.
    """
    assert 0 <= batch_idx < total_batches
    num_subcontexts = len(subcontexts)
    num_subcontexts_with_extra = total_batches % num_subcontexts
    num_subcontexts_with_normal = num_subcontexts - num_subcontexts_with_extra

    batches_per_subcontext_normal = total_batches // num_subcontexts
    batches_per_subcontext_extra = batches_per_subcontext_normal + 1

    num_batches_with_extra = num_subcontexts_with_extra * batches_per_subcontext_extra
    num_batches_with_normal = (
        num_subcontexts_with_normal * batches_per_subcontext_normal
    )

    assert num_batches_with_extra + num_batches_with_normal == total_batches

    if batch_idx < num_batches_with_extra:
        subcontext_idx = batch_idx // batches_per_subcontext_extra
    else:
        subcontext_idx = (
            num_subcontexts_with_extra
            + (batch_idx - num_batches_with_extra) // batches_per_subcontext_normal
        )

    return subcontexts[subcontext_idx]


def responses_and_chats_to_training_examples(
    convos: list[CartridgesConvoWithLogprobs],
    answer_chats: list[list[dict]],
) -> list[TrainingExample]:
    examples = []
    for convo_response, chat in zip(
        convos,
        answer_chats,
        strict=True,
    ):
        messages = chat[1:] + [
            {"role": "assistant", "content": convo_response.assistant_text}
        ]

        header_locations = np.where(convo_response.token_ids == 128006)[0].tolist()
        assert len(header_locations) == len(chat) + 1
        assert header_locations[0] == 1
        _, prefix_end_idx, _ = header_locations

        token_ids = convo_response.token_ids[prefix_end_idx:]

        try:
            assert len(token_ids) > convo_response.num_output_tokens
            assert convo_response.top_logprob_logprobs.shape == convo_response.top_logprob_ids.shape
            assert convo_response.top_logprob_logprobs.shape[0] == len(token_ids) - 1, "You probably need to pull down on tokasaurus or your first message is not a system message"
        except AssertionError as e:
            print(f"Assertion error: {e}")
            print(f"Token IDs: {token_ids}")
            print(f"Top logprob IDs: {convo_response.top_logprob_ids}")
            print(f"Top logprob logprobs: {convo_response.top_logprob_logprobs}")
            return []

        examples.append(
            TrainingExample(
                messages=[TrainingExample.Message(**message) for message in messages],
                top_logprob_ids=convo_response.top_logprob_ids,
                top_logprob_logprobs=convo_response.top_logprob_logprobs.astype(np.float32),  # We can convert to float32 to save space in the file
                token_ids=token_ids,
                num_output_tokens=convo_response.num_output_tokens,
                type="todo",
                metadata={},
            )
        )
    return examples
