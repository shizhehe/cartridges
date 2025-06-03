import math
import random
from capsules.datasets import msg
from capsules.generate.structs import Section


def split_into_passages(content: str, lines_per_passage: int) -> list[str]:
    """
    Splits the content into passages of a specified number of lines.
    """
    lines = content.splitlines()
    passages = []
    for i in range(0, len(lines), lines_per_passage):
        passage = "\n".join(lines[i : i + lines_per_passage])
        passages.append(passage)
    return passages


def next_passage_prediction(document_title: str, content: str, lines_per_passage: int):
    convos = []
    passages = split_into_passages(content, lines_per_passage)

    for passage_index, passage in enumerate(passages):
        num_lines = len(
            passages[min(passage_index + 1, len(passages) - 1)].splitlines()
        )
        user_message = f"""Here is a passage from the document "{document_title}"".
I want you to tell me the next {num_lines} lines from the document that come after this passage.
<passage>
{passage}
</passage>
I want you to tell me the next {num_lines} lines from the document that come after this passage.
Respond with just the text verbatim from the content and nothing else. That is, your response should be a direct quote from the document.
If it is the end of the document, respond with "<END OF DOCUMENT>".

"""

        assistant_message = (
            "<END OF DOCUMENT>"
            if passage_index == len(passages) - 1
            else passages[passage_index + 1]
        )
        convos.append(
            [
                msg(role="user", content=user_message),
                msg(role="assistant", content=assistant_message),
            ]
        )
    return convos


def previous_passage_prediction(
    document_title: str, content: str, lines_per_passage: int
):
    convos = []
    passages = split_into_passages(content, lines_per_passage)

    for passage_index, passage in enumerate(passages):
        num_lines = len(
            # wrap around is fine
            passages[passage_index - 1].splitlines()
        )
        user_message = f"""Here is a passage from the document "{document_title}"".
I want you to tell me the previous {num_lines} lines from the document that come before this passage.
<passage>
{passage}
</passage>
I want you to tell me the previous {num_lines} lines from the document that come before this passage.
Respond with just the text verbatim from the content and nothing else. That is, your response should be a direct quote from the document.
If it is the beginning of the document, respond with "<BEGINNING OF DOCUMENT>"."""

        assistant_message = (
            "<BEGINNING OF DOCUMENT>"
            if passage_index == 0
            else passages[passage_index - 1]
        )
        convos.append(
            [
                msg(role="user", content=user_message),
                msg(role="assistant", content=assistant_message),
            ]
        )
    return convos


def unshuffle(
    document_title: str,
    content: str,
    lines_per_passage: int,
    num_shuffles: int = 3,
    seed: int = 4,
):
    random.seed(seed)
    convos = []
    passages = split_into_passages(content, lines_per_passage)

    for passage_index, passage in enumerate(passages):
        for shuffle_index in range(num_shuffles):
            # shuffle passages
            lines = passage.splitlines()
            shuffled_lines = lines.copy()
            random.shuffle(shuffled_lines)

            user_message = f"""Here is a passage from the document "{document_title}" with its lines shuffled out of order.
I want you to reorder the lines to reconstruct the original passage.
<shuffled_passage>
{"\n".join(shuffled_lines)}
</shuffled_passage>
Respond with just the correctly ordered text verbatim from the document and nothing else. Your response should be the passage with lines in their proper order."""

            convos.append(
                [
                    msg(role="user", content=user_message),
                    msg(role="assistant", content=passage),
                ]
            )

    return convos


def fill_in_the_blank_masked_words(
    document_title: str,
    content: str,
    lines_per_passage: int,
    mask_percentage: float = 0.4,
    mask_token: str = "[MASK]",
    num_iters: int = 3,
    seed: int = 42,
) -> list:
    random.seed(seed)
    convos = []
    passages = split_into_passages(content, lines_per_passage)

    for passage_index, passage in enumerate(passages):
        words = passage.split()
        num_words = len(words)

        if num_words == 0:
            continue  # Skip empty passages

        for _ in range(num_iters):

            num_to_mask = math.ceil(
                num_words * mask_percentage
            )  # Use ceil to ensure masking happens even for low percentages
            num_to_mask = min(
                num_words, max(0, num_to_mask)
            )  # Clamp between 0 and num_words

            if num_to_mask == 0 and num_words > 0:
                # If percentage is very low, ensure at least one word is masked if possible
                indices_to_mask = random.sample(range(num_words), 1)
            elif num_to_mask > 0:
                indices_to_mask = random.sample(range(num_words), num_to_mask)
            else:
                # No words to mask (either num_words was 0 initially, or somehow num_to_mask became 0)
                continue  # Skip if nothing can be masked

            mask_indices_set = set(indices_to_mask)
            masked_words = [
                mask_token if i in mask_indices_set else word
                for i, word in enumerate(words)
            ]
            masked_passage = " ".join(masked_words)

            # Determine the approximate number of masked words for the prompt
            actual_masked_count = len(indices_to_mask)

            user_message = f"""The following passage is from the document "{document_title}".
    Approximately {actual_masked_count} words have been replaced with {mask_token}.
    Your task is to reconstruct the original passage by filling in the missing words.
    <masked_passage>
    {masked_passage}
    </masked_passage>
    Respond with ONLY the original, complete passage verbatim from the document and nothing else.
    Your response should be the passage with all {mask_token} placeholders correctly filled in."""

            assistant_message = passage  # The original passage is the target

            convos.append(
                [
                    msg(role="user", content=user_message),
                    msg(role="assistant", content=assistant_message),
                ]
            )

    return convos


# def between_passage_prediction(
#     document_title: str, content: str, lines_per_passage: int, gap_size: int = 1
# ):
#     convos = []
#     passages = split_into_passages(content, lines_per_passage)

#     # Need at least 3 passages to have something in between
#     if len(passages) < gap_size + 2:
#         return convos

#     for passage_index in range(len(passages) - (gap_size + 1)):
#         first_passage = passages[passage_index]
#         last_passage = passages[passage_index + gap_size + 1]
#         middle_passage = passages[passage_index + 1 : passage_index + gap_size + 1]

#         # Combine middle passages if gap_size > 1
#         middle_text = "\n".join(middle_passage)

#         user_message = f"""..."""


# def next_passage_prediction(
#     document_title: str, subcontext: list[Section], lines_per_passage: int
# ):
#     convos = []
#     for section in subcontext:
#         passages = split_into_passages(section.content, lines_per_passage)

#         for passage_index, passage in enumerate(passages):
#             user_message = f"""Here is a passage from the document "{document_title}" in the section "{section.title}".
# I want you to tell me the next {len(passages[min(passage_index + 1, len(passages) - 1)].splitlines())} lines that come after this passage. This the end of the section, respond with "END OF SECTION".
# <passage>
# {passage}
# </passage>
# Respond with just the text verbatim from the content and nothing else.
# That is, your respond should either be a direct quote from the document, or the text "END OF SECTION"."""

#             if passage_index == len(passages) - 1:
#                 assistant_message = "END OF SECTION"
#             else:
#                 assistant_message = passages[passage_index + 1]
#             convos.append(
#                 [
#                     msg(role="user", content=user_message),
#                     msg(role="assistant", content=assistant_message),
#                 ]
#             )
#     return convos


# def previous_passage_prediction(
#     document_title: str, subcontext: list[Section], lines_per_passage: int
# ):
#     convos = []
#     for section in subcontext:
#         passages = split_into_passages(section.content, lines_per_passage)

#         for passage_index, passage in enumerate(passages):
#             # this negavie roll around is fine
#             prev_lines_count = len(passages[passage_index - 1].splitlines())

#             user_message = f"""Here is a passage from the document "{document_title}" in the section "{section.title}".
# I want you to tell me the previous {prev_lines_count} lines that come before this passage. If this is the beginning of the section, respond with "BEGINNING OF SECTION".
# <passage>
# {passage}
# </passage>
# Respond with just the text verbatim from the content and nothing else.
# That is, your respond should either be a direct quote from the document, or the text "BEGINNING OF SECTION"."""

#             if passage_index == 0:
#                 assistant_message = "BEGINNING OF SECTION"
#             else:
#                 assistant_message = passages[passage_index - 1]

#             convos.append(
#                 [
#                     msg(role="user", content=user_message),
#                     msg(role="assistant", content=assistant_message),
#                 ]
#             )

#     return convos


# GAP_SIZE = 1


# def between_passage_prediction(
#     document_title: str, subcontext: list[Section], lines_per_passage: int
# ):
#     convos = []
#     for section in subcontext:
#         passages = split_into_passages(section.content, lines_per_passage)

#         # Need at least 3 passages to have something in between
#         if len(passages) < GAP_SIZE + 2:
#             continue

#         for passage_index in range(len(passages) - (GAP_SIZE + 1)):
#             first_passage = passages[passage_index]
#             last_passage = passages[passage_index + GAP_SIZE + 1]
#             middle_passage = passages[passage_index + 1 : passage_index + GAP_SIZE + 1]

#             # Combine middle passages if gap_size > 1
#             middle_text = "\n".join(middle_passage)

#             user_message = f"""Here are two passages from the document "{document_title}" in the section "{section.title}".
# I want you to tell me what passage comes between these two passages.
# <first_passage>
# {first_passage}
# </first_passage>

# <last_passage>
# {last_passage}
# </last_passage>

# Respond with just the text verbatim that comes between these passages and nothing else."""

#             assistant_message = middle_text
#             convos.append(
#                 [
#                     msg(role="user", content=user_message),
#                     msg(role="assistant", content=assistant_message),
#                 ]
#             )

#     return convos
