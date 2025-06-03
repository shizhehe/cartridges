from copy import deepcopy
import random
import time
from typing import Literal, Optional, Tuple
from capsules.clients.base import Client
from capsules.clients.tokasaurus_batch import TokasaurusBatchClient
from capsules.context import StructuredContext
from transformers import AutoTokenizer
from capsules.utils import get_logger

logger = get_logger(__name__)

TOKEN_BUFFER = 2
SECTION_BUFFER = 2

import abc
from dataclasses import dataclass


@dataclass
class DictParent:
    parent: "ContextTreeDictNode"
    key: str


@dataclass
class ListParent:
    parent: "ContextTreeListNode"
    index: int


@dataclass
class ContextTree(abc.ABC):
    parent_data: DictParent | ListParent | None

    def leaves(self) -> list["ContextTreeLeaf"]:
        raise NotImplementedError("Must be implemented in subclasses")
    
    def nodes(self) -> list["ContextTree"]:
        raise NotImplementedError("Must be implemented in subclasses")
    def path(self) -> str:
        if self.parent_data is None:
            return ""

        parent_path = self.parent_data.parent.path()

        if isinstance(self.parent_data, DictParent):
            suffix = f".{self.parent_data.key}"
            if parent_path == "":
                suffix = suffix[1:]

        elif isinstance(self.parent_data, ListParent):
            suffix = f"[{self.parent_data.index}]"

        else:
            raise ValueError(f"Unknown parent data type: {type(self.parent_data)}")

        return f"{parent_path}{suffix}"


@dataclass
class ContextTreeLeaf(ContextTree):
    value: int | float | str | None

    num_tokens: int
    summary: str | None = None


    def leaves(self) -> list["ContextTreeLeaf"]:
        return [self]
    
    def nodes(self) -> list["ContextTree"]:
        return [self]


@dataclass
class ContextTreeDictNode(ContextTree):
    value: dict[str, "ContextTree"]
    summary: str | None = None


    def leaves(self) -> list[ContextTreeLeaf]:
        result = []
        for child in self.value.values():
            if isinstance(child, ContextTreeLeaf):
                result.append(child)
            elif isinstance(child, ContextTreeListNode):
                result += child.leaves()
            elif isinstance(child, ContextTreeDictNode):
                result += child.leaves()
            else:
                raise ValueError(f"Unknown child type: {type(child)}")
        return result

    def nodes(self) -> list["ContextTree"]:
        result = [self]
        for child in self.value.values():
            result += child.nodes()
        return result


@dataclass
class ContextTreeListNode(ContextTree):
    value: list["ContextTree"]
    summary: str | None = None


    def leaves(self) -> list[ContextTreeLeaf]:
        result = []
        for child in self.value:
            result += child.leaves()
        return result

    def nodes(self) -> list["ContextTree"]:
        result = [self]
        for child in self.value:
            result += child.nodes()
        return result


class ContextTreePagedText(ContextTreeListNode):
    pass


def maybe_split_text_by_token_limit(
    text: str,
    tokenizer: AutoTokenizer,
    max_tokens_per_section: int,
) -> str | list[str]:
    tokenize = lambda text: tokenizer.encode(text, add_special_tokens=False)

    """
    Split the text into smaller chunks if it exceeds the token limit.
    """
    tokens = tokenize(text)
    if len(tokens) <= max_tokens_per_section:
        return text

    # First try to split by lines
    lines = text.splitlines(keepends=True)
    result = []
    current_chunk = []
    current_chunk_tokens = 0

    def add_line():
        nonlocal current_chunk, current_chunk_tokens
        assert current_chunk
        joined = "".join(current_chunk)
        assert len(tokenize(joined)) <= max_tokens_per_section
        result.append(joined)
        current_chunk = []
        current_chunk_tokens = 0

    for line in lines:
        tokens = tokenize(line)
        num_tokens = len(tokens)

        if num_tokens < max_tokens_per_section:
            if num_tokens + current_chunk_tokens > max_tokens_per_section:
                add_line()

            current_chunk.append(line)
            current_chunk_tokens += num_tokens + TOKEN_BUFFER

            continue

        # TODO(ryan): we could merge the first part of the line with the current chunk, but lets not
        if current_chunk:
            add_line()

        assert current_chunk_tokens == 0
        assert not current_chunk

        for start_idx in range(0, len(line), max_tokens_per_section):
            chunk = tokens[start_idx : start_idx + max_tokens_per_section]
            if len(chunk):
                chunk_text = tokenizer.decode(chunk)
                result.append(chunk_text)

    if current_chunk:
        add_line()

    return result


StructuredContextAtomTypes = StructuredContext | str | int | float
StructuredContextTypes = (
    None | StructuredContextAtomTypes | list[StructuredContextAtomTypes]
)


def structured_context_to_context_tree(
    context: StructuredContext,
    tokenizer: AutoTokenizer,
    max_tokens_per_section: int,
) -> ContextTree:
    """
    Convert the structured context to a context tree.
    """

    def recurse(
        obj: StructuredContextTypes, parent_data: DictParent | ListParent | None
    ) -> ContextTree:

        if isinstance(obj, str):
            # TODO - there is a pydantic error here
            result = maybe_split_text_by_token_limit(
                obj, tokenizer, max_tokens_per_section
            )

            if isinstance(result, str):
                return ContextTreeLeaf(
                    value=result,
                    parent_data=parent_data,
                    num_tokens=len(tokenizer.encode(result)),
                )
            assert isinstance(result, list)

            node = ContextTreePagedText(value=[], parent_data=parent_data)
            for item in result:
                item_node = ContextTreeLeaf(
                    value=item,
                    parent_data=ListParent(parent=node, index=len(node.value)),
                    num_tokens=len(tokenizer.encode(item)),
                )
                node.value.append(item_node)

            return node

        if isinstance(obj, (int, float)) or obj is None:
            return ContextTreeLeaf(
                value=obj,
                parent_data=parent_data,
                num_tokens=2, # kind of a lie - tokens are just used for weighting so this is ifne
            )

        if isinstance(obj, list):
            node = ContextTreeListNode(value=[], parent_data=parent_data)
            for item in obj:
                item_node = recurse(
                    item, ListParent(parent=node, index=len(node.value))
                )
                node.value.append(item_node)
            return node

        node = ContextTreeDictNode(value={}, parent_data=parent_data)

        for field in obj.model_fields:
            assert field not in node.value
            value = getattr(obj, field)

            node.value[field] = recurse(
                value, parent_data=DictParent(parent=node, key=field)
            )

        return node

    return recurse(context, None)


def summarize_context_tree(
    tree: ContextTree,
    tokenizer: AutoTokenizer,
    max_tokens: int,
) -> str:
    pass


def flood_fill_from_leafs(
    leafs: list[ContextTreeLeaf],
    num_nodes_to_include: int,
    sibling_bias: int = 3,
) -> list[ContextTree]:
    nodes = []
    get_cur_node_ids = lambda: set([id(node) for node in nodes])

    for leaf in leafs:
        cur = leaf
        while cur.parent_data is not None:
            nodes.append(cur)
            cur = cur.parent_data.parent

    def add_dict_attrs():
        cur_node_ids = get_cur_node_ids()

        for node in nodes:
            if isinstance(node, ContextTreeDictNode):
                for child in node.value.values():
                    if isinstance(child, ContextTreeLeaf):
                        if id(child) not in cur_node_ids:
                            nodes.append(child)

    add_dict_attrs()

    for _ in range(num_nodes_to_include - len(nodes)):
        nodes_that_could_be_picked = []
        cur_node_ids = get_cur_node_ids()

        def maybe_append(child):
            if id(child) not in cur_node_ids:
                nodes_that_could_be_picked.append(child)

        for node in nodes:
            if isinstance(node, ContextTreeListNode):
                for child in node.value:
                    maybe_append(child)
            elif isinstance(node, ContextTreeDictNode):
                for child in node.value.values():
                    maybe_append(child)
            else:
                if not isinstance(node, ContextTreeLeaf):
                    print(type(node))
                assert isinstance(node, ContextTreeLeaf)
                parent_data = node.parent_data
                if isinstance(parent_data, ListParent):
                    for _ in range(sibling_bias):
                        if parent_data.index > 0:
                            maybe_append(
                                parent_data.parent.value[parent_data.index - 1]
                            )

                        if parent_data.index + 1 < len(parent_data.parent.value):
                            maybe_append(
                                parent_data.parent.value[parent_data.index + 1]
                            )

        assert nodes_that_could_be_picked  # todo

        new_node = random.choice(nodes_that_could_be_picked)

        assert new_node.parent_data is not None
        assert id(new_node.parent_data.parent) in cur_node_ids
        nodes.append(new_node)
        add_dict_attrs()

    return nodes



def flood_fill_from_leafs_tokens(
    leafs: list[ContextTreeLeaf],
    max_tokens: int | Tuple[int, int],
    sibling_bias: int = 3,
) -> list[ContextTree]:
    nodes: list[ContextTree] = []
    get_cur_node_ids = lambda: set([id(node) for node in nodes])
    
    if not isinstance(max_tokens, int):
        max_tokens = random.randint(max_tokens[0], max_tokens[1])
    
    # for leaf in leafs:
    #     cur = leaf
    #     while cur.parent_data is not None:
    #         nodes.append(cur)
    #         cur = cur.parent_data.parent

    def add_dict_attrs():
        cur_node_ids = get_cur_node_ids()

        for node in nodes:
            if isinstance(node, ContextTreeDictNode):
                for child in node.value.values():
                    if isinstance(child, ContextTreeLeaf):
                        if id(child) not in cur_node_ids:
                            nodes.append(child)

    add_dict_attrs()

    while sum(node.num_tokens for node in nodes if isinstance(node, ContextTreeLeaf)) < max_tokens:
        nodes_that_could_be_picked = []
        cur_node_ids = get_cur_node_ids()

        def maybe_append(child):
            if id(child) not in cur_node_ids:
                nodes_that_could_be_picked.append(child)

        for node in nodes:
            if isinstance(node, ContextTreeListNode):
                for child in node.value:
                    maybe_append(child)
            elif isinstance(node, ContextTreeDictNode):
                for child in node.value.values():
                    maybe_append(child)
            else:
                if not isinstance(node, ContextTreeLeaf):
                    print(type(node))
                assert isinstance(node, ContextTreeLeaf)
                parent_data = node.parent_data
                if isinstance(parent_data, ListParent):
                    for _ in range(sibling_bias):
                        if parent_data.index > 0:
                            maybe_append(
                                parent_data.parent.value[parent_data.index - 1]
                            )

                        if parent_data.index + 1 < len(parent_data.parent.value):
                            maybe_append(
                                parent_data.parent.value[parent_data.index + 1]
                            )

        if len(nodes_that_could_be_picked) == 0:
            break

        new_node = random.choice(nodes_that_could_be_picked)

        assert new_node.parent_data is not None
        assert id(new_node.parent_data.parent) in cur_node_ids
        nodes.append(new_node)
        add_dict_attrs()

    return nodes


def serialize_with_elide(ctx_tree, nodes, include_path_in_tag: bool = True) -> str:
    node_ids = set([id(node) for node in nodes])

    data = []

    def get_path_tag(node):
        if include_path_in_tag:
            return node.path()
        else:
            return node.path().split(".")[-1]

    def recurse(node):
        if isinstance(node, ContextTreeLeaf):
            if id(node) not in node_ids:
                return
            print_tag = node.parent_data is None or not isinstance(
                node.parent_data.parent, ContextTreePagedText
            )
            if print_tag:
                data.append(f"<{get_path_tag(node)}>")
            data.append(f"{node.value}")
            if print_tag:
                data.append(f"</{get_path_tag(node)}>")
            return

        if isinstance(node, ContextTreeDictNode):
            data.append(f"<{get_path_tag(node)}>")
            if node.summary is not None:
                data.extend(["<summary>", f"{node.summary}", "</summary>"])
            for key, child in sorted(
                node.value.items(),
                key=lambda item: (-1 if isinstance(item[1], ContextTreeLeaf) else 0),
            ):
                # if isinstance(child, ContextTreeLeaf):
                #     if id(child) not in node_ids:
                #         breakpoint()

                #     assert id(child) in node_ids, "asblasbhfdfdf"

                if id(child) in node_ids:
                    recurse(child)
                else:
                    data.extend([f"<{child.path()}>", "[Elided]", f"</{child.path()}>"])
            data.append(f"</{get_path_tag(node)}>")
            return

        assert isinstance(node, ContextTreeListNode)
        if len(node.value) == 0:
            data.append(f"<{get_path_tag(node)}:  empty list />")
            return


        ranges = []
        for index, child in enumerate(node.value):
            shown = id(child) in node_ids

            if not ranges:
                ranges.append((shown, index, None))
                continue

            if ranges[-1][0] == shown:
                continue

            ranges[-1] = (ranges[-1][0], ranges[-1][1], index)
            ranges.append((shown, index, None))

        ranges[-1] = (ranges[-1][0], ranges[-1][1], len(node.value))

        if isinstance(node, ContextTreePagedText):
            data.append(f"<{get_path_tag(node)}: {len(node.value)} pages of text>")
            for shown, start, end in ranges:
                if shown:
                    tag = (
                        f"pages {start + 1} through {end} of {get_path_tag(node)}"
                        if start + 1 != end
                        else f"page {start + 1} of {get_path_tag(node)}"
                    )
                    data.append(f"<{tag}>")
                    for index in range(start, end):
                        child = node.value[index]
                        assert id(child) in node_ids
                        recurse(child)
                    data.append(f"</{tag}>")
                else:
                    data.append(
                        f"[pages {start + 1} through {end} of {get_path_tag(node)} elided]"
                        if start + 1 != end
                        else f"[page {start + 1} of {get_path_tag(node)} elided]"
                    )
        else:
            data.append(f"<{get_path_tag(node)}: list with {len(node.value)} elements>")
            if node.summary is not None:
                data.extend(["<summary>", f"{node.summary}", "</summary>"])
            for shown, start, end in ranges:
                if shown:
                    for index in range(start, end):
                        child = node.value[index]
                        # data.append(f"<element {start + 1}>")
                        assert id(child) in node_ids
                        recurse(child)
                        # data.append(f"</element {start + 1}>")
                else:
                    data.append(
                        f"[{get_path_tag(node)}: elements {start + 1} through {end} elided]"
                        if start + 1 != end
                        else f"[list element {start + 1} elided]"
                    )
        data.append(f"</{get_path_tag(node)}>")

    recurse(ctx_tree)
    return "\n".join(data)

# First describe what the subsection is (e.g. "an email" or "a chapter from a book") then provide specific details.
# Focus on the information in this subsection that is most likely not contained in the other subsections.

TEXT_SUMMARIZATION_PROMPT = f"""\
Please summarize the following subsection from a larger corpus of text in 1-2 sentences (less than {{summarization_length}} tokens).

Include specific data including names, dates, and numbers.

<text>
{{text}}
</text>

Do not include any other text other than the summary in your response.\
"""

LIST_SUMMARIZATION_PROMPT = f"""\
Please summarize the following list of elements in 1-2 sentences (less than {{summarization_length}} tokens).

First describe what it is a list of (e.g. "a list of emails" or "a list of chapters from a book") then provide specific details.

<text>
{{text}}
</text>

Do not include any other text other than the summary in your response.\
"""

DICT_SUMMARIZATION_PROMPT = f"""\
Please summarize the following element in 1-2 sentences (less than {{summarization_length}} tokens).

Describe what the element represents (e.g. "an email", "a person") and provide specific details including names, dates, and numbers.

<text>
{{text}}
</text>

Do not include any other text other than the summary in your response.\
"""


TEXT_TITLE_PROMPT = f"""\
Please provide a title for the following subsection from a larger corpus of text in 1-2 sentences (less than {{summarization_length}} tokens).

For example:
Email sent by John Doe [from: john.doe@example.com, to: jane.smith@example.com]
Chapter 3: "The History of the Future" [author: "John Doe"]

<text>
{{text}}
</text>

Do not include any other text other than the title in your response.\
"""

LIST_TITLE_PROMPT = f"""\
Please provide a brief title for the following data. You can include metadata in brackets.

For example:
Emails sent by John Doe [address: john.doe@example.com]
Chapters from "The History of the Future" [author: "John Doe"]

<text>
{{text}}
</text>

Do not include any other text other than the title in your response.\
"""

DICT_TITLE_PROMPT = f"""\
Please provide a brief title for the following data. You can include metadata in brackets.

For example:
Email from John Doe to Jane Smith [date: 2021-01-01] [subject: "Meeting next week"]
Chapter 3: "The History of the Future" [author: "John Doe"]

<text>
{{text}}
</text>

Do not include any other text other than the title in your response.\
"""

PROMPTS = {
    "text": TEXT_TITLE_PROMPT,
    "list": LIST_TITLE_PROMPT,
    "dict": DICT_TITLE_PROMPT,
}



def summarize_context_tree(
    tokenizer: AutoTokenizer,
    client: Client,
    tree: ContextTree,
    min_tokens_to_summarize: int = 512,
    max_tokens_to_summarize: int = 8192,
    max_tokens_in_summary: int = 128,
    downward_pass: bool = False,
    temperature: float = 0.0,
) -> str:
    tree = deepcopy(tree)
    while tree.summary is None:
        # force the model to summarize the root node no matter how short it is
        requests = _get_summarization_requests(
            node=tree, 
            tokenizer=tokenizer,
            min_tokens_to_summarize=min_tokens_to_summarize,
            max_tokens_to_summarize=max_tokens_to_summarize,
            force=True
        )

        requests_needing_summarization: list[SummarizationRequest] = []
        for request in requests:

            if (
                request.force or 
                request.type != "text" or
                len(tokenizer.encode(request.text)) > min_tokens_to_summarize
            ):
                requests_needing_summarization.append(request)
            else:
                # we support breaking up the list into chunks if it's too long 
                if request.type == "list": 
                    if request.node.summary is None:
                        request.node.summary = []
                    request.node.summary.append((request.start_idx, request.end_idx, request.text))
                else:
                    request.node.summary = request.text


        if requests_needing_summarization:
            kwargs = dict(
                chats=[
                    [
                        {
                            "role": "user",
                            "content": PROMPTS[request.type].format(
                                text=request.text,
                                summarization_length=max_tokens_in_summary,
                            ),
                        }
                    ]
                    for request in requests_needing_summarization
                ],
                max_completion_tokens=max_tokens_in_summary,
                temperature=temperature,
            )
            if isinstance(client, TokasaurusBatchClient):
                response = client.chat_with_logprobs(**kwargs)
                responses = [sample.assistant_text for sample in response]
            else:
                response = client.chat(**kwargs)
                responses = [sample.text for sample in response.samples]
        
            for response, request in zip(responses, requests_needing_summarization):
                if request.type == "list": 
                    if request.node.summary is None:
                        request.node.summary = []
                    request.node.summary.append((request.start_idx, request.end_idx, response))
                else:
                    request.node.summary = response
        
        # merge the list summaries into a single string
        for request in requests:
            if isinstance(request.node.summary, list):
                if len(request.node.summary) == 1:
                    request.node.summary = request.node.summary[0][-1]
                else:
                    summary = ""
                    for start_idx, end_idx, text in request.node.summary:
                        summary += f"[Summary of elements {start_idx} through {end_idx}] {text}\n\n"
                    request.node.summary = summary
    if downward_pass:
        tree = _summarize_downward(tree, client=client, max_tokens_in_summary=max_tokens_in_summary)
    return tree

@dataclass
class SummarizationRequest:
    text: str
    node: ContextTree

    type: Literal["text", "list", "dict"]

    # used for very long list summaries
    start_idx: int | None = None
    end_idx: int | None = None

    # force the model to summarize, no matter how short it is
    force: bool = False


def _get_summarization_requests(
    node: ContextTree,
    tokenizer: AutoTokenizer,
    requests: Optional[list[SummarizationRequest]] = None,  
    min_tokens_to_summarize: int = 512,
    max_tokens_to_summarize: int = 8192,
    force: bool = False,
) -> list[SummarizationRequest]:
    if requests is None:
        requests = []
    kwargs = {
        "tokenizer": tokenizer,
        "min_tokens_to_summarize": min_tokens_to_summarize,
        "max_tokens_to_summarize": max_tokens_to_summarize,
        "requests": requests,
    }


    if isinstance(node, ContextTreeLeaf):
        if node.summary is None:
            requests.append(SummarizationRequest(text=node.value, node=node, type="text", force=force))
            return requests
        else:
            return requests
        
    elif isinstance(node, ContextTreeDictNode):
        ready = True
        data = {}
        for child in node.value.values():  
            if child.summary is None:
                ready = False
                _get_summarization_requests(child, **kwargs)
            else:
                data[child.path()] = child.summary
        
        if ready:
            requests.append(SummarizationRequest(text=str(data), node=node, type="dict", force=force))
    
    elif isinstance(node, ContextTreeListNode):
        ready = True
        data = []
        for child in node.value:
            if isinstance(child, ContextTreeLeaf):
                if child.summary is not None:
                    data.append(child.summary)
                elif  child.num_tokens > min_tokens_to_summarize:
                    ready = False
                    _get_summarization_requests(child, **kwargs)
                else:
                    data.append(child.value)
            else:
                if child.summary is not None:
                    data.append(child.summary)
                else:
                    ready = False
                    _get_summarization_requests(child, **kwargs)
                
        if data and ready:
            curr_start_idx = 0
            curr_num_tokens = 0
            curr_data = []
            
            for idx, elem in enumerate(data):
                num_tokens = len(tokenizer.encode(elem))
                
                if curr_data and (curr_num_tokens + num_tokens > max_tokens_to_summarize):
                    requests.append(
                        SummarizationRequest(
                            text=str(curr_data),
                            node=node,
                            type="list",
                            start_idx=curr_start_idx,
                            end_idx=idx,
                            force=force,
                        )
                    )
                    curr_start_idx = idx
                    curr_num_tokens = num_tokens
                    curr_data = [elem]
                else:
                    curr_num_tokens += num_tokens
                    curr_data.append(elem)
            
            if curr_data:
                requests.append(
                    SummarizationRequest(
                        text=str(curr_data),
                        node=node,
                        type="list",
                        start_idx=curr_start_idx,
                        end_idx=len(data),
                        force=force,
                    )
                )
    else:
        raise ValueError(f"Unknown node type: {type(node)}")
    
    return requests


TEMPLATE = """\
Below we provide a section from a larger corpus of text as well as the path to that section from the root of the corpus.

<path-to-section>
{toc_str}
</path-to-section>

<corpus-text>
{corpus_text}
</corpus-text>

<path-to-section>
{toc_str}
</path-to-section>

Please write a succinct sentence describing the section including details like names, dates, and numbers.
For example:
"This is an email [From: John Doe, To: Jane Smith, Date: 2021-01-01] from a large corpus of emails from the Enron corporation."
"This is the 3rd chapter [Title: The fall of the Roman Empire, idx: 2] about the fall of the Roman Empire from a textbook on European History [Author: Chrisopher Clark, Title: The History of Europe]".
"""


def _summarize_downward(
    tree: ContextTree,
    client: Client,
    max_tokens_in_summary: int = 128,
    temperature: float = 0.0,
) -> ContextTree:
    leaves = tree.leaves()

    # (1) Collect all the requests for summaries so we can launch in batch
    # --- begin collect requests ---
    t0 = time.time()
    requests: str = []
    for leaf in leaves:
        path: list[Tuple[str, ContextTree]] = []
        node: ContextTree = leaf 
        while node.parent_data is not None:
            key = node.parent_data.key if isinstance(node.parent_data, DictParent) else node.parent_data.index
            path.append((key, node))
            node = node.parent_data.parent
        path.reverse()

        path_str = "root"
        toc_strs = []
        for idx, (key, node) in enumerate(path):
            toc_strs.append(f"\t" * idx + f"`{node.path()}`: {node.summary}")        
        prompt = TEMPLATE.format(corpus_text=leaf.value, toc_str="\n".join(toc_strs))
        requests.append(prompt)
    logger.info(f"Collected {len(requests)} requests in {time.time() - t0} seconds")
    # --- end collect requests ---

    # (2) Launch the requests in batch
    # --- begin launch requests ---
    t0 = time.time()
    kwargs = dict(
        chats=[
            [
                {
                    "role": "user",
                    "content": request,
                }
            ]
            for request in requests
        ],
        max_completion_tokens=max_tokens_in_summary,
        temperature=temperature,
    )
    if isinstance(client, TokasaurusBatchClient):
        response = client.chat_with_logprobs(**kwargs)
        responses = [sample.assistant_text for sample in response]
    else:
        response = client.chat(**kwargs)
        responses = [sample.text for sample in response.samples]
    logger.info(f"Launched {len(requests)} requests in {time.time() - t0} seconds")
    # --- end launch requests ---
    
    # (3) Update the summaries with the results 
    # --- begin update summaries ---
    t0 = time.time()
    for response, node in zip(responses, leaves):
        node.summary = response
    logger.info(f"Updated {len(requests)} summaries in {time.time() - t0} seconds")
    # --- end update summaries ---

    return tree
