from dataclasses import dataclass
import hashlib
import json
import string
from pydrantic import BaseConfig
import random


class CodeHopSynthConfig(BaseConfig):
    seed: int
    num_files: int
    num_methods_per_file: int
    method_name_length: int
    deepest_call_chain: int

    input_vocab_size: int
    output_vocab_size: int
    function_name_vocab_size: int

    def hash(self):
        return hashlib.md5(json.dumps(self.model_dump()).encode()).hexdigest()[:6]


random_alpha_string = lambda length: "".join(
    random.choices(string.ascii_letters, k=length)
).lower()

# random_alpha_string = lambda length: random.choice(
#     ['apple', 'ball', 'cat', 'dog', 'eat', 'fly', 'good', 'happy', 'ice', 'jump', 'kite', 'lion', 'moon', 'nest', 'open', 'play', 'quiet', 'run', 'sun', 'tree', 'use', 'vase', 'walk', 'xray', 'yes', 'zoo', 'act', 'add', 'ask', 'big', 'boy', 'buy', 'call', 'car', 'city', 'cold', 'come', 'cook', 'cry', 'cut', 'dark', 'day', 'deep', 'draw', 'dream', 'drink', 'dry', 'duck', 'dust', 'each', 'easy', 'egg', 'end', 'eye', 'face', 'fall', 'far', 'fast', 'fat', 'few', 'fill', 'find', 'fine', 'fire', 'fish', 'five', 'flat', 'food', 'foot', 'four', 'free', 'fresh', 'frog', 'full', 'game', 'gate', 'give', 'glad', 'glass', 'gold', 'gray', 'green', 'grow', 'hair', 'half', 'hand', 'hang', 'hard', 'hate', 'have', 'head', 'hear', 'heavy', 'help', 'here', 'high', 'hold', 'home', 'hope', 'hot']
# )

vocab = [
    "apple",
    "ball",
    "cat",
    "dog",
    "eat",
    "fly",
    "good",
    "happy",
    "ice",
    "jump",
    "kite",
    "lion",
    "moon",
    "nest",
    "open",
    "play",
    "quiet",
    "run",
    "sun",
    "tree",
    "use",
    "vase",
    "walk",
    "xray",
    "yes",
    "zoo",
    "act",
    "add",
    "ask",
    "big",
    "boy",
    "buy",
    "call",
    "car",
    "city",
    "cold",
    "come",
    "cook",
    "cry",
    "cut",
    "dark",
    "day",
    "deep",
    "draw",
    "dream",
    "drink",
    "dry",
    "duck",
    "dust",
    "each",
    "easy",
    "egg",
    "end",
    "eye",
    "face",
    "fall",
    "far",
    "fast",
    "fat",
    "few",
    "fill",
    "find",
    "fine",
    "fire",
    "fish",
    "five",
    "flat",
    "food",
    "foot",
    "four",
    "free",
    "fresh",
    "frog",
    "full",
    "game",
    "gate",
    "give",
    "glad",
    "glass",
    "gold",
    "gray",
    "green",
    "grow",
    "hair",
    "half",
    "hand",
    "hang",
    "hard",
    "hate",
    "have",
    "head",
    "hear",
    "heavy",
    "help",
    "here",
    "high",
    "hold",
    "home",
    "hope",
    "hot",
]


@dataclass
class MethodCall:
    file: str | None  # None indicates it's a local method call
    method: str
    method_obj: "Method"

    def __eq__(self, other):
        return isinstance(other, MethodCall) and self.file == other.file and self.method == other.method


@dataclass
class LiteralStr:
    content: str

    def __eq__(self, other):
        return isinstance(other, LiteralStr) and self.content == other.content


@dataclass
class InputStr: ...


@dataclass
class Method:
    name: str

    cond_eq: str

    case_true_return_value: MethodCall | LiteralStr
    case_false_return_value: MethodCall | LiteralStr

    call_chain_depth: int = 0

    def call(self, x):
        # return_val = self.case_true_return_value if x[:1] == self.cond_input_prefix else self.case_false_return_value
        return_val = self.case_true_return_value
        if x == self.cond_eq:
            return_val = self.case_true_return_value
        else:
            return_val = self.case_false_return_value

        if isinstance(return_val, MethodCall):
            return return_val.method_obj.call(x)
        elif isinstance(return_val, LiteralStr):
            return return_val.content

        assert False


@dataclass
class CodeHopFile:
    name: str
    methods: list[Method]


def make_return_value(
    local_methods: list[Method],
    methods_other_files: list[tuple[Method, CodeHopFile]],
    output_vocab: list[str],
) -> tuple[MethodCall | LiteralStr, int]:
    return_values_choices = [
        "literal",
    ]

    # if len(local_methods):
    #     return_values_choices.append("local_method_call")
    if len(methods_other_files):
        return_values_choices.append("other_file_method_call")

    call_chain_depth = 0
    return_value_type = random.choice(return_values_choices)
    if return_value_type == "literal":
        return LiteralStr(content=random.choice(output_vocab)), 0

    if return_value_type == "local_method_call":
        method = random.choice(local_methods)
        call_chain_depth = max(method.call_chain_depth + 1, call_chain_depth)

        return (
            MethodCall(
                file=None,
                method=method.name,
                method_obj=method,
            ),
            1,
        )

    assert len(methods_other_files) > 0, "No candidate files available for method call"
    method, file = random.choice(methods_other_files)
    call_chain_depth = max(method.call_chain_depth + 1, call_chain_depth)

    return (
        MethodCall(
            file=file.name,
            method=method.name,
            method_obj=method,
        ),
        call_chain_depth,
    )


@dataclass
class CodeHop:
    files: list[CodeHopFile]
    input_vocab: set[str]
    output_vocab: set[str]


def make_code_hop(
    config: CodeHopSynthConfig,
) -> CodeHop:
    random.seed(config.seed)

    output_vocab = random.sample(vocab, config.output_vocab_size)
    input_vocab = random.sample(vocab, config.input_vocab_size)
    method_names = random.sample(vocab, config.function_name_vocab_size)
    files: list[CodeHopFile] = []
    file_names = set()

    for _ in range(config.num_files):

        this_function_method_names = list(method_names)

        random.shuffle(this_function_method_names)
        methods: list[Method] = []

        available_methods_other_files = [
            (method, file)
            for file in files
            for method in file.methods
            if method.call_chain_depth < config.deepest_call_chain
        ]

        for method_name, _ in zip(
            this_function_method_names, range(config.num_methods_per_file)
        ):

            available_methods_this_file = [
                method
                for method in methods
                if method.call_chain_depth < config.deepest_call_chain
            ]

            return_values_true, true_call_chain_depth = make_return_value(
                local_methods=available_methods_this_file,
                methods_other_files=available_methods_other_files,
                output_vocab=output_vocab,
            )
            for _ in range(1000):
                return_values_false, false_call_chain_depth = make_return_value(
                    local_methods=available_methods_this_file,
                    methods_other_files=available_methods_other_files,
                    output_vocab=output_vocab,
                )
                if return_values_false != return_values_true:
                    break
            else:
                raise ValueError("Could not find a different return value")


            methods.append(
                Method(
                    name=method_name,
                    cond_eq=random.choice(input_vocab),
                    case_true_return_value=return_values_true,
                    case_false_return_value=return_values_false,
                    call_chain_depth=max(true_call_chain_depth, false_call_chain_depth),
                )
            )

        for _ in range(100):
            file_name = random_alpha_string(8)
            if file_name not in file_names:
                break
        else:
            raise ValueError("Could not find a unique file name after 100 tries")

        file = CodeHopFile(
            name=file_name,
            methods=methods,
        )
        file_names.add(file_name)
        files.append(file)
    return CodeHop(files=files, input_vocab=input_vocab, output_vocab=output_vocab)


def serialize_output(output: MethodCall | LiteralStr) -> str:
    return (
        f'"{output.content}"'
        if isinstance(output, LiteralStr)
        else (
            f"{output.file}.{output.method}(x)"
            if output.file is not None
            else f"{output.method}(x)"
        )
    )


def serialize_file(file: CodeHopFile):
    imports = set()

    for method in file.methods:
        for return_val in [
            method.case_true_return_value,
            method.case_false_return_value,
        ]:
            if isinstance(return_val, MethodCall):
                if return_val.file is not None:
                    imports.add(return_val.file)

    method_strings = []
    for method in file.methods:
        #         method_strings.append(
        #             f"""
        # def {method.name}(x):
        #     return {serialize_output(method.case_true_return_value)}
        # """
        #         )

        #         method_strings.append(
        #             f"""
        # def {method.name}(x):
        #     if x[:1] == "{method.cond_input_prefix}":
        #         return {serialize_output(method.case_true_return_value)}
        #     else:
        #         return {serialize_output(method.case_false_return_value)}
        # """
        #         )

        method_strings.append(
            f"""def {method.name}(x):
    if x == "{method.cond_eq}":
        return {serialize_output(method.case_true_return_value)}
    else:
        return {serialize_output(method.case_false_return_value)}
    """)

    return f"""{"\n".join([f'import {mod}' for mod in imports])}
{"\n\n".join(method_strings)}
"""
