from dataclasses import dataclass
import string
from pydrantic import BaseConfig
import random


class CodeHopSynthConfig(BaseConfig):
    seed: int
    num_files: int
    num_methods_per_file: int
    method_name_length: int
    deepest_call_chain: int
    concat_string_length: int
    file_name_length: int
    num_args: int = 3


# random_alpha_string = lambda length: "".join(
#     random.choices(string.ascii_letters, k=length)
# ).lower()

random_alpha_string = lambda length: random.choice(
    ['apple', 'ball', 'cat', 'dog', 'eat', 'fly', 'good', 'happy', 'ice', 'jump', 'kite', 'lion', 'moon', 'nest', 'open', 'play', 'quiet', 'run', 'sun', 'tree', 'use', 'vase', 'walk', 'xray', 'yes', 'zoo', 'act', 'add', 'ask', 'big', 'boy', 'buy', 'call', 'car', 'city', 'cold', 'come', 'cook', 'cry', 'cut', 'dark', 'day', 'deep', 'draw', 'dream', 'drink', 'dry', 'duck', 'dust', 'each', 'easy', 'egg', 'end', 'eye', 'face', 'fall', 'far', 'fast', 'fat', 'few', 'fill', 'find', 'fine', 'fire', 'fish', 'five', 'flat', 'food', 'foot', 'four', 'free', 'fresh', 'frog', 'full', 'game', 'gate', 'give', 'glad', 'glass', 'gold', 'gray', 'green', 'grow', 'hair', 'half', 'hand', 'hang', 'hard', 'hate', 'have', 'head', 'hear', 'heavy', 'help', 'here', 'high', 'hold', 'home', 'hope', 'hot']
)


@dataclass
class MethodCall:
    file: str | None  # None indicates it's a local method call
    method: str
    method_obj: "Method"


@dataclass
class LiteralStr:
    content: str


@dataclass
class InputStr: ...


@dataclass
class Method:
    name: str

    cond_input_prefix: str

    case_true_return_value: list[MethodCall | LiteralStr | InputStr]
    case_false_return_value: list[MethodCall | LiteralStr | InputStr]

    call_chain_depth: int = 0

    def call(self, x):
        # return_val = self.case_true_return_value if x[:1] == self.cond_input_prefix else self.case_false_return_value
        return_val = self.case_true_return_value

        res = []

        for item in return_val:
            if isinstance(item, MethodCall):
                res.append(item.method_obj.call(x))
            elif isinstance(item, LiteralStr):
                res.append(item.content)
            elif isinstance(item, InputStr):
                res.append(x)

        return " ".join(res)


@dataclass
class CodeHopFile:
    name: str
    methods: list[Method]


def make_return_value(
    local_methods: list[Method],
    methods_other_files: list[tuple[Method, CodeHopFile]],
    num_elements: int,
    concat_string_length: int,
) -> tuple[list[MethodCall | LiteralStr | InputStr], int]:
    return_values_choices = [
        "literal",
    ]

    # if len(local_methods):
    #     return_values_choices.append("local_method_call")
    if len(methods_other_files):
        return_values_choices.append("other_file_method_call")

    return_values = []
    call_chain_depth = 0
    for return_value_type in random.choices(
        return_values_choices, k=(num_elements - 1)
    ):
        if return_value_type == "literal":
            return_values.append(
                LiteralStr(content=random_alpha_string(concat_string_length))
            )
            continue

        if return_value_type == "local_method_call":
            method = random.choice(local_methods)
            call_chain_depth = max(method.call_chain_depth + 1, call_chain_depth)

            return_values.append(
                MethodCall(
                    file=None,
                    method=method.name,
                    method_obj=method,
                )
            )

            continue

        assert (
            len(methods_other_files) > 0
        ), "No candidate files available for method call"
        method, file = random.choice(methods_other_files)
        call_chain_depth = max(method.call_chain_depth + 1, call_chain_depth)

        return_values.append(
            MethodCall(
                file=file.name,
                method=method.name,
                method_obj=method,
            )
        )

    return_values.append(InputStr())
    random.shuffle(return_values)
    return return_values, call_chain_depth


def make_code_hop(
    config: CodeHopSynthConfig,
):
    random.seed(config.seed)

    files: list[CodeHopFile] = []

    method_names = set()
    for _ in range(config.num_methods_per_file):
        for _ in range(100):
            file_name = random_alpha_string(config.file_name_length)
            if file_name not in method_names:
                break
        else:
            raise ValueError("Could not find a unique file name after 100 tries")

        method_names.add(file_name)

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

        for method_name in this_function_method_names:

            available_methods_this_file = [
                method
                for method in methods
                if method.call_chain_depth < config.deepest_call_chain
            ]

            return_values_left, left_call_chain_depth = make_return_value(
                local_methods=available_methods_this_file,
                methods_other_files=available_methods_other_files,
                num_elements=config.num_args,
                concat_string_length=config.concat_string_length,
            )
            return_values_right, right_call_chain_depth = make_return_value(
                local_methods=available_methods_this_file,
                methods_other_files=available_methods_other_files,
                num_elements=config.num_args,
                concat_string_length=config.concat_string_length,
            )
            conditional_case = random_alpha_string(1)

            methods.append(
                Method(
                    name=method_name,
                    cond_input_prefix=conditional_case,
                    case_true_return_value=return_values_left,
                    case_false_return_value=return_values_right,
                    call_chain_depth=max(left_call_chain_depth, right_call_chain_depth),
                )
            )

        file = CodeHopFile(
            name=random_alpha_string(config.file_name_length),
            methods=methods,
        )
        files.append(file)
    return files


def serialize_output(output: list[MethodCall | LiteralStr | InputStr]) -> str:
    return " + ' ' + ".join(
        [
            (
                f'"{val.content}"'
                if isinstance(val, LiteralStr)
                else (
                    "x"
                    if isinstance(val, InputStr)
                    else (
                        f"{val.file}.{val.method}(x)"
                        if val.file is not None
                        else f"{val.method}(x)"
                    )
                )
            )
            for val in output
        ]
    )


def serialize_file(file: CodeHopFile):
    imports = set()

    for method in file.methods:
        for return_val in (
            method.case_true_return_value + method.case_false_return_value
        ):
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
                f"""
    def {method.name}(x):
        return {serialize_output(method.case_true_return_value)}
    """
            )

    

    return f"""{"\n".join([f'import {mod}' for mod in imports])}

{"\n\n".join(method_strings)}
"""
