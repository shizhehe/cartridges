from dataclasses import dataclass
import json
import os
import pickle
import random
import string
import hashlib

from pydrantic import RunConfig
from pydrantic.variables import FormatStringVariable

from cartridges.data.codehop.structs import CodeHop, CodeHopFile, Method, MethodCall, LiteralStr


class MakeCodeHopConfig(RunConfig):
    output_dir: str = os.environ.get("CARTRIDGES_OUTPUT_DIR", ".")

    seed: int = 42

    num_files: int=2
    num_methods_per_file: int=8
    deepest_call_chain: int=8

    input_vocab_size: int=128
    output_vocab_size: int=128
    function_name_vocab_size: int=128

    def run(self):
        code_hop = make_code_hop(self)
        return code_hop

    def hash(self):
        
        data = self.model_dump()

        for key in ["output_dir", "run_id", "run_dir", "launch_id", "script_id"]:
            data.pop(key, None)

        return hashlib.md5(json.dumps(data).encode()).hexdigest()[:6]


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


def make_code_hop(
    config: MakeCodeHopConfig,
) -> CodeHop:
    random.seed(config.seed)
    import wonderwords
    nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
    adjs = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")

    def filter_words(word: str) -> bool:
        return (
            " " not in word and
            "-" not in word and
            "." not in word
        )
    nouns = [noun.lower() for noun in nouns if filter_words(noun)]  # remove compound words
    adjs = [adj.lower() for adj in adjs if filter_words(adj)]  # remove compound words
    words = [f"{adj}_{noun}" for adj in adjs for noun in nouns]
    vocab = sorted(list(set(words)))

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
            file_name = random.choice(vocab)
            if file_name not in file_names and file_name not in method_names:
                break
        else:
            raise ValueError("Could not find a unique file name after 100 tries")

        file = CodeHopFile(
            name=file_name,
            methods=methods,
        )
        file_names.add(file_name)
        files.append(file)
    code_hop = CodeHop(files=files, input_vocab=input_vocab, output_vocab=output_vocab)

    config_hash = config.hash()
    repo_dir = os.path.join(config.run_dir, f"repo-{config_hash}")
    os.makedirs(repo_dir, exist_ok=True)
    for file in files:
        with open(os.path.join(repo_dir, f"{file.name}.py"), "w") as f:
            f.write(serialize_file(file))
    dataset_path = os.path.join(config.run_dir, f"dataset-{config_hash}.pkl")
    pickle.dump(code_hop, open(dataset_path, "wb"))
    print(f"CodeHop dataset generated successfully!")
    print(f"Repository files written to: {repo_dir}")
    print(f"Dataset pickle saved to: {dataset_path}")
    print(f"Generated {len(files)} files with {sum(len(file.methods) for file in files)} total methods")
    return code_hop



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


if __name__ == "__main__":
    import pydrantic
    config = MakeCodeHopConfig(
        num_files=4,
        num_methods_per_file=10,
        deepest_call_chain=2,
        input_vocab_size=8,
        output_vocab_size=8,
        function_name_vocab_size=36,
        run_id=FormatStringVariable(
            "codehop-nf{num_files}-nm{num_methods_per_file}-dc{deepest_call_chain}-iv{input_vocab_size}-ov{output_vocab_size}-fn{function_name_vocab_size}"
        )
    )
    pydrantic.main([config])