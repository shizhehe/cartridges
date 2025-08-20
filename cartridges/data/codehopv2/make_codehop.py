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

    vocab_size: int=128
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

    vocab = random.sample(vocab, config.vocab_size)
    all_method_names = [
        f"apply_{adj}_random_map"
        for adj in random.sample(adjs, config.function_name_vocab_size)
    ]
    files: list[CodeHopFile] = []
    file_names = set()

    for _ in range(config.num_files):
        method_names = random.sample(all_method_names, config.num_methods_per_file)
        methods: list[Method] = []

        available_methods_other_files = [
            (method, file)
            for file in files
            for method in file.methods
            if method.call_chain_depth < config.deepest_call_chain
        ]

        for method_name in method_names:

            available_methods_this_file = [
                method
                for method in methods
                if method.call_chain_depth < config.deepest_call_chain
            ]

            mapping = {
                input_str: make_return_value(
                    local_methods=available_methods_this_file,
                    methods_other_files=available_methods_other_files,
                    output_vocab=vocab,
                )
                for input_str in vocab
            }

            methods.append(
                Method(
                    name=method_name,
                    mapping={k: v for k, (v, _) in mapping.items()},
                    call_chain_depth=max(call_chain_depth for _, call_chain_depth in mapping.values()),
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
    code_hop = CodeHop(files=files, vocab=vocab)

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

def serialize_method(method: Method):
    out = f"def {method.name}(x):\n"
    for input_str, return_val in method.mapping.items():
        out += f"    if x == {input_str}:\n"
        out += f"        return {serialize_output(return_val)}\n"
    return out + "\n"

def serialize_file(file: CodeHopFile):
    imports = set()

    for method in file.methods:
        for return_val in method.mapping.values():
            if isinstance(return_val, MethodCall):
                if return_val.file is not None:
                    imports.add(return_val.file)

    method_strings = []
    for method in file.methods:
        method_strings.append(serialize_method(method))

    return f"""{"\n".join([f'import {mod}' for mod in imports])}
{"\n\n".join(method_strings)}
"""


if __name__ == "__main__":
    import pydrantic
    config = MakeCodeHopConfig(
        num_files=4,
        num_methods_per_file=10,
        deepest_call_chain=2,
        vocab_size=4,
        function_name_vocab_size=36,
        run_id=FormatStringVariable(
            "codehop-nf{num_files}-nm{num_methods_per_file}-dc{deepest_call_chain}-v{vocab_size}-fn{function_name_vocab_size}"
        )
    )
    pydrantic.main([config])