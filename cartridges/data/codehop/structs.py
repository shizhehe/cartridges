from dataclasses import dataclass


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


@dataclass
class CodeHop:
    files: list[CodeHopFile]
    input_vocab: set[str]
    output_vocab: set[str]

