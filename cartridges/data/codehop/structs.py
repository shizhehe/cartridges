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


    mapping: dict[str, MethodCall | LiteralStr]

    call_chain_depth: int = 0

    def call(self, x):
        return_val = self.mapping[x]

        if isinstance(return_val, MethodCall):
            return return_val.method_obj.call(x, )
        elif isinstance(return_val, LiteralStr):
            return return_val.content
        else:
            raise ValueError(f"Invalid type for return value: {type(return_val)}")
        
    def call_with_depth(self, x) -> tuple[str, int]:
        return_val = self.mapping[x]
        if isinstance(return_val, MethodCall):
            val, depth = return_val.method_obj.call_with_depth(x) 
            return val, depth + 1
        elif isinstance(return_val, LiteralStr):
            return return_val.content, 1
        else:
            raise ValueError(f"Invalid type for return value: {type(return_val)}")

@dataclass
class CodeHopFile:
    name: str
    methods: list[Method]


@dataclass
class CodeHop:
    files: list[CodeHopFile]
    vocab: list[str]

