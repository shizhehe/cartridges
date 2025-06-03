
from capsules.context import StructuredContext

def indent(text: str) -> str:
    return text.replace("\n", "\n\t")

MAX_LIST_LENGTH = 32

class SummarizationPromise:
    def __init__(
        self, 
        text: str
    ):
        self.text = text

    def __str__(self):
        return self.text


MAX_TEXT_LENGTH = 32
def summarize_text(text: str) -> str:
    if len(text) > MAX_TEXT_LENGTH:
        return text[:MAX_TEXT_LENGTH].replace("\n", " ") + "..."
    else:
        return text

def get_base_type_outline(obj: str | int | float):
	if isinstance(obj, str):
		if len(obj) > 30:
			return summarize_text(obj)
		else:
			return obj
	else:
		return str(obj)

def get_list_outline(obj: list):
    outlines = []
    for item in obj:
        if isinstance(item, dict):
            sub_outline = get_dict_outline(item)
        elif isinstance(item, list):
            sub_outline = get_list_outline(item)
        elif isinstance(item, StructuredContext):
            sub_outline = get_outline(item)
        else:
            sub_outline = get_base_type_outline(item)
        outlines.append("\t- " + indent(sub_outline))

    return "List of length " + str(len(obj)) + "\n" + "\n".join(outlines)

def get_dict_outline(obj: dict):
    outline = ""
    for key, value in obj.items():
        sub_outline = get_outline(value)
        outline += f"\t{key}: {indent(sub_outline)}"
    return "Dict of length " + str(len(obj)) + "\n" + outline

def get_context_outline(obj: StructuredContext) -> str:
        title = hasattr(obj, "title") and obj.title or obj.__class__.__name__
        out = f"{title}\n"

        basic_fields, complex_fields, ctx_fields = [], [], []
        for field_name, field in obj.model_fields.items():
            value = getattr(obj, field_name)
            value_outline = get_outline(value)
            field_str = f"\t{field_name}: {indent(value_outline)}"

            if isinstance(value, StructuredContext):
                ctx_fields.append(field_str)
            elif isinstance(value, (dict, list)):
                complex_fields.append(field_str)
            else:
                basic_fields.append(field_str)

        return out + "\n".join(basic_fields + ctx_fields + complex_fields)

def get_outline(obj: StructuredContext | dict | list | str | int | float) -> str:
    if isinstance(obj, StructuredContext):
        outline = get_context_outline(obj)
    elif isinstance(obj, list):
        outline = get_list_outline(obj)
    elif isinstance(obj, dict):
        outline = get_dict_outline(obj)
    else:
        outline = get_base_type_outline(obj)
    return outline
