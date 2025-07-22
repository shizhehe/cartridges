
def short_model_name(model_name: str) -> str:
    model_name = model_name.lower()
    model_name = model_name.split("/")[-1]
    if model_name.endswith("-instruct"):
        model_name = model_name[:-len("-instruct")]
    return model_name