import re
import json

def extract_last_json(text):
    """Extract and parse the last code block between triple backticks as JSON."""
    pattern = r"```.*?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return json.loads(matches[-1].strip()) if matches else None