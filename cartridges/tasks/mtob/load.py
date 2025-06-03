from dataclasses import dataclass
import json
from typing import Dict, List, Optional, Union
from openai import BaseModel
import requests
import zipfile
import io
import os

URL = "https://github.com/lukemelas/mtob/raw/refs/heads/main/dataset-encrypted-with-password-kalamang.zip"

class MTOBExample(BaseModel):
    original_id: Optional[Union[str, int]] = None
    ground_truth: Optional[str] = None
    original: str
    translation: str
    url: str

class MTOBData(BaseModel):
    wordlist_ke: Dict[str, List[str]]
    wordlist_ek: Dict[str, str]
    grammar_book_tex: str
    grammar_book_txt: str

    train_examples: List[MTOBExample]
    test_examples_ek: List[MTOBExample]
    test_examples_ke: List[MTOBExample]
    human_held_out_train_examples_ke: List[MTOBExample]
    human_held_out_train_examples_ek: List[MTOBExample]
    human_test_examples_ke: List[MTOBExample]
    human_test_examples_ek: List[MTOBExample]

def _json_load_and_rm_canary(data: str):
    data = json.loads(data)
    if isinstance(data, list):
        data = [d for d in data if "big-bench-canary" not in d]
    elif isinstance(data, dict) and "big-bench-canary" in data:
        del data["big-bench-canary"]
    return data

def load_mtob():
    pwd = "kalamang".encode()
    response = requests.get(URL)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        wordlists = json.loads(z.read("resources/wordlist.json", pwd=pwd))
        data = MTOBData(
            wordlist_ke=wordlists["ke"],
            wordlist_ek=wordlists["ek"],
            grammar_book_tex=z.read("resources/grammar_book.tex", pwd=pwd),
            grammar_book_txt=z.read("resources/grammar_book.txt", pwd=pwd),
            train_examples=_json_load_and_rm_canary(z.read("splits/train_examples.json", pwd=pwd)),
            test_examples_ek=_json_load_and_rm_canary(z.read("splits/test_examples_ek.json", pwd=pwd)),
            test_examples_ke=_json_load_and_rm_canary(z.read("splits/test_examples_ke.json", pwd=pwd)),
            human_held_out_train_examples_ke=_json_load_and_rm_canary(z.read("splits/human_held_out_train_examples_ke.json", pwd=pwd)),
            human_held_out_train_examples_ek=_json_load_and_rm_canary(z.read("splits/human_held_out_train_examples_ek.json", pwd=pwd)),
            human_test_examples_ke=_json_load_and_rm_canary(z.read("splits/human_test_examples_ke.json", pwd=pwd)),
            human_test_examples_ek=_json_load_and_rm_canary(z.read("splits/human_test_examples_ek.json", pwd=pwd)),
        )
        return data


if __name__ == "__main__":
    print("Start loading MTOB")
    data = load_mtob()
    print("Done loading MTOB")
    breakpoint()

