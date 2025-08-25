
import os
import pickle
from textwrap import dedent
from typing import Dict, Optional, Tuple

from transformers import PreTrainedTokenizerFast

from cartridges.data.codehop.structs import LiteralStr
from cartridges.datasets import GenerateEvalDataset, GenerateEvalDatasetElement
from cartridges.initialization.tokenization_utils import MODEL_TO_CHAT_TEMPLATE


class CodeHopGenerateDataset(GenerateEvalDataset):
    class Config(GenerateEvalDataset.Config):
        _pass_as_config = True
        make_run_dir: str
        thinking: bool = False

    def __init__(
        self, 
        config: Config, 
        tokenizer: PreTrainedTokenizerFast,
        seed: int
    ):
        self.config = config
        self.tokenizer = tokenizer

        # Load the code hop dataset from the pickle file
        # Find the dataset file with pattern dataset-{hash}.pkl
        import glob
        dataset_pattern = os.path.join(config.make_run_dir, "dataset-*.pkl")
        dataset_files = glob.glob(dataset_pattern)
        if not dataset_files:
            raise FileNotFoundError(f"No dataset pickle file found matching pattern {dataset_pattern}")
        dataset_path = dataset_files[0]  # Take the first match
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset pickle file not found at {dataset_path}")
        
        with open(dataset_path, "rb") as f:
            code_hop = pickle.load(f)
        
        files = code_hop.files

        questions = []
        for file in files:
            for method in file.methods:
                for vocab_word in code_hop.vocab:
                    # question = dedent(f"""\
                    #     Please tell me the string output of running the following python code.
                    #     Respond with just a literal string in quotes.
                    #     Do not include any other text.
                        
                    #     ```
                    #     import {file.name}
                    #     print({file.name}.{method.name}("{vocab_word}"))
                    #     ```"""
                    # )
                    question = dedent(f"""\
                        Please tell me the string output of running the following python code.
                        
                        ```
                        import {file.name}
                        print({file.name}.{method.name}("{vocab_word}"))
                        ```

                        Respond with the following format: 
                        `The output of {file.name}.{method.name}("{vocab_word}") will be \"{{your answer}}\"."""
                    )
            
                    answer, depth = method.call_with_depth(vocab_word)
                    value = method.mapping[vocab_word]
                    meta = {
                        "file_name": file.name,
                        "method_name": method.name,
                        "input_word": vocab_word,
                        "is_literal": isinstance(value, LiteralStr),
                        "depth": depth,
                    }
                    questions.append((question, answer, meta))
        self.questions = questions

    
    def __getitem__(
        self, index: int
    ) -> GenerateEvalDatasetElement:
        question, answer, meta = self.questions[index]
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=MODEL_TO_CHAT_TEMPLATE.get(self.tokenizer.name_or_path, None),
            enable_thinking=self.config.thinking,
        )
        return GenerateEvalDatasetElement(
            input_ids=input_ids,
            answer=answer,
            convo_id=f"codehop_{index}",
            metadata={
                "idx": index,
                **meta,
            },
            prompt=question,
        )

    def __len__(self):
        return len(self.questions)

    def score(
        self,
        pred: str,
        answer: str,
        convo_id: str
    ) -> Tuple[bool, Dict[str, Optional[str]]]:
        return answer.lower() in pred.lower(), {}