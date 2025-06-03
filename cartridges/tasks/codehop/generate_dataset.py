

from textwrap import dedent
from typing import Dict, Optional, List, Tuple

from pydrantic import ObjectConfig
from transformers import PreTrainedTokenizerFast

from capsules.datasets import CapsuleGenerateDataset, CapsuleGenerateDatasetElement, TEMPLATE
from capsules.tasks.codehop.code_hop_synth import CodeHopFile, CodeHopSynthConfig, make_code_hop

class CodeHopGenerateDataset(CapsuleGenerateDataset):
    class Config(ObjectConfig):
        _pass_as_config = True
        code_hop_config: CodeHopSynthConfig

        
        
    def __init__(
        self, 
        config: Config, 
        tokenizer: PreTrainedTokenizerFast
    ):
        self.config = config
        self.tokenizer = tokenizer

        code_hop = make_code_hop(config.code_hop_config)
        files = code_hop.files

        questions = []
        for file in files:
            for method in file.methods:
                for vocab_word in code_hop.input_vocab:
                    question = dedent(f"""\
                        Please tell me the string output of running the following python code.
                        Respond with just a literal string in quotes.
                        Do not include any other text.
                        
                        ```
                        import {file.name}

                        print({file.name}.{method.name}("{vocab_word}"))
                        ```"""
                    )
            
                    answer = method.call(vocab_word)
                    questions.append((question, answer))
        self.questions = questions

    
    def __getitem__(
        self, index: int
    ) -> CodeHopFile:
        question, answer = self.questions[index]
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=TEMPLATE,
        )
        return CapsuleGenerateDatasetElement(
            input_ids=input_ids,
            answer=answer,
            convo_id=f"codehop_{index}",
            metadata={"idx": index},
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