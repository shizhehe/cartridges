import json
from typing import List, Optional, Tuple, Dict
import random

from pydrantic import ObjectConfig
from transformers import PreTrainedTokenizerFast

from cartridges.data.ruler.niah import NIAHConfig, NIAHQuery, NIAHSample
from cartridges.datasets import CartridgeGenerateDataset, CartridgeGenerateDatasetElement
from cartridges.data.longhealth.utils import LongHealthQuestion, LongHealthPatient, load_longhealth_dataset
from cartridges.initialization.tokenization_utils import MODEL_TO_CHAT_TEMPLATE, MODELS_WITH_THINKING





class NIAHGenerateDataset(CartridgeGenerateDataset):
    class Config(ObjectConfig):
        _pass_as_config = True
        niah_path: Optional[str] = None
        sample_idx: int = 0
        thinking: bool = True

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast, seed: int):
        self.config = config
        self.tokenizer = tokenizer


        with open(self.config.niah_path, "r") as f:
            self.data = json.load(f)
        
        # self.niah_config = NIAHConfig(**self.data["config"])

        sample = self.data["samples"][self.config.sample_idx]
        self.sample = NIAHSample(
            context=sample["context"],
            queries=[NIAHQuery(**query) for query in sample["queries"]]
        )
        self.queries = self.sample.queries
    

        self.tokenizer = tokenizer


    def __getitem__(
        self, index: int
    ) -> CartridgeGenerateDatasetElement:
        # convo: ContextConvo = ContextConvo.model_validate(self.data[index])
        queries: NIAHQuery = self.queries[index]

        kwargs = {}
        if self.tokenizer.name_or_path in MODELS_WITH_THINKING:
            kwargs["enable_thinking"] = self.config.thinking
        elif self.config.thinking:
            cot_prompt = "Think before responding. Put your chain of thought between the <thinking> and </thinking> tags before providing your answer."
        else:
            cot_prompt = ""

        answer_prompt = queries.answer_prompt
        if len(queries.answers) > 1:
            answer_prompt = answer_prompt.replace("The special magic", f"The {len(queries.answers)} different special magic")
            duplicate_prompt = "Do not output the same value twice."
        else:
            duplicate_prompt = ""

        prompt = f"{queries.query}\n\n{cot_prompt}{answer_prompt}{duplicate_prompt}"

        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=MODEL_TO_CHAT_TEMPLATE.get(self.tokenizer.name_or_path, None),
            **kwargs,  
        )


        return CartridgeGenerateDatasetElement(
            input_ids=input_ids,
            prompt=prompt,
            answer=queries.answers,
            convo_id=index,
            metadata={"idx": index}
        )

    def __len__(self):
        return len(self.queries)

    def score(
        self,
        pred: str,
        answer: List[str],
        convo_id: str
    ) -> Tuple[bool, Dict[str, Optional[str]]]:
        
        pred_answers = pred.split(":")[-1].strip("{}'\" .,\t\n")

        if len(answer) == 1:
            correct = str(answer[0]) == str(pred_answers)
        else:
            pred_answers = set([a.strip() for a in pred_answers.split(",")])
            answers = set(str(a) for a in answer)
            correct = pred_answers == answers

        return correct, {"pred_answers": str(pred_answers)}



        
            
