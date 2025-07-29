import json
from typing import List, Optional, Tuple, Dict
import random

from pydrantic import ObjectConfig
from transformers import PreTrainedTokenizerFast

from cartridges.data.ruler.niah import NIAHConfig, NIAHQuery, NIAHSample
from cartridges.data.ruler.variable_tracking import VariableTrackingConfig, VariableTrackingQuery, VariableTrackingSample
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

        prompt = f"{queries.query}\n\n{queries.answer_prompt}"

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
        
        pred_answers = pred.split(":")[-1].strip("{}'\" ")

        if len(answer) == 1:
            correct = str(answer[0]) == str(pred_answers)
        else:
            pred_answers = set([a.strip() for a in pred_answers.split(",")])
            answers = set(str(a) for a in answer)
            correct = pred_answers == answers

        return correct, {"pred_answers": str(pred_answers)}


class VariableTrackingGenerateDataset(CartridgeGenerateDataset):
    class Config(ObjectConfig):
        _pass_as_config = True
        variable_tracking_path: Optional[str] = None
        sample_idx: int = 0
        thinking: bool = True

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast, seed: int):
        self.config = config
        self.tokenizer = tokenizer

        with open(self.config.variable_tracking_path, "r") as f:
            self.data = json.load(f)
        
        sample = self.data["samples"][self.config.sample_idx]
        self.sample = VariableTrackingSample(
            context=sample["context"],
            queries=[VariableTrackingQuery(**query) for query in sample["queries"]]
        )
        self.queries = self.sample.queries

    def __getitem__(
        self, index: int
    ) -> CartridgeGenerateDatasetElement:
        query: VariableTrackingQuery = self.queries[index]

        kwargs = {}
        if self.tokenizer.name_or_path in MODELS_WITH_THINKING:
            kwargs["enable_thinking"] = self.config.thinking

        # Combine context and query for variable tracking
        full_prompt = f"{self.sample.context}\n\nQuestion: {query.query}\n\n{query.answer_prompt}"

        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": full_prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=MODEL_TO_CHAT_TEMPLATE.get(self.tokenizer.name_or_path, None),
            **kwargs,  
        )

        return CartridgeGenerateDatasetElement(
            input_ids=input_ids,
            prompt=full_prompt,
            answer=query.answers,
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
        
        # Extract predicted variables from the response
        # Look for the part after "they are:" or similar
        pred_lower = pred.lower()
        if "they are:" in pred_lower:
            pred_answers = pred.split("they are:")[-1].strip()
        elif "are:" in pred_lower:
            pred_answers = pred.split("are:")[-1].strip()
        else:
            pred_answers = pred.strip()
        
        # Clean up the prediction - remove punctuation and split by commas
        pred_answers = pred_answers.strip("{}'\".,; ")
        pred_variables = set()
        
        if pred_answers:
            # Split by common delimiters and clean each variable
            for var in pred_answers.replace(",", " ").replace(";", " ").split():
                var = var.strip("{}'\".,; ")
                if var:
                    pred_variables.add(var.upper())
        
        # Convert expected answers to set for comparison
        expected_variables = set(str(var).upper() for var in answer)
        
        # Check if prediction matches expected variables
        correct = pred_variables == expected_variables
        
        return correct, {
            "pred_variables": sorted(list(pred_variables)),
            "expected_variables": sorted(list(expected_variables))
        }
