import uuid

from transformers import PreTrainedTokenizerFast
from pydrantic import ObjectConfig
import pandas as pd

from capsules.datasets import CapsuleGenerateDataset, CapsuleGenerateDatasetElement, TEMPLATE
from capsules.utils import get_logger

logger = get_logger(__name__)


PROMPT_TEMPLATE = """\
Please answer this multiple choice question, giving only your answer:

{question}"""

class OpenStaxMultipleChoiceGenerateDataset(CapsuleGenerateDataset):
    class Config(ObjectConfig):
        _pass_as_config = True

        # path to the json file
        path: str


    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):
        self.config = config
        
        self.data = pd.read_json(config.path).to_dict(orient="records")

        self.tokenizer = tokenizer
        logger.info("Datasets loaded")


    def __getitem__(
        self, index: int
    ) -> CapsuleGenerateDatasetElement:
        # convo: ContextConvo = ContextConvo.model_validate(self.data[index])
        prompt = PROMPT_TEMPLATE.format(question=self.data[index]["question"])

        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=TEMPLATE,
        )

        return CapsuleGenerateDatasetElement(
            input_ids=input_ids,
            prompt=prompt,
            answer=self.data[index]["answer"],
            convo_id=self.data[index]["number"],
            metadata={"idx": index,}
        )

    def score(
        self,
        pred: str,
        element: CapsuleGenerateDatasetElement,
    ):
        return pred.strip().lower()[0] == element.answer.strip().lower()[0]
