import random
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel
from datasets import load_dataset

from cartridges.structs import Context, Document
from cartridges.datasets import CartridgeTrainDataset, CartridgeGenerateDataset, CartridgeGenerateDatasetElement, TEMPLATE
from cartridges.context import BaseContextConfig
from cartridges.utils import get_logger

logger = get_logger(__name__)

class EnronEmail(BaseModel):
    email_id: str
    content: str

class EnronDataset(BaseModel):
    dataset_id: str
    emails: Dict[str, EnronEmail]

def load_enron_dataset(
    split: str = "train",
    max_samples: Optional[int] = None
) -> List[EnronDataset]:
    # Load dataset from Hugging Face
    dataset = load_dataset("LLM-PBE/enron-email", split=split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    datasets = []
    
    # Process Enron emails
    for idx, item in enumerate(dataset):
        email = EnronEmail(
            email_id=str(idx),
            content=item.get('text', '')
        )
        
        enron_dataset = EnronDataset(
            dataset_id=str(idx),
            emails={
                "email": email
            }
        )
        datasets.append(enron_dataset)

    return datasets

class EnronContextConfig(BaseContextConfig):
    split: str = "train"
    max_samples: Optional[int] = None

    def instantiate(self) -> Context:
        datasets = load_enron_dataset(
            self.split,
            self.max_samples
        )
        documents = [
            Document(
                title=f"Email {email.email_id}",
                path=f"{dataset.dataset_id}/{email.email_id}.txt",
                content=email.content
            )
            for dataset in datasets
            for email_id, email in dataset.emails.items()
        ]

        context = Context(
            title=f"Enron-Emails",
            documents=documents,
        )
        print("documents[0]", documents[0])
        print("context", context)
        return context 