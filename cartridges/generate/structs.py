from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path
import random
import tempfile
from typing import List, Literal, Optional

import numpy as np
import requests
from pydrantic import BaseConfig
from pydantic import BaseModel
import torch.distributed as dist
import pandas as pd
from tqdm import tqdm
import wandb

from capsules.clients.base import InputToken, Sample, Client, ClientConfig

from capsules.utils import get_logger

import pyarrow as pa

from capsules.utils.wandb import download_artifact, get_artifact_dir

logger = get_logger(__name__)


class Section(BaseModel):    
    content: str
    desc: str
    path: Optional[str] = None
    type_: Literal["document", "page", "section"] = "document"

    # The number of tokens in the section
    tokens: Optional[int] = None

    def __str__(self):
        return f"""<section desc="{self.desc}">
{self.content}
</section>"""

    @property
    def title(self) -> str:
        import warnings
        warnings.warn("The 'desc' attribute is deprecated, use 'title' instead.", DeprecationWarning)
        return self.desc

    @classmethod
    def from_path_or_url(
        cls, document_path_or_url: str, title: Optional[str] = None
    ) -> "Document":

        if title is None:
            title = document_path_or_url

        # Check if the input is a URL and use requests.get if so
        if document_path_or_url.startswith(("http://", "https://")):
            response = requests.get(document_path_or_url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return cls(
                title=title,
                content=response.text,
                path=document_path_or_url,
            )
        else:
            # Otherwise, read from a local file
            with open(document_path_or_url, "r", encoding="utf-8") as f:
                return cls(title=title, path=document_path_or_url, content=f.read())

    def to_string(self) -> str:
        return f"""<{self.type_} title="{self.title}">
{self.content}
</{self.type_}>"""

# SE (04/15): Section's used to be called Documents
Document = Section

class Context(BaseModel):
    title: str
    sections: list[Section]
    text: Optional[str] = None
    
    @property
    def documents(self) -> list[Section]:
        import warnings
        warnings.warn("This is deprecated, now called sections", DeprecationWarning)
        return self.sections

    def to_string(self) -> str:
        if self.text is not None:
            return self.text
        else:
            return "\n\n".join(doc.to_string() for doc in self.documents)

    @classmethod
    def from_path_or_url(
        cls, document_path_or_url: str, title: Optional[str] = None
    ) -> "Context":
        if not document_path_or_url.startswith(("http://", "https://")):
            if Path(document_path_or_url).exists():
                if Path(document_path_or_url).is_dir():
                    return cls.from_directory(document_path_or_url, title)

        return cls(
            documents=[Document.from_path_or_url(document_path_or_url, title=title)],
            title=title,
        )
    

    @classmethod
    def from_directory(cls, directory: str, title: str) -> "Context":
        documents = []
        for file in os.listdir(directory):
            path = os.path.join(directory, file)
            with open(path, "r", encoding="utf-8") as f:
                documents.append(Document(title=file, content=f.read(), path=path))
        return cls(documents=documents, title=title)

    def to_yaml(self, path: str):
        import yaml

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f)

    @classmethod
    def from_yaml(cls, path: str) -> "Context":
        import yaml
        
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if "documents" in data:
            # SE (04/15): Backwards compatibility with old contexts
            for doc in data["documents"]:
                doc["desc"] = doc["title"]
                del doc["title"]
            data["sections"] = data["documents"]
            del data["documents"]



        return cls.model_validate(data)


SECTION_BUFFER_TOKENS = 50

class SectionedContext(BaseModel):
    title: str
    sections: list[Section]

    def to_string(self):
        return "\n\n".join(str(doc) for doc in self.sections)


class Logprob(BaseModel):
    token_id: int
    logprob: float

class Message(BaseModel):
    content: str
    role: Literal["user", "assistant", "system"]
    
    token_ids: Optional[List[int]] = None
    logprobs: Optional[List[List[Logprob]]] = None

    # SE (04/08): The sample field is deprecated, we use the token_ids and top_logprobs fields instead.
    sample: Optional[Sample] = None

class ContextConvo(BaseModel):
    id: Optional[str] = None
    messages: list[Message]

    type: str
    metadata: dict

    # Optionally include the context on which the assistant and user messages were 
    # conditioned.
    assistant_context: Optional[list[Message]] = None
    user_context: Optional[list[Message]] = None

    # Optionally include the tokenized version


    def _repr_html_(self) -> str:
        import markdown

        html = """
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        <div class='context-convo p-4'>
        """
        for message in self.messages:
            if message.role == "user":
                role_class = "bg-blue-100 text-blue-800"
            else:
                role_class = "bg-green-100 text-green-800"
            role_display = f"<strong style='font-size: 1.5em;'>{message.role.capitalize()}</strong>"
            content_html = markdown.markdown(message.content)
            html += f"""
            <div class='p-2 my-2 rounded {role_class}'>
                {role_display} {content_html}
            </div>
            """
        html += "</div>"
        return html

    def to_html(self) -> str:
        return self._repr_html_()


class LazyContextConvo:
    def __init__(self, table: pa.Table, index: int):
        self._table = table
        self._index = index

        self._row = None

        assert {
            "id",
            "messages",
            "type",
            "metadata",
        }.issubset(set(self._table.schema.names))

    def __getattr__(self, name):
        return self.materialize().__getattribute__(name)

    def materialize(self) -> ContextConvo:
        if self._row is None:
            self._row = ContextConvo.model_validate(
                {
                    name: self._table.column(j)[self._index].as_py()
                    for j, name in enumerate(self._table.schema.names)
                }
            )
        return self._row


@dataclass
class ContextConvoDataset:  # RE: if this is a basemodel, intializing this object is very slow.
    context: Context
    rows: List[
        ContextConvo | dict
    ]  # SE: It's very slow to validate all the rows at once, so we want the option of validating on the fly later

    config: BaseConfig = None

    @staticmethod
    def save_rows(rows: List[ContextConvo | dict], path: str):
        rows = [
            row.materialize() if isinstance(row, LazyContextConvo) else row
            for row in rows
        ]
        # HACK: if we just use .model_dump, pandas mucks with our data and makes
        # it so we can't turn it back into pydantic objects nicely.
        # this is probably handleable, but let's deal with it later.
        df = pd.DataFrame([row.model_dump() for row in rows])
        df.to_feather(path)
        return df

    @staticmethod
    def load_rows(path: Path, lazy: bool = True) -> List[ContextConvo]:
        if str(path).endswith(".pkl"):
            import pickle 
            with open(path, "rb") as source:
                table = pickle.load(source)
        else:
            with pa.memory_map(str(path.absolute()), "r") as source:
                reader = pa.ipc.open_file(source)
                table = reader.read_all()
                return [LazyContextConvo(table, index) for index in range(table.shape[0])]
        return table

    def save(
        self,
        directory: str,
        to_wandb: bool = False,
        wandb_name: str = "dataset",
        save_wandb_preview: bool = False,
    ):
        if os.path.exists(directory):
            raise FileExistsError(f"Directory {directory} already exists")

        os.makedirs(directory)

        self.context.to_yaml(os.path.join(directory, "context.yaml"))
        self.config.to_yaml(os.path.join(directory, "config.yaml"))

        path = os.path.join(directory, "dataset.feather")
        df = self.save_rows(self.rows, path)

        if to_wandb:
            artifact = wandb.Artifact(name=wandb_name, type="dataset")
            artifact.add_dir(local_path=directory, name="dataset")
            wandb.log_artifact(artifact)

            # important to wait for the artifact to be saved so we get the version
            artifact.wait()
            logger.info(f"Saved dataset to wandb as artifact {artifact.name}")

        if save_wandb_preview:
            sampled_rows = random.sample(self.rows, min(256, len(self.rows)))

            sampled_rows = [
                ContextConvo(**row) if isinstance(row, dict) else row
                for row in sampled_rows
            ]
            preview_df = pd.DataFrame(
                [
                    {
                        "question": row.messages[0].content,
                        "answer": row.messages[1].content,
                        "type": row.type,
                        "id": row.id,
                    }
                    for row in sampled_rows
                ]
            )
            wandb.log({"dataset_preview": preview_df})

    @classmethod
    def load(
        cls,
        directory_or_artifact: str,
        is_wandb: bool = False,
    ) -> "ContextConvoDataset":
        """
        Load a dataset from a local feather file or a W&B artifact.
        If loading from W&B, downloads to a temporary directory that's deleted after use.

        Args:
            path: String path to the local feather file or W&B artifact in format
                "entity/project/artifact:alias" or just "artifact:alias" if inside a run
            is_wandb: Boolean flag indicating whether the path is a W&B artifact

        Returns:
            List of DatasetRow objects
        """
        dataset_dir = (
            get_artifact_dir(directory_or_artifact) # / "dataset"
            if is_wandb
            else Path(directory_or_artifact)
        )

        if not dataset_dir.exists() and is_wandb:
            download_artifact(directory_or_artifact)

        try:
            data = cls.load_rows(dataset_dir) 
            context = data['context']
            config = None
        except:
            data = cls.load_rows(dataset_dir/ "dataset/dataset.feather")
            context = Context.from_yaml(dataset_dir / "dataset/context.yaml")
            config = BaseConfig.from_yaml(dataset_dir / "dataset/config.yaml", strict=False)
       
        # Return data - temporary directory will be cleaned up when exiting the context
        return cls(
            context=context,
            rows=data,
            config=config,
        )

@dataclass
class TrainingExample:
    @dataclass
    class Message:
        content: str
        role: Literal["user", "assistant", "system"]

    """
    Attrs:
        messages: conversation as openai text.

        top_logprobs_ids: [num_tokens - 1, k_top_logprobs]
        top_logprobs_logprobs: [num_tokens - 1, k_top_logprobs]
        token_ids: [num_tokens, k_top_logprobs]
        num_output_token_ids: int

        metadata: arbitrary metadata
        type: type of this context convo
    """

    messages: list[TrainingExample.Message]

    num_output_tokens: int
    token_ids: np.ndarray
    top_logprob_ids: np.ndarray
    top_logprob_logprobs: np.ndarray

    type: str
    metadata: dict
    mask: np.ndarray | None = None

    context: str | None = None
