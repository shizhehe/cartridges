
import random

from transformers import AutoTokenizer
from capsules.generate.context_convo_generators.base import ContextConvoGenerator
from capsules.generate.structs import Context, ContextConvo, Message, Document


class NTPContextConvoGenerator(ContextConvoGenerator):

    class Config(ContextConvoGenerator.Config):
        num_sections_per_convo: int = 3
        num_preview_tokens: int = 128
        max_tokens_per_section: int = 1024

        tokenizer: str = "meta-llama/Llama-3.2-3B-Instruct"

    def __init__(self, config: Config, context: Context):
        self.config = config
        self.context = context
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)

        self.tokenized_documents = [
            self.tokenizer.encode(doc.to_string())
            for doc in self.context.documents
        ]

    def sample_convos(self, start_idx: int, end_idx: int) -> list[ContextConvo]:
        return [self._sample_convo(idx) for idx in range(start_idx, end_idx)]
    
    def _sample_convo(self, idx: int) -> ContextConvo:

        sections, preview_sections, documents = zip(
            *[self._sample_section() for _ in range(self.config.num_sections_per_convo)]
        )

        user_prompt = "Please complete the sections of the following document. "
        for i, preview_section in enumerate(preview_sections):
            user_prompt += f"{i+1}. {preview_section}\n"

        response_prompt = ""
        for i, section in enumerate(sections):
            response_prompt += f"{i+1}. {section}\n"

        return ContextConvo(
            messages=[
                Message(
                    role="user",
                    content=user_prompt,
                ),
                Message(
                    role="assistant",
                    content=response_prompt,
                )
            ],
            id=f"programmatic{idx}_{'-'.join([doc.path for doc in documents])}",
            type=f"ntp-nsections{self.config.num_sections_per_convo}-npreview{self.config.num_preview_tokens}-max{self.config.max_tokens_per_section}",
            metadata={}
        )
        

    def _sample_section(self) -> tuple[str, str, Document]:
        idx = random.randint(0, len(self.tokenized_documents) - 1)

        tokenized_document = self.tokenized_documents[idx % len(self.tokenized_documents)]
        document = self.context.documents[idx % len(self.context.documents)]

        if len(tokenized_document) > self.config.max_tokens_per_section:
            random_start_idx = random.randint(0, len(tokenized_document) - self.config.max_tokens_per_section)
            tokenized_document = tokenized_document[random_start_idx:random_start_idx + self.config.max_tokens_per_section]
        section = self.tokenizer.decode(tokenized_document)
        
        preview = self.tokenizer.decode(tokenized_document[:self.config.num_preview_tokens])
        
        return section, preview, document
        
        

