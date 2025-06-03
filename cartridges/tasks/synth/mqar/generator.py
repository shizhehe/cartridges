from dataclasses import dataclass
import random
import numpy as np
from typing import List
import torch

from cartridges.generate.context_convo_generators.base import ContextConvoGenerator, QuestionData
from cartridges.structs import Context, ContextConvo, Message
from cartridges.tasks.synth.mqar.mqar_data import MQARConfig, multiquery_ar

@dataclass
class MQARGenerator(ContextConvoGenerator):
    class Config(ContextConvoGenerator.Config):
        vocab_size: int
        num_examples: int
        input_seq_len: int
        num_kv_pairs: int
        power_a: float
        random_non_queries: bool
        include_slices: bool

    def __init__(self, config: Config, context: Context):
        super().__init__(config, context)
        self.config = config
        
    def sample_convos(self, start_idx: int, end_idx: int) -> List[ContextConvo]:
        num_samples = end_idx - start_idx
        
        # Create MQARConfig and generate data
        mqar_config = MQARConfig(
            vocab_size=self.config.vocab_size,
            num_examples=num_samples,
            input_seq_len=self.config.input_seq_len,
            power_a=self.config.power_a,
            num_kv_pairs=self.config.num_kv_pairs,
            random_non_queries=self.config.random_non_queries,
            include_slices=self.config.include_slices
        )
        
        # Generate data using the config
        questions = mqar_config.build(seed=start_idx)
        
        convos = []
        for i, question_data in enumerate(questions):
            
            # DZ: (there is duplicate data in here) 
            convo = ContextConvo(
                messages=[
                    Message(role="system", content=question_data.metadata["context"]),
                    Message(role="user", content=question_data.question),
                    Message(role="assistant", content=str(question_data.metadata["value"]))
                ],
                type="mqar",
                id=f"mqar_{start_idx + i}",
                metadata={
                    "vocab_size": self.config.vocab_size,
                    "input_seq_len": self.config.input_seq_len,
                    "num_kv_pairs": self.config.num_kv_pairs,
                    "power_a": self.config.power_a,
                    "random_non_queries": self.config.random_non_queries,
                    "include_slices": self.config.include_slices,
                    **question_data.metadata #DZ: can probably remove this
                }
            )
            convos.append(convo)
            
        return convos 
    
    def get_questions(self, num_samples: int) -> list[QuestionData]:
        # Generate questions using our multiquery_ar function
        questions = multiquery_ar(
            vocab_size=self.config.vocab_size,
            num_examples=self.config.num_examples,
            num_kv_pairs=self.config.num_kv_pairs,
            include_slices=self.config.include_slices,
            seed=42  # Fixed seed for reproducibility
        )

        if len(questions) > num_samples:
            questions = random.sample(questions, num_samples)
            
        return questions