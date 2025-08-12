import abc
from pydrantic import ObjectConfig

from cartridges.structs import Conversation

import numpy as np


class ConvoSynthesizer(abc.ABC):

    class Config(ObjectConfig):
        _pass_as_config: bool = True

    def __init__(self, config: Config):
        self.config = config

    @abc.abstractmethod
    def sample_convos(
        self,
        batch_idx: int,
        num_convos: int,
        total_batches: int,
    ) -> list[Conversation]:
        raise NotImplementedError()

class AsyncConvoSynthesizer(ConvoSynthesizer):


    @abc.abstractmethod
    async def sample_convos(
        self,
        batch_idx: int,
        num_convos: int,
        total_batches: int,
    ) -> list[Conversation]:
        raise NotImplementedError()
