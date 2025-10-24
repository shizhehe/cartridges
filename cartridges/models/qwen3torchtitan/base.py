# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn

from torch.nn.attention.flex_attention import BlockMask


AttentionMasksType = dict[str, BlockMask] | BlockMask

class BaseTokenizer(ABC):
    # base tokenizer interface, for typing purpose mainly
    def __init__(self):
        self.eos_id = 0

    @abstractmethod
    def encode(self, *args, **kwargs) -> list[int]:
        ...

    @abstractmethod
    def decode(self, *args, **kwargs) -> str:
        ...

    @abstractmethod
    def get_vocab_size(self) -> int:
        ...



class ModelProtocol(Protocol):
    """Defines the interface for a model class.

    This is used to enforce that all model classes have some methods that are
    required by the trainer.
    """

    def __init__(self, model_args: BaseModelArgs) -> None:
        pass

    @abstractmethod
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize model weights.

        Args:
            buffer_device: Optional device to place buffers on during initialization.
        """
        pass

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        raise NotImplementedError(
            "This model does not support attention masking/Flex Attention."
        )
