import abc
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
from venv import logger
from pydrantic import ObjectConfig
import torch
import torch.nn as nn

from typing import Any, Optional

from capsules.generate.structs import Context
from transformers import DynamicCache
from typing import List


@dataclass
class AttnConfig:
    n_layers: int
    n_heads: int
    head_dim: int


# SE (03/24): I had to add nn.Module to the inheritance because I was getting the error
# TrainableCache has no attribute `.to`. Still not sure why this started happening --
# perhaps there was an update to the transformers library.z
class TrainableCacheComposable(DynamicCache, nn.Module):

    def __init__(
        self,
        cache_paths: List[Path] = [],
    ):
        super().__init__()

        all_cache_keys = None
        all_cache_values = None
        max_len = 0

        for i, path in enumerate(cache_paths):
            if not isinstance(path, str):
                raise TypeError(f"path must be a string, got {type(path)}")
            print(path)
            checkpoint = torch.load(path, map_location="cuda", weights_only=False)

            # Ensure necessary keys are in the checkpoint
            for key in [
                "trainable_keys",
                "trainable_values",
                "fixed_keys",
                "fixed_values",
            ]:
                if key not in checkpoint:
                    raise KeyError(f"Key '{key}' not found in checkpoint")

            n_layers = len(checkpoint["trainable_keys"])
            n_heads = checkpoint["trainable_keys"][0].size(1)
            num_tokens = checkpoint["trainable_keys"][0].size(2)
            head_dim = checkpoint["trainable_keys"][0].size(3)

            if len(checkpoint["fixed_keys"]) != n_layers:
                raise AssertionError(
                    "Mismatch in number of layers between trainable and fixed keys"
                )
            if checkpoint["fixed_keys"]:
                if (
                    checkpoint["fixed_keys"][0].size(1) != n_heads
                    or checkpoint["fixed_keys"][0].size(3) != head_dim
                ):
                    raise AssertionError(
                        "Mismatch in head configuration between trainable and fixed keys"
                    )

            config = AttnConfig(n_layers=n_layers, n_heads=n_heads, head_dim=head_dim)
            # Here, num_tokens is inferred from trainable keys, but note that the total tokens may be different if fixed tokens exist.
            # The number of fixed tokens can be inferred from fixed_keys if available.
            num_frozen_tokens = (
                checkpoint["fixed_keys"][0].size(2) if checkpoint["fixed_keys"] else 0
            )

            num_tokens = num_tokens + num_frozen_tokens
            num_frozen_tokens = num_frozen_tokens

            if all_cache_keys is None:
                assert all_cache_values is None
                all_cache_keys = [j for j in checkpoint["trainable_keys"]]
                all_cache_values = [j for j in checkpoint["trainable_values"]]
            else:
                assert all_cache_keys is not None
                try:
                    assert (
                        len(all_cache_keys)
                        == len(checkpoint["trainable_keys"])
                        == len(all_cache_values)
                        == len(checkpoint["trainable_values"])
                    )
                except:
                    breakpoint()

                print("I VALUE IN LOOP", i)
                assert i == 1
                for layer_idx in range(len(all_cache_keys)):
                    all_cache_keys[layer_idx] = torch.cat(
                        [
                            all_cache_keys[layer_idx],
                            checkpoint["trainable_keys"][layer_idx],
                        ],
                        dim=2,
                    )
                    all_cache_values[layer_idx] = torch.cat(
                        [
                            all_cache_values[layer_idx],
                            checkpoint["trainable_values"][layer_idx],
                        ],
                        dim=2,
                    )

            max_len = num_tokens

            if i == 0:
                fixed_keys = checkpoint["fixed_keys"]
                fixed_values = checkpoint["fixed_values"]

        ### PARAMETERS ###

        # for vec in itertools.chain(all_cache_keys, all_cache_values):
        # assert vec.shape == (1, config.n_heads, 2 * (num_tokens - 1), config.head_dim)

        self.fixed_keys = nn.ParameterList(
            [nn.Parameter(keys_vec.contiguous()) for keys_vec in fixed_keys]
            if num_frozen_tokens
            else []
        )
        self.fixed_values = nn.ParameterList(
            [nn.Parameter(values_vec.contiguous()) for values_vec in fixed_values]
            if num_frozen_tokens
            else []
        )

        self.num_frozen_tokens = num_frozen_tokens

        for param in itertools.chain(self.fixed_keys, self.fixed_values):
            param.requires_grad = False

        self.trainable_keys = nn.ParameterList(
            [nn.Parameter(keys_vec.contiguous()) for keys_vec in all_cache_keys]
        )
        self.trainable_values = nn.ParameterList(
            [nn.Parameter(values_vec.contiguous()) for values_vec in all_cache_values]
        )

        self.num_tokens = max_len
        self.config = config
        # breakpoint()

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return (
            super().get_seq_length(layer_idx)
            + ( self.num_tokens * 2 ) - 1

        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return a list[tuple(Tensor, Tensor)] shaped like typical Hugging Face past_key_values.
        i.e. past_key_values[layer] = (key, value)
        """
        key_states, value_states = super().update(
            key_states, value_states, layer_idx, cache_kwargs
        )

        assert key_states.shape == value_states.shape
        batch_size = key_states.shape[0]

        if self.num_frozen_tokens > 0:
            fixed_keys = self.fixed_keys[layer_idx].repeat(batch_size, 1, 1, 1)
            fixed_values = self.fixed_values[layer_idx].repeat(batch_size, 1, 1, 1)

            trainable_keys = self.trainable_keys[layer_idx].repeat(batch_size, 1, 1, 1)
            trainable_values = self.trainable_values[layer_idx].repeat(
                batch_size, 1, 1, 1
            )

            key_states = torch.cat([fixed_keys, trainable_keys, key_states], dim=-2)
            value_states = torch.cat(
                [fixed_values, trainable_values, value_states], dim=-2
            )
        else:
            trainable_keys = self.trainable_keys[layer_idx].repeat(batch_size, 1, 1, 1)
            trainable_values = self.trainable_values[layer_idx].repeat(
                batch_size, 1, 1, 1
            )

            key_states = torch.cat([trainable_keys, key_states], dim=-2)
            value_states = torch.cat([trainable_values, value_states], dim=-2)

        return (key_states, value_states)

    def update_separate(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return a list[tuple(Tensor, Tensor)] shaped like typical Hugging Face past_key_values.
        i.e. past_key_values[layer] = (key, value)
        """
        key_states, value_states = super().update(
            key_states, value_states, layer_idx, cache_kwargs
        )

        assert key_states.shape == value_states.shape
        batch_size = key_states.shape[0]

        if self.num_frozen_tokens > 0:

            shared_key_states = torch.cat(
                [self.fixed_keys[layer_idx], self.trainable_keys[layer_idx]], dim=-2
            )
            shared_value_states = torch.cat(
                [self.fixed_values[layer_idx], self.trainable_values[layer_idx]], dim=-2
            )

        else:
            raise NotImplementedError("No frozen tokens")
            trainable_keys = self.trainable_keys[layer_idx].repeat(batch_size, 1, 1, 1)
            trainable_values = self.trainable_values[layer_idx].repeat(
                batch_size, 1, 1, 1
            )

            key_states = torch.cat([trainable_keys, key_states], dim=-2)
            value_states = torch.cat([trainable_values, value_states], dim=-2)

        return (key_states, value_states), (shared_key_states, shared_value_states)

    def clear(self):
        self.key_cache: list[torch.Tensor] = [[] for _ in range(self.config.n_layers)]
        self.value_cache: list[torch.Tensor] = [[] for _ in range(self.config.n_layers)]

    def save(self, path: str):
        """Saves the trainable keys and values to the specified path."""
        torch.save(
            {
                "trainable_keys": self.trainable_keys,
                "trainable_values": self.trainable_values,
                "fixed_keys": self.fixed_keys,
                "fixed_values": self.fixed_values,
            },
            path,
        )

    @classmethod
    def from_pretrained(cls, path: str, device: Optional[str] = None):
        if not isinstance(path, str):
            raise TypeError(f"path must be a string, got {type(path)}")
        print(path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Ensure necessary keys are in the checkpoint
        for key in ["trainable_keys", "trainable_values", "fixed_keys", "fixed_values"]:
            if key not in checkpoint:
                raise KeyError(f"Key '{key}' not found in checkpoint")

        n_layers = len(checkpoint["trainable_keys"])
        n_heads = checkpoint["trainable_keys"][0].size(1)
        num_tokens = checkpoint["trainable_keys"][0].size(2)
        head_dim = checkpoint["trainable_keys"][0].size(3)

        if len(checkpoint["fixed_keys"]) != n_layers:
            raise AssertionError(
                "Mismatch in number of layers between trainable and fixed keys"
            )
        if checkpoint["fixed_keys"]:
            if (
                checkpoint["fixed_keys"][0].size(1) != n_heads
                or checkpoint["fixed_keys"][0].size(3) != head_dim
            ):
                raise AssertionError(
                    "Mismatch in head configuration between trainable and fixed keys"
                )

        config = AttnConfig(n_layers=n_layers, n_heads=n_heads, head_dim=head_dim)
        # Here, num_tokens is inferred from trainable keys, but note that the total tokens may be different if fixed tokens exist.
        # The number of fixed tokens can be inferred from fixed_keys if available.
        num_frozen_tokens = (
            checkpoint["fixed_keys"][0].size(2) if checkpoint["fixed_keys"] else 0
        )

        return cls(
            config=config,
            num_tokens=num_tokens + num_frozen_tokens,
            keys=[
                (
                    torch.cat([fixed, trainable], dim=2).contiguous()
                    if num_frozen_tokens > 0
                    else trainable
                )
                for fixed, trainable in zip(
                    checkpoint["fixed_keys"], checkpoint["trainable_keys"]
                )
            ],
            values=[
                (
                    torch.cat([fixed, trainable], dim=2).contiguous()
                    if num_frozen_tokens > 0
                    else trainable
                )
                for fixed, trainable in zip(
                    checkpoint["fixed_values"], checkpoint["trainable_values"]
                )
            ],
            num_frozen_tokens=num_frozen_tokens,
        )
