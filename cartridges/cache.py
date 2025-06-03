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

from cartridges.context import StructuredContext
from transformers import DynamicCache


@dataclass
class AttnConfig:
    n_layers: int
    n_heads: int
    head_dim: int


# SE (03/24): I had to add nn.Module to the inheritance because I was getting the error
# TrainableCache has no attribute `.to`. Still not sure why this started happening --
# perhaps there was an update to the transformers library.
class TrainableCache(DynamicCache, nn.Module):

    def __init__(
        self,
        config: AttnConfig,
        num_tokens: int,
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
        num_frozen_tokens: int = 0,
    ):
        super().__init__()
        assert 0 <= num_frozen_tokens < num_tokens

        self.config = config
        self.num_trainable_tokens = num_tokens - num_frozen_tokens
        self.num_frozen_tokens = num_frozen_tokens

        assert len(keys) == config.n_layers
        assert len(values) == config.n_layers

        for vec in itertools.chain(keys, values):
            assert vec.shape == (1, config.n_heads, num_tokens, config.head_dim)

        self.fixed_keys = nn.ParameterList(
            [
                nn.Parameter(keys_vec[:, :, :num_frozen_tokens].contiguous())
                for keys_vec in keys
            ]
            if num_frozen_tokens
            else []
        )
        self.fixed_values = nn.ParameterList(
            [
                nn.Parameter(values_vec[:, :, :num_frozen_tokens].contiguous())
                for values_vec in values
            ]
            if num_frozen_tokens
            else []
        )

        for param in itertools.chain(self.fixed_keys, self.fixed_values):
            param.requires_grad = False

        self.trainable_keys = nn.ParameterList(
            [
                nn.Parameter(keys_vec[:, :, num_frozen_tokens:].contiguous())
                for keys_vec in keys
            ]
        )
        self.trainable_values = nn.ParameterList(
            [
                nn.Parameter(values_vec[:, :, num_frozen_tokens:].contiguous())
                for values_vec in values
            ]
        )

    def clone(self):
        return TrainableCache(
            config=self.config,
            num_tokens=self.num_trainable_tokens + self.num_frozen_tokens,
            keys=[
                (
                    torch.cat([fixed, trainable], dim=2).contiguous()
                    if self.num_frozen_tokens > 0
                    else trainable
                )
                for fixed, trainable in zip(
                    self.fixed_keys,
                    self.trainable_keys,
                    strict=True,
                )
            ],
            values=[
                (
                    torch.cat([fixed, trainable], dim=2).contiguous()
                    if self.num_frozen_tokens > 0
                    else trainable
                )
                for fixed, trainable in zip(
                    self.fixed_values,
                    self.trainable_values,
                    strict=True,
                )
            ],
            num_frozen_tokens=self.num_frozen_tokens,
        )

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return (
            super().get_seq_length(layer_idx)
            + self.num_trainable_tokens
            + self.num_frozen_tokens
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
            # raise NotImplementedError("No frozen tokens")
            shared_key_states = self.trainable_keys[layer_idx].contiguous()
            shared_value_states = self.trainable_values[layer_idx].contiguous()

            # trainable_keys = self.trainable_keys[layer_idx].repeat(batch_size, 1, 1, 1)
            # trainable_values = self.trainable_values[layer_idx].repeat(
            #     batch_size, 1, 1, 1
            # )

            # key_states = torch.cat([trainable_keys, key_states], dim=-2)
            # value_states = torch.cat([trainable_values, value_states], dim=-2)

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


class KVCacheFactory(abc.ABC):
    class Config(ObjectConfig):
        _pass_as_config = True

        # SE (03/26): we freeze the first token to prevent forgetting
        num_frozen_tokens: int = 1

    def __init__(self, config: Config):
        self.config = config

    @abc.abstractmethod
    def initalize_kv_cache(
        self, context: StructuredContext, tokenizer, model, attn_config: AttnConfig
    ) -> TrainableCache:
        raise NotImplementedError()


class KVCacheFactoryWithStateSaving(abc.ABC):
    class Config(KVCacheFactory.Config):
        directory: str
        is_wandb: bool
        force_recreate: bool = False

    def __init__(self, config: Config):
        self.config = config

    @abc.abstractmethod
    def initalize_kv_cache_impl(
        self,
        context: StructuredContext,
        tokenizer,
        model,
        attn_config: AttnConfig,
    ) -> tuple[TrainableCache, dict]:
        raise NotImplementedError()

    @property
    def local_kv_cache_path(self) -> Path:
        # TODO: better file extension
        return Path(self.config.directory) / "kv_cache.torch"

    @property
    def local_metadata_path(self) -> Path:
        # TODO: better file extension
        return Path(self.config.directory) / "metadata.json"

    def maybe_load_cached(self) -> Optional[TrainableCache]:
        if self.config.force_recreate:
            return

        if not self.config.is_wandb:
            if self.local_kv_cache_path.exists():
                logger.info(
                    f"State Saving KV initializer: loading KV cache from: {self.local_kv_cache_path}"
                )
                return TrainableCache.from_pretrained(
                    str(self.local_kv_cache_path.absolute()),
                )

            return

        raise NotImplementedError("Need to add saving to wanb")

    def initalize_kv_cache(
        self, context: StructuredContext, tokenizer, model, attn_config: AttnConfig
    ) -> TrainableCache:
        maybe_cache = self.maybe_load_cached()
        if maybe_cache is not None:
            assert (
                maybe_cache.num_trainable_tokens + maybe_cache.num_frozen_tokens
                == self.config.num_tokens
            )
            assert maybe_cache.config == attn_config
            return maybe_cache

        cache, metadata = self.initalize_kv_cache_impl(
            context, tokenizer, model, attn_config
        )

        Path(self.config.directory).mkdir(parents=True, exist_ok=True)

        with open(self.local_metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        cache.save(str(self.local_kv_cache_path.absolute()))
        logger.info(
            f"State Saving KV initializer: saving KV cache to: {self.local_kv_cache_path}"
        )

        return cache
