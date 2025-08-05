import abc
from collections import defaultdict
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
from venv import logger
from pydrantic import ObjectConfig
import torch
import torch.nn as nn

from typing import Optional


@dataclass
class AttnConfig:
    n_layers: int
    n_heads: int
    head_dim: int

CARTRIDGE_SEQ_ID = -1

@dataclass
class CacheSequence:
    """Cache data for a single sequence across all layers."""
    keys: list[torch.Tensor]  # List of pre-allocated tensors per layer (1, n_heads, max_seq_len, head_dim)
    values: list[torch.Tensor]  # List of pre-allocated tensors per layer (1, n_heads, max_seq_len, head_dim)
    length: int  # Current number of tokens stored in this sequence (shared across layers)
    
    def append_token(self, layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor, max_seq_len: int) -> bool:
        """Append a single token to this sequence cache for a specific layer. Returns True if successful."""
        if self.length >= max_seq_len:
            return False  # Would exceed max length
        
        # Insert the single token at the current position
        self.keys[layer_idx][:, :, self.length:self.length+1] = new_key
        self.values[layer_idx][:, :, self.length:self.length+1] = new_value
        return True
    
    def get_cached_tensors(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the cached keys and values for a specific layer, truncated to actual length."""
        if self.length == 0:
            # Return empty tensors with correct shape
            keys_tensor = self.keys[layer_idx]
            values_tensor = self.values[layer_idx]
            return (torch.empty(1, keys_tensor.shape[1], 0, keys_tensor.shape[3], 
                              dtype=keys_tensor.dtype, device=keys_tensor.device),
                   torch.empty(1, values_tensor.shape[1], 0, values_tensor.shape[3],
                              dtype=values_tensor.dtype, device=values_tensor.device))
        return self.keys[layer_idx][:, :, :self.length], self.values[layer_idx][:, :, :self.length]

class FastTrainableCache(nn.Module):
    """A trainable packed cache for generation with FlexAttention.
    
    The cache must do two things, which a standard Hugging Face cache does not:

    - Keep track of sequence membership of the cache and expose it to the model via
    the seq_ids method. The model will use this once per forward pass to construct 
    the appropriate block mask. 
    - Keep track of keys and values and expose them to the model in a packed manner via 
    the update method.
    
    This implementation keeps each sequence in its own pre-allocated tensor to ensure
    that the block mask is contiguous when sequences are concatenated.

    Args:
        config: The attention configuration, which we use to construct the cache
        max_seq_len: Maximum sequence length for pre-allocation
        init_keys (list[torch.Tensor], optional): A `config.n_layers` length list of 
            trainable keys for the cache, should be of shape (1, n_heads, num_trainable_tokens, head_dim).
        init_values (list[torch.Tensor]): A `config.n_layers` length list of 
            trainable values for the cache, should be of shape (1, n_heads, num_trainable_tokens, head_dim).
        num_frozen_tokens (int): The number of the trainable tokens to freeze at the 
            beginning of the cache.
    """
    def __init__(
        self,        
        config: AttnConfig,
        max_seq_len: int,
        init_keys: list[torch.Tensor]=None,
        init_values: list[torch.Tensor]=None,
        num_frozen_tokens: int = 0,
        device: str = 'cuda:0',
    ):
        super().__init__()
        self.config = config
        self.max_seq_len = max_seq_len
        self.device = device
        
        # Pre-allocate cache sequences
        # Dictionary mapping seq_id -> CacheSequence
        self._sequence_caches: dict[int, CacheSequence] = {}
        # Track sequence lengths globally (updated by update_seq_ids)
        self._seq_id_to_len: dict[int, int] = defaultdict(int)
        # Track which sequences are active and their order
        self._active_sequences: list[int] = []
        # Compatibility with original cache interface
        self._keys = [None] * config.n_layers
        self._values = [None] * config.n_layers
        self._num_tokens = 0

        assert (init_keys is None) == (init_values is None)
        if init_keys is None:
            self._num_trainable_tokens, self._num_frozen_tokens = 0, 0
            self.frozen_keys, self.frozen_values = None, None
            self.trainable_keys, self.trainable_values = None, None
            self._seq_ids = None
            self._init_seq_ids = None
        else:
            self._num_init_tokens = init_keys[0].shape[2]
            self._num_frozen_tokens = num_frozen_tokens
            self._num_trainable_tokens = self._num_init_tokens - num_frozen_tokens
            assert len(init_keys) == config.n_layers == len(init_values)
            
            # we initialize the seq ids for the first 
            # `num_trainable_tokens + num_frozen_tokens` tokens to -1, which means that 
            # the tokens are part of the cartridge and should be attended to by 
            # all tokens.
            _seq_ids = torch.full(
                (self._num_init_tokens,),
                fill_value=CARTRIDGE_SEQ_ID, 
                dtype=torch.long,
            )
            self.register_buffer("_init_seq_ids", _seq_ids)
            self.register_buffer("_seq_ids", _seq_ids)  # .to moves the tensor to the correct device

            for vec in itertools.chain(init_keys, init_values):
                assert vec.shape == (1, config.n_heads, self._num_init_tokens, config.head_dim)

            self.frozen_keys = nn.ParameterList(
                [
                    nn.Parameter(keys_vec[:, :, :num_frozen_tokens].contiguous())
                    for keys_vec in init_keys
                ]
                if num_frozen_tokens
                else []
            )
            self.frozen_values = nn.ParameterList(
                [
                    nn.Parameter(values_vec[:, :, :num_frozen_tokens].contiguous())
                    for values_vec in init_values
                ]
                if num_frozen_tokens
                else []
            )

            for param in itertools.chain(self.frozen_keys, self.frozen_values):
                param.requires_grad = False

            self.trainable_keys = nn.ParameterList(
                [
                    nn.Parameter(keys_vec[:, :, num_frozen_tokens:].contiguous())
                    for keys_vec in init_keys
                ]
            )
            self.trainable_values = nn.ParameterList(
                [
                    nn.Parameter(values_vec[:, :, num_frozen_tokens:].contiguous())
                    for values_vec in init_values
                ]
            )
            
    def update_seq_ids(self, curr_seq_ids: torch.Tensor) -> None:
        """Update sequence tracking with new sequence IDs. Called before first layer."""
        # Update global sequence tracking
        if self._seq_ids is None:
            self._seq_ids = curr_seq_ids.clone()
        else:
            self._seq_ids = torch.cat([self._seq_ids, curr_seq_ids], dim=0)
        self._num_tokens += curr_seq_ids.shape[0]
        
        # Count tokens per sequence ID
        active_seq_ids, counts = torch.unique(curr_seq_ids, return_counts=True)
        
        for seq_id, count in zip(active_seq_ids, counts):
            seq_id_item = seq_id.item()
            if seq_id_item == CARTRIDGE_SEQ_ID:
                continue  # Skip cartridge tokens
                
            # Update sequence length tracking
            self._seq_id_to_len[seq_id_item] += count.item()
            
            # Add to active sequences if new
            if seq_id_item not in self._active_sequences:
                self._active_sequences.append(seq_id_item)
                
                # Initialize cache for this sequence across all layers
                self._sequence_caches[seq_id_item] = CacheSequence(
                    keys=[torch.zeros(1, self.config.n_heads, self.max_seq_len, self.config.head_dim,
                                     dtype=torch.float32, device=self.device) 
                          for _ in range(self.config.n_layers)],
                    values=[torch.zeros(1, self.config.n_heads, self.max_seq_len, self.config.head_dim,
                                       dtype=torch.float32, device=self.device) 
                           for _ in range(self.config.n_layers)],
                    length=0
                )

    def update(
        self, 
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
        new_seq_ids: torch.Tensor,
        layer_idx: int,
        skip_append: bool = False,
    ):
        """Update the cache with new keys and values while maintaining sequence contiguity.
        
        Args:
            new_keys: (1, num_heads, seq_len, head_dim) tensor of new keys
            new_values: (1, num_heads, seq_len, head_dim) tensor of new values  
            new_seq_ids: (seq_len,) tensor of sequence ids for the new tokens
            layer_idx: index of the layer in the model.
            skip_append: if True, do not append the new keys and values to the cache, 
                just return the concatenation of the new_keys and values. 
        """
        assert new_seq_ids.shape[0] == new_keys.shape[2]
        assert new_seq_ids.shape[0] == new_values.shape[2]
        
        # Update the current layer's cached data
        if not skip_append:
            self._update_layer_cache(new_keys, new_values, new_seq_ids, layer_idx)
        
        # Build the final concatenated output
        return self._build_concatenated_output(layer_idx, new_keys, new_values, new_seq_ids, skip_append)
    
    def _update_layer_cache(self, new_keys, new_values, new_seq_ids, layer_idx):
        """Update the cache for a specific layer."""
        if layer_idx == 0:
            # On first layer, increment the sequence lengths
            unique_seq_ids = torch.unique(new_seq_ids)
            for seq_id in unique_seq_ids:
                seq_id_item = seq_id.item()
                if seq_id_item != CARTRIDGE_SEQ_ID and seq_id_item in self._sequence_caches:
                    cache = self._sequence_caches[seq_id_item]
                    # Count how many tokens belong to this sequence and increment length
                    seq_token_count = (new_seq_ids == seq_id).sum().item()
                    cache.length += seq_token_count
        
        # Group tokens by sequence ID and append to each layer
        for i, seq_id in enumerate(new_seq_ids):
            seq_id_item = seq_id.item()
            if seq_id_item == CARTRIDGE_SEQ_ID:
                continue  # Skip cartridge tokens
            
            if seq_id_item in self._sequence_caches:
                cache = self._sequence_caches[seq_id_item]
                # Append single token to the sequence cache for this layer
                single_key = new_keys[:, :, i:i+1]
                single_value = new_values[:, :, i:i+1]
                
                # Calculate the position for this token within the sequence
                # We need to find what position this token should go to
                tokens_before = (new_seq_ids[:i] == seq_id).sum().item()
                current_pos = cache.length - (new_seq_ids == seq_id).sum().item() + tokens_before
                
                cache.keys[layer_idx][:, :, current_pos:current_pos+1] = single_key
                cache.values[layer_idx][:, :, current_pos:current_pos+1] = single_value
    
    def _build_concatenated_output(self, layer_idx, new_keys, new_values, new_seq_ids, skip_append):
        """Build the final concatenated keys and values for this layer."""
        keys_list = []
        values_list = []
        
        # Add frozen tokens first
        if self._num_frozen_tokens > 0:
            keys_list.append(self.frozen_keys[layer_idx])
            values_list.append(self.frozen_values[layer_idx])
        
        # Add trainable tokens
        if self._num_trainable_tokens > 0:
            keys_list.append(self.trainable_keys[layer_idx])
            values_list.append(self.trainable_values[layer_idx])
        
        # Add cached sequences in order, truncated to their actual length
        for seq_id in self._active_sequences:
            if seq_id in self._sequence_caches:
                cache = self._sequence_caches[seq_id]
                cached_keys, cached_values = cache.get_cached_tensors(layer_idx)
                if cached_keys.shape[2] > 0:
                    keys_list.append(cached_keys)
                    values_list.append(cached_values)
        
        # Only add new keys/values if skipping append (for generation)
        # When not skipping append, new tokens are already stored in sequence caches
        if skip_append:
            # Filter out cartridge tokens from new keys/values
            non_cartridge_mask = new_seq_ids != CARTRIDGE_SEQ_ID
            if non_cartridge_mask.any():
                filtered_keys = new_keys[:, :, non_cartridge_mask]
                filtered_values = new_values[:, :, non_cartridge_mask]
                if filtered_keys.shape[2] > 0:
                    keys_list.append(filtered_keys)
                    values_list.append(filtered_values)
        
        if not keys_list:
            # Return empty tensors if no keys/values
            return (torch.empty(1, self.config.n_heads, 0, self.config.head_dim, 
                              dtype=new_keys.dtype, device=new_keys.device),
                   torch.empty(1, self.config.n_heads, 0, self.config.head_dim,
                              dtype=new_values.dtype, device=new_values.device))
        
        return torch.cat(keys_list, dim=2), torch.cat(values_list, dim=2)
    
    def num_tokens(self) -> int:
        """Get the sequence length of the cache."""
        return self._num_frozen_tokens + self._num_trainable_tokens + self._num_tokens
    
    def num_cartridge_tokens(self) -> int:
        """Get the number of tokens in the cartridge."""
        return self._num_frozen_tokens + self._num_trainable_tokens
    
    def seq_ids(self) -> torch.Tensor:
        """Returns the sequence ids of the cache in the same order as concatenated output."""
        seq_ids_list = []
        
        # Add cartridge seq_ids first (frozen + trainable)
        if self._init_seq_ids is not None:
            seq_ids_list.append(self._init_seq_ids)
        
        # Add sequence IDs for each active sequence in order using global length tracking
        for seq_id in self._active_sequences:
            actual_len = self._seq_id_to_len[seq_id]
            if actual_len > 0:
                # Create tensor of sequence IDs for this sequence on the correct device
                seq_tensor = torch.full((actual_len,), seq_id, 
                                       dtype=torch.long, device=self.device)
                seq_ids_list.append(seq_tensor)
        
        if not seq_ids_list:
            # Return empty tensor if no sequences
            return torch.empty(0, dtype=torch.long, device=self.device)
        
        return torch.cat(seq_ids_list, dim=0)
       
    def clear(self):
        self._keys = [None] * self.config.n_layers
        self._values = [None] * self.config.n_layers
        self._sequence_caches = {}
        self._seq_id_to_len = defaultdict(int)
        self._active_sequences = []
        self._num_tokens = 0
        self._seq_ids = self._init_seq_ids

    def save(self, path: str):
        """Saves the trainable keys and values to the specified path."""
        torch.save(
            {
                "trainable_keys": self.trainable_keys,
                "trainable_values": self.trainable_values,
                "frozen_keys": self.frozen_keys,
                "frozen_values": self.frozen_values,
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
        for key in ["trainable_keys", "trainable_values", "frozen_keys", "frozen_values"]:
            if key not in checkpoint:
                raise KeyError(f"Key '{key}' not found in checkpoint")

        n_layers = len(checkpoint["trainable_keys"])
        n_heads = checkpoint["trainable_keys"][0].size(1)
        head_dim = checkpoint["trainable_keys"][0].size(3)

        if len(checkpoint["frozen_keys"]) != n_layers:
            raise AssertionError(
                "Mismatch in number of layers between trainable and fixed keys"
            )
        if checkpoint["frozen_keys"]:
            if (
                checkpoint["frozen_keys"][0].size(1) != n_heads
                or checkpoint["frozen_keys"][0].size(3) != head_dim
            ):
                raise AssertionError(
                    "Mismatch in head configuration between trainable and fixed keys"
                )

        config = AttnConfig(n_layers=n_layers, n_heads=n_heads, head_dim=head_dim)
        # Here, num_tokens is inferred from trainable keys, but note that the total tokens may be different if fixed tokens exist.
        # The number of fixed tokens can be inferred from frozen_keys if available.
        num_frozen_tokens = (
            checkpoint["frozen_keys"][0].size(1) if checkpoint["frozen_keys"] else 0
        )

        return cls(
            config=config,
            max_seq_len=1024,  # Default max sequence length
            init_keys=[
                (
                    torch.cat([fixed, trainable], dim=2).contiguous()
                    if num_frozen_tokens > 0
                    else trainable
                )
                for fixed, trainable in zip(
                    checkpoint["frozen_keys"], checkpoint["trainable_keys"]
                )
            ],
            init_values=[
                (
                    torch.cat([fixed, trainable], dim=2).contiguous()
                    if num_frozen_tokens > 0
                    else trainable
                )
                for fixed, trainable in zip(
                    checkpoint["frozen_values"], checkpoint["trainable_values"]
                )
            ],
            num_frozen_tokens=num_frozen_tokens,
            device=device if device is not None else 'cpu',
        )


class KVCacheFactory(abc.ABC):
    class Config(ObjectConfig):
        _pass_as_config = True

        # SE (03/26): we freeze the first token to prevent forgetting
        num_frozen_tokens: int = 1

    def __init__(self, config: Config):
        self.config = config

    @abc.abstractmethod
    def initialize_kv_cache(
        self, tokenizer, model, attn_config: AttnConfig, max_seq_len: int = 1024
    ) -> FastTrainableCache:
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
        tokenizer,
        model,
        attn_config: AttnConfig,
        max_seq_len: int = 1024,
    ) -> tuple[FastTrainableCache, dict]:
        raise NotImplementedError()

    @property
    def local_kv_cache_path(self) -> Path:
        # TODO: better file extension
        return Path(self.config.directory) / "kv_cache.torch"

    @property
    def local_metadata_path(self) -> Path:
        # TODO: better file extension
        return Path(self.config.directory) / "metadata.json"

    def maybe_load_cached(self) -> Optional[FastTrainableCache]:
        if self.config.force_recreate:
            return

        if not self.config.is_wandb:
            if self.local_kv_cache_path.exists():
                logger.info(
                    f"State Saving KV initializer: loading KV cache from: {self.local_kv_cache_path}"
                )
                return FastTrainableCache.from_pretrained(
                    str(self.local_kv_cache_path.absolute()),
                )

            return

        raise NotImplementedError("Need to add saving to wanb")

    def initalize_kv_cache(
        self, tokenizer, model, attn_config: AttnConfig, max_seq_len: int = 1024
    ) -> FastTrainableCache:
        maybe_cache = self.maybe_load_cached()
        if maybe_cache is not None:
            assert (
                maybe_cache._num_trainable_tokens + maybe_cache._num_frozen_tokens
                == self.config.num_tokens
            )
            assert maybe_cache.config == attn_config
            return maybe_cache

        cache, metadata = self.initalize_kv_cache_impl(
            tokenizer, model, attn_config, max_seq_len
        )

        Path(self.config.directory).mkdir(parents=True, exist_ok=True)

        with open(self.local_metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        cache.save(str(self.local_kv_cache_path.absolute()))
        logger.info(
            f"State Saving KV initializer: saving KV cache to: {self.local_kv_cache_path}"
        )

        return cache