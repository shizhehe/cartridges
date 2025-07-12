from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from pydrantic import ObjectConfig
from pydantic import BaseModel
from cartridges.clients.usage import Usage
from dataclasses import dataclass, asdict

class ClientConfig(ObjectConfig):
    _pass_as_config: bool = True

    model_name: str

    show_progress_bar: bool = False

    def instantiate(self, *args, **kwargs) -> "Client":
        return super().instantiate(*args, **kwargs)


class Client(ABC):
    def __init__(self, config: ClientConfig):
        self.config = config

    @abstractmethod
    async def chat(
        self, 
        chats: List[List[Dict[str, Any]]], 
        temperature: float = 0.6, 
        stop: List[str] = [], 
        max_completion_tokens: Optional[int] = None,
        frequency_penalty: float = 0.0,
        top_logprobs: int = 1,
        logprobs_start_message: Optional[int] = None,
        modal_upstream_id: Optional[str] = None,
    ) -> ClientResponse:
        raise NotImplementedError


@dataclass(slots=True)
class ClientResponse:
    samples: List[ClientSample]
    usage: Usage
    
    timings: Optional[List[Dict[str, Any]]] = None
    def to_dict(self):
        return asdict(self)


@dataclass(slots=True)
class ClientSample:
    text: str
    token_ids: Optional[List[int]] = None

    top_logprobs: Optional[TopLogprobs] = None



@dataclass(slots=True)
class TopLogprobs:
    """
    logprobs   – [num_tokens , num_top_logprobs]  (natural-log p)
    token_ids  – [num_tokens , num_top_logprobs]  (int64)
    """
    logprobs:  np.ndarray
    token_ids: np.ndarray

    # ------------------------------------------------------------------
    # Vectorised flattening (no Python loop)
    # Keeps the minimum set of columns so that each row’s cumulative
    # probability mass ≥ `threshold` (the crossing column is included).
    # Returns a `FlatTopLogprobs` object.
    # ------------------------------------------------------------------
    def flatten(self, threshold: float = 0.99) -> FlatTopLogprobs:
        if self.logprobs.ndim != 2 or self.token_ids.ndim != 2:
            raise ValueError("logprobs and token_ids must be 2-D arrays")
        if self.logprobs.shape != self.token_ids.shape:
            raise ValueError("logprobs and token_ids must have identical shapes")
        if not (0.0 < threshold <= 1.0):
            raise ValueError("threshold must be in (0,1]")

        T, K = self.logprobs.shape

        # -- 1. convert log-p -> p and sort each row descending
        probs        = np.exp(self.logprobs)                     # [T , K]
        sort_idx     = np.argsort(-probs, axis=1)                # [T , K]
        probs_sorted = np.take_along_axis(probs,       sort_idx, axis=1)
        ids_sorted   = np.take_along_axis(self.token_ids, sort_idx, axis=1)
        logp_sorted  = np.take_along_axis(self.logprobs, sort_idx, axis=1)

        # -- 2. cumulative mass & cut-off per row
        cum_mass  = np.cumsum(probs_sorted, axis=1)              # [T , K]
        cut_idx   = (cum_mass >= threshold).argmax(axis=1)       # [T]

        # -- 3. boolean mask (keep 0 … cut_idx inclusive)
        mask      = np.arange(K) < (cut_idx[:, None] + 1)        # [T , K]

        # -- 4. flatten
        token_idx = np.repeat(np.arange(T), K)[mask.ravel()]     # [N]
        token_id  = ids_sorted[mask]                             # [N]
        logprobs  = logp_sorted[mask]                            # [N]

        return FlatTopLogprobs(
            token_idx=token_idx,
            token_id=token_id,
            logprobs=logprobs,
            shape=(T, K),
        )


@dataclass(slots=True)
class FlatTopLogprobs:
    """
    A *flat* representation produced by TopLogprobs.flatten()

        • token_idx  – 1-D   [N]    row number in the original tensor
        • token_id   – 1-D   [N]    vocabulary id
        • logprobs   – 1-D   [N]    log-probabilities (natural-log)

    The original dense tensors can be rebuilt with `.reconstruct()`.
    """
    token_idx:  np.ndarray          # int64
    token_id:   np.ndarray          # int64
    logprobs:   np.ndarray          # float32/float64
    shape:      tuple[int, int]     # (num_tokens , num_top_logprobs)

    # ------------------------------------------------------------------
    # Re-inflate the sparse/flat representation back to the dense form
    # used by `TopLogprobs` (missing cells are filled with −inf / −1).
    # ------------------------------------------------------------------
    def reconstruct(self) -> "TopLogprobs":
        num_tokens, num_top = self.shape

        dense_logp = np.full(self.shape, -np.inf, dtype=self.logprobs.dtype)
        dense_ids  = np.full(self.shape, -1,      dtype=self.token_id.dtype)

        dense_logp[self.token_idx, np.arange(self.token_idx.size)] = self.logprobs
        dense_ids [self.token_idx, np.arange(self.token_idx.size)] = self.token_id

        return TopLogprobs(logprobs=dense_logp, token_ids=dense_ids)


