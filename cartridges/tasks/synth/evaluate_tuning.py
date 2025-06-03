""" Evaluate the performance of cache tuning on a synthetic task. 
"""

from typing import List, Optional
import pandas as pd
from pydrantic import RunConfig
import torch
from torch import nn
from torch.utils.data import Dataset
import tqdm
import wandb


from capsules.tasks.synth.config import DataConfig, LoggerConfig, ModelConfig
from capsules.tasks.synth.cache import BaseKVCache
from capsules.tasks.synth.data import TuningDataset
from capsules.tasks.synth.pretrain import compute_metrics
from capsules.utils.wandb import WandBConfig, prepare_wandb


class EvaluateTuningConfig(RunConfig):
    data: DataConfig
    
    model: ModelConfig
    kv_cache: BaseKVCache.Config

    
    max_epochs: int = 100
    
    wandb: Optional[WandBConfig] = None

    # stop training once this metric reaches the threshold
    # set metric to None to disable early stopping
    early_stopping_metric: str = "valid/accuracy"
    early_stopping_threshold: float = 0.99
    slice_keys: List[str] = []

    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    seed: int = 123

    def run(self):
        from capsules.tasks.synth.evaluate_tuning import evaluate_tuning
        evaluate_tuning(self)

def train_cache(
    cache: BaseKVCache,
    model: nn.Module,
    tuning_dataset: TuningDataset,
    num_steps: int,
    max_seq_len: int,
) -> BaseKVCache:
    
    for _ in range(num_steps):
        inputs, targets = tuning_dataset.sample(max_seq_len=max_seq_len)
        # This is basically stateful MQAR
        # e.g. in MQAR inputs is just the token for a query
        # and targets is the token for the corresponding value

        # something like this TODO: double check, it was generate dby cursor 
        logits = model(inputs, cache=cache)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    

    


def evaluate_tuning(config: EvaluateTuningConfig):

    model = model.instantiate()
    model.to(config.device)

    cache = BaseKVCache(config.kv_cache)
    cache.to(config.device)

    data = prepare_data_for_tuning(config.data)

    prepare_wandb(config.wandb, config.to_dict())


    
    with torch.no_grad(), tqdm(
        total=len(data),
        desc=f"Evaluate Tuning",
        postfix={"loss": "-", "acc": "-"},
    ) as iterator:
        for ctx, query_inputs, query_targets, slices in data:
            # `ctx`` is an object that we can train the cache on
            # e.g. in MQAR this is the list of key-value pairs BEFORE the query
            # `query_inputs` is a sequence of inputs on which we will evaluate 
            # the cache
            # e.g. in MQAR this is the sequence of queries 
            # `query_targets` is the sequence of targets for the queries

            cache = train_cache(
                cache, 
                model,
                ctx
            )

            query_inputs, query_targets = query_inputs.to(config.device), query_targets.to(config.device)
            logits = model(query_inputs, cache=cache)

           
            # SE: important to
            preds = torch.argmax(logits, dim=-1).cpu()
            results.extend(compute_metrics(preds, query_targets.cpu(), slices))
            
            iterator.update(1)

        results = pd.DataFrame(results)
        test_accuracy = results["accuracy"].mean()

        # logging and printing
        metrics = {
            "valid/accuracy": test_accuracy.item(),
        }

        # compute metrics for slices
        # for key in self.slice_keys:
        #     acc_by_slice = results.groupby(key)["accuracy"].mean()
        #     for value, accuracy in acc_by_slice.items():
        #         metrics[f"valid/{key}/accuracy-{value}"] = accuracy

        iterator.set_postfix(metrics)
        wandb.log(metrics)
    return metrics
