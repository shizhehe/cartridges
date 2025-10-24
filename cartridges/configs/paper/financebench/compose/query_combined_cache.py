from pathlib import Path
from typing import Optional
import torch
from typing import List
import collections
import os
import wandb
import json
from tqdm import tqdm
from transformers import AutoTokenizer

from capsules.clients.together import TogetherClient
from capsules.kv_initialization.strategies.base_compose import TrainableCacheComposable
from capsules.kv_initialization.base import TrainableCache
from capsules.train import TrainConfig, CacheAndModel

device = "cuda" if torch.cuda.is_available() else "cpu"

from capsules.configs.paper.financebench.compose.experiments import CARTRIDGES
from capsules.configs.paper.financebench.compose.eval_utils import ( QA_PAIRS, run_query_set )



def load_single_model_and_cache_from_wandb(
    wandb_run_id: str,
    filename: str,
    device: str = "cuda",
) -> tuple[CacheAndModel, AutoTokenizer, TrainConfig]:
    train_config = TrainConfig.from_wandb(wandb_run_id, strict=False)
    model = train_config.model.instantiate().to(device)
    tokenizer = AutoTokenizer.from_pretrained(train_config.tokenizer)

    dir_ = train_config.run_dir + "/" + wandb_run_id

    for filename in ["cache-step1024.pt", "cache-step1020.pt", "cache-step1016.pt"]:
        try:
            out = wandb.restore(
                filename,
                run_path=wandb_run_id,
                root=dir_,  # this is intentional
            )
            cache = TrainableCache.from_pretrained(os.path.join(dir_, filename), device=device)
            break
        except Exception as e:
            cache = None

    context_str, icl_client = _load_icl_model(train_config, tokenizer)
    return CacheAndModel(cache=cache, model=model), tokenizer, train_config, context_str, icl_client


def load_model_and_cache_from_wandb(
    WANDB_RUNS, 
    device: str = "cuda",
) -> tuple[CacheAndModel, AutoTokenizer, TrainConfig]:
    
    wandb_run_id = WANDB_RUNS[0]
    train_config = TrainConfig.from_wandb(wandb_run_id, strict=False)

    model = train_config.model.instantiate().to(device)
    tokenizer = AutoTokenizer.from_pretrained(train_config.tokenizer)

    paths = []
    for run_id in WANDB_RUNS:
        dir_ = train_config.run_dir + "/" + run_id
        out = None
        try:
            filename = "cache-step1024.pt"
            out = wandb.restore(filename, run_path=run_id, root=dir_, )
            paths.append(  str(  (Path(dir_) / filename).absolute() ) )
        except Exception as e:
            pass 

        if out is None:
            try:
                filename = "cache-step1020.pt"
                out = wandb.restore(
                    filename,
                    run_path=run_id,
                    root=dir_,  # this is intentional
                )
                paths.append(  str(  (Path(dir_) / filename).absolute() ) )
            except Exception as e:
                pass

        if out is None:
            try:
                filename = "cache-step1016.pt"
                out = wandb.restore(
                    filename,
                    run_path=run_id,
                    root=dir_,  # this is intentional
                )
                paths.append(  str(  (Path(dir_) / filename).absolute() ) )
            except Exception as e:
                pass

    cache = TrainableCacheComposable(paths,)
    context_str, icl_client = _load_icl_model(train_config, tokenizer)

    return (
        CacheAndModel(cache=cache, model=model),
        tokenizer,
        train_config,
        context_str,
        icl_client,
    )


def _load_icl_model(train_config: TrainConfig, tokenizer: AutoTokenizer):
    # FIXME: add more models
    mapping = {
        "meta-llama/Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "meta-llama/Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct-Turbo",
    }
    model_name = train_config.model.pretrained_model_name_or_path
    if model_name in mapping:
        model_name = mapping[model_name]

        icl_client = TogetherClient.Config(
            model_name=model_name,
        ).instantiate()
    else:
        raise ValueError(f"Model {model_name} not found in mapping.")

    from capsules.datasets import ContextConvoDataset
    MAX_LENGTH = 110_000
    return "todo", icl_client


class ModelManager:
    def __init__(self):
        self.model: Optional[CacheAndModel] = None
        self.train_config: Optional[TrainConfig] = None
        self.tokenizer = None

        # for ICL generations
        self.context_str: Optional[str] = None
        self.icl_client: Optional[TogetherClient] = None

    def load_model(self, WANDB_RUNS: str):
        # print(f"Loading model!")
        if len(WANDB_RUNS) == 1:
            (
                self.model,
                self.tokenizer,
                self.train_config,
                self.context_str,
                self.icl_client,
            ) = load_single_model_and_cache_from_wandb(WANDB_RUNS[0], "cache-step1024.pt", device=device)
        else:
            (
                self.model,
                self.tokenizer,
                self.train_config,
                self.context_str,
                self.icl_client,
            ) = load_model_and_cache_from_wandb(WANDB_RUNS, device=device)

    def is_model_loaded(self):
        return self.model is not None and self.tokenizer is not None


# Sweep
def run_sweep(do_print):

    for pair, inputs in QA_PAIRS.items():
        comparative_messages = inputs["questions"]
        comparative_answers = inputs["answers"]

        if not comparative_answers: 
            continue

        company1, company2 = pair.split("_")
        cache_size_to_paths_1 = CARTRIDGES[company1]
        cache_size_to_paths_2 = CARTRIDGES[company2]

        print(f"======"*10)
        print(f"Running query set for {pair}")
        print(f"======"*10)


        cache_size_to_deps_to_scores = collections.defaultdict(dict)

        # for cache_size, run_id_dics in cache_size_to_paths_1.items():
        for cache_size in ["2048", "1024", "4096", "512"]:
            if cache_size not in cache_size_to_paths_1:
                continue 

            run_id_dics = cache_size_to_paths_1[cache_size]

            for deps, run_id in run_id_dics.items():


                # load the right cartridges 
                if cache_size not in cache_size_to_paths_2:
                    continue
                if deps not in cache_size_to_paths_2[cache_size]:
                    continue
                run_id_2 = cache_size_to_paths_2[cache_size][deps]
                WANDB_RUNS = [run_id, run_id_2] 
                model_manager = ModelManager()  # type: ignore
                model_manager.load_model(WANDB_RUNS)

                # run the data through 
                responses, score = run_query_set(model_manager.tokenizer, model_manager.model, comparative_messages, comparative_answers, do_print=do_print)
                print(f"Running query set for {company1} and {company2}; Cache size: {cache_size}; Deps: {deps}; Score: {score}\n")

                if deps not in cache_size_to_deps_to_scores[cache_size]:
                    cache_size_to_deps_to_scores[cache_size][deps] = [score]
                else:
                    cache_size_to_deps_to_scores[cache_size][deps].append(score)
        
                if not do_print:
                    with open(f"{company1}_{company2}_cache_size_to_deps_to_scores_10.json", "w") as f:
                        json.dump(cache_size_to_deps_to_scores, f, indent=4)


def run_baseline_cache():

    cache_size_to_deps_to_scores = collections.defaultdict(dict)

    for pair, inputs in QA_PAIRS.items():

        comparative_messages = inputs["questions"]
        comparative_answers = inputs["answers"]

        if not comparative_answers: 
            continue

        company1, company2 = pair.split("_")
        cache_size_to_paths_1 = CARTRIDGES[company1]
        cache_size_to_paths_2 = CARTRIDGES[company2]

        for company in [cache_size_to_paths_1, cache_size_to_paths_2]:

            for cache_size, run_id_dics in company.items():

                for deps, run_id in run_id_dics.items():

                    WANDB_RUNS = [run_id] 
                    model_manager = ModelManager()  # type: ignore
                    model_manager.load_model(WANDB_RUNS)

                    if model_manager.model is None:
                        continue

                    # run the data through 
                    responses, score = run_query_set(model_manager.tokenizer, model_manager.model, comparative_messages, comparative_answers, do_print=False)
                    print(f"Running query set for {pair}; Cache size: {cache_size}; Deps: {deps}; Score: {score}\n")

                    if deps not in cache_size_to_deps_to_scores[cache_size]:
                        cache_size_to_deps_to_scores[cache_size][deps] = [score]
                    else:
                        cache_size_to_deps_to_scores[cache_size][deps].append(score)
            
                    with open("baseline10_cache_size_to_deps_to_scores.json", "w") as f:
                        json.dump(cache_size_to_deps_to_scores, f, indent=4)

                    break



if __name__ == "__main__":

    do_print = False
    # run_sweep(do_print)

    run_baseline_cache()


