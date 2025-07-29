# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

"""
Create a dataset jsonl file for variable tracking.

python variable_tracking.py \
    --save_dir=./ \
    --save_name=vt \
    --tokenizer_path='EleutherAI/gpt-neox-20b' \
    --tokenizer_type=hf \
    --max_seq_length=4096 \
    --tokens_to_generate=30 \
    --num_samples=10 \
    --random_seed=42 \
    --num_chains=1 \
    --num_hops=4 \
    --template="[INST] Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{context}\nQuestion: Find all variables that are assigned the value {query} in the text above. [/INST] Answer: According to the chain(s) of variable assignment in the text above, {num_v} variables are assgined the value {query}, they are: "
"""
from copy import copy
from dataclasses import dataclass
import hashlib
import os
import re
import json
from typing import List, Literal, Tuple
import uuid
import numpy as np
from pathlib import Path
import pydrantic
from tqdm import tqdm
import random
import sys
from collections import defaultdict
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize
import logging
from pydrantic import BaseConfig, RunConfig
import string
import heapq

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


CONTEXT_TEMPLATE = """[INST] Memorize and track the chain(s) of variable assignment hidden in the following text.

{context}
Question: Find all variables that are assigned the value {query} in the text above. [/INST] Answer: According to the chain(s) of variable assignment in the text above, {num_v} variables are assgined the value {query}, they are:"""



from cartridges.data.ruler.constants import TASKS

class VariableTrackingConfig(BaseConfig):
    max_seq_length: int = 100_000
    num_samples: int = 1
    tokens_to_generate: int = 30
    tokenizer: str = "Qwen/Qwen3-4B"

    context_template: str = CONTEXT_TEMPLATE

    num_chains: int = 1
    num_hops: int = 4

    type_haystack: Literal['essay', 'noise'] = 'noise'
    remove_newline_tab: bool = False
    
    model_template_token: int = 0
    seed: int = 42
    

class GenerateVariableTrackingConfig(RunConfig):
    variable_tracking: VariableTrackingConfig
    save_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")

    def run(self):
        # Set random seeds
        random.seed(self.variable_tracking.seed)
        np.random.seed(self.variable_tracking.seed)
                
        # Load tokenizer
        tokenizer = HFTokenizer(self.variable_tracking.tokenizer)


        # get a hash of the config
        tokenizer_str = self.variable_tracking.tokenizer.split("/")[-1].replace("-", "_").lower()
        config_hash = hashlib.sha256(str(self.variable_tracking.model_dump()).encode()).hexdigest()[:8]
        config_str = f"{tokenizer_str}-l{self.variable_tracking.max_seq_length}-n{self.variable_tracking.num_samples}-c{self.variable_tracking.num_chains}-h{self.variable_tracking.num_hops}-{self.variable_tracking.type_haystack}-{config_hash}"
        save_file = Path(self.save_dir) / f'{config_str}.json'
        save_file.parent.mkdir(parents=True, exist_ok=True)
        samples = generate_samples(
            config=self.variable_tracking,
            tokenizer=tokenizer
        )
        # Helper function to convert dataclass instances to dictionaries recursively
        def dataclass_to_dict(obj):
            if isinstance(obj, list):
                return [dataclass_to_dict(item) for item in obj]
            elif hasattr(obj, "__dataclass_fields__"):
                return {field: dataclass_to_dict(getattr(obj, field)) for field in obj.__dataclass_fields__}
            else:
                return obj

        # Convert samples to JSON format
        samples_json = {
            "config": self.variable_tracking.model_dump(),
            "samples": [dataclass_to_dict(sample) for sample in samples]
        }
        
        # Write the JSON data to the specified file
        with open(save_file, 'w') as f:
            json.dump(samples_json, f, indent=4)
        
        # write_manifest(save_file, write_jsons)
        print(f"Saved {len(samples_json['samples'])} samples to {save_file}")

        # Debug print
        print("Sample context:")
        print(samples[0].context[:500] + "...")
        print(f"\nNumber of queries: {len(samples[0].queries)}")
        for i, query in enumerate(samples[0].queries):
            print(f"Query {i+1}: {query.query}")
            print(f"Answers: {query.answers}")
            print()



@dataclass
class VariableTrackingQuery:
    query: str
    answers: List[str]
    answer_prompt: str

@dataclass
class VariableTrackingSample:
    context: str
    queries: List[VariableTrackingQuery]

class HFTokenizer:
    def __init__(self, model_path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def text_to_tokens(self, text: str) -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def tokens_to_text(self, tokens: List[int]) -> str:
        text = self.tokenizer.convert_tokens_to_string(tokens)
        return text



def get_haystack(type_haystack: str):
    """Get haystack content based on type."""
    needle = "One of the special magic {type_needle_v} for {key} is: {value}."
    if type_haystack == 'essay':
        essay = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data/PaulGrahamEssays.json")
        essay = json.load(open(essay))['text']
        haystack = re.sub(r'\s+', " ", essay).split(" ")
    elif type_haystack == 'noise':
        haystack = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    elif type_haystack == 'needle':
        haystack = needle
    else:
        raise NotImplementedError(f'{type_haystack} is not implemented.')
    return haystack


# Words
import wonderwords
nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
adjs = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
# verbs = wonderwords.random_word._get_words_from_text_file("verblist.txt")
words = [f"{adj}-{noun}" for adj in adjs for noun in nouns]
words = sorted(list(set(words)))


# Positions
DEPTHS = list(np.round(np.linspace(0, 100000, num=40, endpoint=True)).astype(int))

def generate_chains(num_chains, num_hops, is_icl=False):
    vars_all = []
    k = 5 if not is_icl else 3
    num_hops = num_hops if not is_icl else min(10, num_hops)
    vars_all = [''.join(random.choices(string.ascii_uppercase, k=k)).upper() for _ in range((num_hops+1) * num_chains)]
    while len(set(vars_all)) < num_chains * (num_hops+1):
        vars_all.append(''.join(random.choices(string.ascii_uppercase, k=k)).upper())

    vars_ret = []
    chains_ret = []
    used_values = set()
    
    for i in range(0, len(vars_all), num_hops+1):
        this_vars = vars_all[i:i+num_hops+1]
        vars_ret.append(this_vars)
        if is_icl:
            this_chain = [f"VAR {this_vars[0]} = 12345"]
        else:
            # Generate unique initial value
            while True:
                value = str(np.random.randint(10000, 99999))
                if value not in used_values:
                    used_values.add(value)
                    break
            this_chain = [f"VAR {this_vars[0]} = {value}"]
        for j in range(num_hops):
            this_chain.append(f"VAR {this_vars[j+1]} = VAR {this_vars[j]}")
        chains_ret.append(this_chain)
    return vars_ret, chains_ret

def shuffle_sublists_heap(lst):
    heap = []
    for i in range(len(lst)):
        heapq.heappush(heap, (random.random(), i, 0))  # Push first element of each list with random priority
    shuffled_result = []
    while heap:
        _, list_idx, elem_idx = heapq.heappop(heap)  # Get the lowest random priority element
        shuffled_result.append(lst[list_idx][elem_idx])

        # If there are more elements in the same sublist, add the next one
        if elem_idx + 1 < len(lst[list_idx]):
            heapq.heappush(heap, (random.random(), list_idx, elem_idx + 1))
    return shuffled_result


def generate_random_number(num_digits=7):
    lower_bound = 10**(num_digits - 1)
    upper_bound = 10**num_digits - 1
    return str(random.randint(lower_bound, upper_bound))

def generate_random_word():
    word = random.choice(words)
    return word

def generate_random_uuid():
    return str(uuid.UUID(int=random.getrandbits(128), version=4))

def generate_random(type_needle: str):
    if type_needle == 'numbers':
        return generate_random_number()
    elif type_needle == 'words':
        return generate_random_word()
    elif type_needle == 'uuids':
        return generate_random_uuid()
    else:
        raise NotImplementedError(f'{type_needle} is not implemented.')



def generate_input_output(num_noises, config: VariableTrackingConfig, is_icl=False):
    vars_chains, chains = generate_chains(config.num_chains, config.num_hops, is_icl=is_icl)
    
    haystack = get_haystack(config.type_haystack)

    if config.type_haystack == 'essay':
        if num_noises <= len(haystack):
            text = " ".join(haystack[:num_noises])
        else:
            # Repeat haystack as many times as needed and slice to num_noises
            repeats = (num_noises + len(haystack) - 1) // len(haystack)  # Ceiling division
            text = " ".join((haystack * repeats)[:num_noises])
        document_sents = sent_tokenize(text.strip())
        chains_flat = shuffle_sublists_heap(chains)
        insertion_positions = [0] + \
                              sorted([int(len(document_sents) * (depth / 100)) for depth in random.sample(DEPTHS, len(chains_flat))]) + \
                              [len(document_sents)]
        document_sents_list = []
        for i in range(1,len(insertion_positions)):
            last_pos = insertion_positions[i-1]
            next_pos = insertion_positions[i]
            document_sents_list.append(" ".join(document_sents[last_pos:next_pos]))
            if i-1 < len(chains_flat):
                document_sents_list.append(chains_flat[i-1].strip() + ".")
        context = " ".join(document_sents_list)

    elif config.type_haystack == 'noise':
        sentences = [haystack] * num_noises
        for chain in chains:
            positions = list(sorted(random.sample(range(len(sentences)), len(chain))))
            for insert_pi, j in zip(positions, range(len(chain))):
                sentences.insert(insert_pi+j, chain[j])
        context = "\n".join(sentences)

    context = context.replace(". \n", ".\n")

    # Create queries - one for each chain
    queries = []
    for i, (vars_chain, chain) in enumerate(zip(vars_chains, chains)):
        # Get the initial value from the first assignment in this chain
        initial_value = chain[0].split("=")[-1].strip()
        
        # All variables in this chain should resolve to the initial value
        answers = vars_chain  # All variables in the chain
        
        query_text = f"Find all variables that are assigned the value {initial_value} in the text above."
        
        answer_prompt = f"According to the chain(s) of variable assignment in the text above, {len(answers)} variables are assigned the value {initial_value}. List only the variable names (without the VAR keyword) inside <answer></answer> tags, one per line:"
        
        queries.append(VariableTrackingQuery(
            query=query_text,
            answers=answers,
            answer_prompt=answer_prompt
        ))
    
    # Create context template without the query (since we'll have multiple queries)
    context_only_template = """Memorize and track the chain(s) of variable assignment hidden in the following text.

{context}"""
    
    formatted_context = context_only_template.format(context=context)
    
    sample = VariableTrackingSample(
        context=formatted_context,
        queries=queries
    )

    return sample

def generate_samples(config: VariableTrackingConfig, tokenizer: HFTokenizer, incremental: int = 10):
    write_jsons = []
    max_seq_length = config.max_seq_length - config.model_template_token

    if config.type_haystack == 'essay':
        incremental = 500
    elif config.type_haystack == 'noise':
        incremental = 10

    if config.type_haystack != 'essay' and config.max_seq_length < 4096:
        incremental = 5

    # Estimate tokens per question to determine reasonable upper bound
    sample = generate_input_output(incremental, config, is_icl=False)
    sample_input_text = sample.context + sample.queries[0].query
    sample_tokens = len(tokenizer.text_to_tokens(sample_input_text))
    tokens_per_haystack = sample_tokens / incremental

    # Let's do 3x to allow for some slack since we can get unlucky due to sampling.
    estimated_max_noises = int((max_seq_length / tokens_per_haystack) * 3)

    # Binary search for optimal haystack size
    lower_bound = incremental
    upper_bound = max(estimated_max_noises, incremental * 2)

    optimal_num_noises = None

    logger.info(f"Estimated {tokens_per_haystack:.1f} tokens per haystack")
    logger.info(f"Starting binary search with bounds: {lower_bound} to {upper_bound}")
    
    while lower_bound <= upper_bound:
        mid = (lower_bound + upper_bound) // 2
        sample = generate_input_output(mid, config, is_icl=False)
        test_input = sample.context + sample.queries[0].query
        total_tokens = len(tokenizer.text_to_tokens(test_input)) + config.tokens_to_generate

        logger.info(f"Testing haystack size: {mid}, resulting tokens: {total_tokens}/{max_seq_length}")

        if total_tokens <= max_seq_length:
            optimal_num_noises = mid
            lower_bound = mid + 1
        else:
            upper_bound = mid - 1

    num_noises = optimal_num_noises if optimal_num_noises is not None else incremental
    logger.info(f'Final optimal haystack size (number of haystack): {num_noises}')

    # Generate samples
    samples = []
    for index in tqdm(range(config.num_samples)):
        used_noises = num_noises
        for _ in range(1000):
            try:
                sample = generate_input_output(used_noises, config, is_icl=False)
                test_input = sample.context + sample.queries[0].query
                if config.remove_newline_tab:
                    test_input = ' '.join(test_input.replace('\n', ' ').replace('\t', ' ').strip().split())
                    sample.context = ' '.join(sample.context.replace('\n', ' ').replace('\t', ' ').strip().split())
                length = len(tokenizer.text_to_tokens(test_input)) + config.tokens_to_generate
                assert length <= max_seq_length, f"{length} exceeds max_seq_length."
                break
            except Exception as e:
                logger.warning(f"Error generating sample: {e}")
                if used_noises > incremental:
                    used_noises -= incremental
        else:
            raise ValueError("Failed to generate samples")
            
        samples.append(sample)
    
    return samples

if __name__ == "__main__":
    config = GenerateVariableTrackingConfig(
        variable_tracking=VariableTrackingConfig(
            seed=42,
            context_template=CONTEXT_TEMPLATE,
            num_chains=128,
            num_hops=3,
            type_haystack='noise',
            tokens_to_generate=128,
            num_samples=1,
            tokenizer="Qwen/Qwen3-4B",
            max_seq_length=100_000,  # Much smaller for testing
        ),
    )
    pydrantic.main([config])
