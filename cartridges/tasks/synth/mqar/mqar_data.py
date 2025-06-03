import numpy as np
from capsules.generate.context_convo_generators.base import QuestionData
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset

from capsules.tasks.synth.config import DataSegmentConfig
from capsules.tasks.synth.data import DataSegment, DataSegmentForTuning, TuningDataset

@dataclass
class MQARData:
    inputs: torch.Tensor
    labels: torch.Tensor
    slices: Optional[dict] = None

class MQARConfig(DataSegmentConfig):
    vocab_size: int
    num_examples: int
    input_seq_len: int
    power_a: float = 0.01
    num_kv_pairs: int = 8
    random_non_queries: bool = False
    include_slices: bool = True

    def build(self, seed: int) -> list[QuestionData]:
        return build_mqar_segment(**self.model_dump(), seed=seed)

    def build_for_tuning(self, seed: int) -> list[QuestionData]:
        return build_mqar_segment_for_tuning(**self.model_dump(), seed=seed)


class MQARTuningDataset(TuningDataset):
    """This is a dataset that prepares data for cache tuning on MQAR. 
    """
    def __init__(
        self, 
        kvs: torch.Tensor,  # a tensor/list of shape (num_kv_pairs * 2)
    ):
        self.kvs = kvs

    def sample(self, max_seq_len: int):
        input_idxs = torch.randint(0, (self.kvs.shape[1] // 2), (max_seq_len,)) * 2
        output_idxs = input_idxs + 1
        inputs = self.kvs[input_idxs]
        outputs = self.kvs[output_idxs]
        return inputs, outputs


def build_mqar_segment_for_tuning(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float=0.01,
    num_kv_pairs: int=8,
    random_non_queries: bool=True,
    include_slices: bool=True,
    **kwargs
) -> DataSegmentForTuning:
    """
    TODO: this function needs to be actually implemetned!!
    """
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    assert num_kv_pairs * 4 <= input_seq_len

    np.random.seed(seed)

    # two tokens for key and value
    context_size = num_kv_pairs * 2

    # create keys so that each key is present exactly once in each example
    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys_unshuffled = np.tile(key_choices, (num_examples, 1))
    keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs)

    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs)

    # create sequences
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    # compute power law
    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a-1)
    p = p / p.sum()

    x = np.stack([np.arange(space, dtype=int)] * num_examples)
    gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs)

    # queries and answers
    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)
    examples = np.concatenate([
        kvs, 
        queries
    ], axis=1)

    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=values, axis=1)

    inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])
    
    # replace all the 0 with random values
    if random_non_queries:
        inputs[inputs == 0] = torch.randint(vocab_size, size=inputs.shape)[inputs == 0]
    return DataSegmentForTuning(
        tuning_dataset=MQARTuningDataset(kvs=kvs), # TODO: check that this is correct!!!),
        query_inputs=inputs, 
        query_targets=labels, 
        slices={"num_kv_pairs": num_kv_pairs, "input_seq_len": input_seq_len}
    )

def build_mqar_segment(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float=0.01,
    num_kv_pairs: int=8,
    random_non_queries: bool=True,
    include_slices: bool=True,
    **kwargs
) -> DataSegment:
    """
    Generates synthetic data for the multi-query associative recall task as described in
    Arora,Eyuboglu, et al. "Zoology: Measuring and improving recall in efficient language models.".

    Example: 
        `multiquery_ar(vocab_size=12, num_kv_pairs=2, input_seq_len=16, random_non_queries=False)` 
        will generate input and label sequences of the form: 
                
                Key   Val  Key  Val            Query                         Query
        Inputs: 2     8    4    7    0    0    4    0    0    0    0    0    2    0    0 
        Labels: -100 -100 -100 -100 -100 -100  7    -100 -100 -100 -100 -100 8    -100 -100

        The -100 labels are ignored by the loss function and metrics.
    
    We include one important note on the power law distribution. In real language data, 
    the gap between repeated bigrams follows a power law. Intuitively, if the bigram
    "common buzzard" appears in text, the probability of the bigram appearing again 
    drops the further away from the orginal mention we are. In our synthetic, we can 
    control this with the power law parameters `train_power_a` and `test_power_a`. 
    Setting these to 1.0 will result in a uniform distribution. You can visualize the
    distribution with the following code:
    ```
    space = 100
    power_a = 0.01  
    p = power_a * np.arange(1, space + 1) ** (power_a-1)
    p = p / p.sum()
    plt.plot(p)
    ```

    Args:
        vocab_size (int): The size of the vocabulary. As discussed in the Zoology 
            paper, large vocabulary sizes (>1k) can be important for highlighting 
            differences between model architectures. Defaults to 8_192.
        num_train_examples (int): The number of training examples to generate. Defaults 
            to 100_000.
        num_test_examples (int): The number of test examples to generate. Defaults to 
            3_000.
        input_seq_len (int): The length of the input sequence. Defaults to 64. In 
            In Figure 2 of the Zoology paper, we vary the input sequence length from 
            64 to 512 and the number of key-value pairs from 4 to 64.
        seed (int): The seed for the random number generator.
        num_kv_pairs (int): The number of key-value pairs.
        train_power_a (float, optional): The power for the power law distribution for 
            training data. Defaults to 0.01.
        test_power_a (float, optional): The power for the power law distribution for 
            test data. Defaults to 0.01.
        random_non_queries (bool, optional): If True, replace all the 0's (as in the 
            example above) with random values in the input. Defaults to True.

    Returns:
        SyntheticData: A SyntheticData object containing the generated train and test 
            inputs and labels.

    Raises:
        Warning: If potential data leakage is detected between the train and test sets.
    """
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    assert num_kv_pairs * 4 <= input_seq_len

    np.random.seed(seed)

    # two tokens for key and value
    context_size = num_kv_pairs * 2

    # create keys so that each key is present exactly once in each example
    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys_unshuffled = np.tile(key_choices, (num_examples, 1))
    keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs)

    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs)

    # create sequences
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    # compute power law
    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a-1)
    p = p / p.sum()

    x = np.stack([np.arange(space, dtype=int)] * num_examples)
    gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs)

    # queries and answers
    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)
    examples = np.concatenate([
        kvs, 
        queries
    ], axis=1)

    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=values, axis=1)

    inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])
    
    # replace all the 0 with random values
    if random_non_queries:
        inputs[inputs == 0] = torch.randint(vocab_size, size=inputs.shape)[inputs == 0]
    return DataSegment(
        inputs, 
        labels, 
        slices={"num_kv_pairs": num_kv_pairs, "input_seq_len": input_seq_len}
    )


def multiquery_ar(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float = 0.01,
    num_kv_pairs: int = 8,
    random_non_queries: bool = True,
    include_slices: bool = True,
    **kwargs
) -> MQARData:
    """
    Generates synthetic data for the multi-query associative recall task as described in
    Arora,Eyuboglu, et al. "Zoology: Measuring and improving recall in efficient language models.".

    Example: 
        `multiquery_ar(vocab_size=12, num_kv_pairs=2, input_seq_len=16, random_non_queries=False)` 
        will generate input and label sequences of the form: 
                
                Key   Val  Key  Val            Query                         Query
        Inputs: 2     8    4    7    0    0    4    0    0    0    0    0    2    0    0 
        Labels: -100 -100 -100 -100 -100 -100  7    -100 -100 -100 -100 -100 8    -100 -100

        The -100 labels are ignored by the loss function and metrics.
    
    We include one important note on the power law distribution. In real language data, 
    the gap between repeated bigrams follows a power law. Intuitively, if the bigram
    "common buzzard" appears in text, the probability of the bigram appearing again 
    drops the further away from the orginal mention we are. In our synthetic, we can 
    control this with the power law parameters `train_power_a` and `test_power_a`. 
    Setting these to 1.0 will result in a uniform distribution. You can visualize the
    distribution with the following code:
    ```
    space = 100
    power_a = 0.01  
    p = power_a * np.arange(1, space + 1) ** (power_a-1)
    p = p / p.sum()
    plt.plot(p)
    ```

    Args:
        vocab_size (int): The size of the vocabulary. As discussed in the Zoology 
            paper, large vocabulary sizes (>1k) can be important for highlighting 
            differences between model architectures. Defaults to 8_192.
        num_train_examples (int): The number of training examples to generate. Defaults 
            to 100_000.
        num_test_examples (int): The number of test examples to generate. Defaults to 
            3_000.
        input_seq_len (int): The length of the input sequence. Defaults to 64. In 
            In Figure 2 of the Zoology paper, we vary the input sequence length from 
            64 to 512 and the number of key-value pairs from 4 to 64.
        seed (int): The seed for the random number generator.
        num_kv_pairs (int): The number of key-value pairs.
        train_power_a (float, optional): The power for the power law distribution for 
            training data. Defaults to 0.01.
        test_power_a (float, optional): The power for the power law distribution for 
            test data. Defaults to 0.01.
        random_non_queries (bool, optional): If True, replace all the 0's (as in the 
            example above) with random values in the input. Defaults to True.

    Returns:
        SyntheticData: A SyntheticData object containing the generated train and test 
            inputs and labels.

    Raises:
        Warning: If potential data leakage is detected between the train and test sets.
    """
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    assert num_kv_pairs * 4 <= input_seq_len

    np.random.seed(seed)

    # two tokens for key and value
    context_size = num_kv_pairs * 2

    # create keys so that each key is present exactly once in each example
    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys_unshuffled = np.tile(key_choices, (num_examples, 1))
    keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs)

    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs)

    # create sequences
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    # compute power law
    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a-1)
    p = p / p.sum()

    x = np.stack([np.arange(space, dtype=int)] * num_examples)
    gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs)

    # queries and answers
    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)
    examples = np.concatenate([
        kvs, 
        queries
    ], axis=1)

    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=values, axis=1)

    inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])
    
    # replace all the 0 with random values
    if random_non_queries:
        inputs[inputs == 0] = torch.randint(vocab_size, size=inputs.shape)[inputs == 0]

    """print("start")
    for i, j in zip(inputs[0], labels[0]):
        print(i, j)
    print("end")
    return MQARData(
        inputs=inputs,
        labels=labels,
        slices={"num_kv_pairs": num_kv_pairs, "input_seq_len": input_seq_len} if include_slices else None
    )"""
    # Convert to QA format 
    question_datas = []
    
    for i in range(num_examples):
        # Create context in natural language
        context = []
        for k, v in zip(keys[i], values[i]):
            context.append(f"Key {k} has value {v}.")
        context_str = " ".join(context)
        
        # For each key-value pair, generate a question
        for k, v in zip(keys[i], values[i]):
            question = f"{k}"
            
            # Create QuestionData object
            question_data = QuestionData(
                question=question,
                sample=None,
                metadata={
                    "context": context_str,
                    "key": k,
                    "value": v,
                    "num_kv_pairs": num_kv_pairs,
                    "example_idx": i
                },
                chunk=context_str  # Store the context as the chunk
            )
            question_datas.append(question_data)
    
    return question_datas


