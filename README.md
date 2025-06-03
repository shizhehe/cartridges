<div align="center">
    <img src="assets/banner.png" height=100 alt="Cartridges logo"/>

**Lightweight and general-purpose long-context representations.**



<!-- ![GitHub Workflow Status](https://github.com/HazyResearch/meerkat/actions/workflows/.github/workflows/ci.yml/badge.svg) -->
[![GitHub](https://img.shields.io/github/license/HazyResearch/cartridges)](https://img.shields.io/github/license/HazyResearch/cartridges)
[![arXiv](https://img.shields.io/badge/arXiv-2402.18668-b31b1b.svg)](https://arxiv.org/abs/2402.18668)



<!-- [**Website**](http://meerkat.wiki)
| [**Quickstart**](http://meerkat.wiki/docs/start/quickstart-df.html)
| [**Docs**](http://meerkat.wiki/docs/index.html)
| [**Contributing**](CONTRIBUTING.md)
| [**Discord**](https://discord.gg/pw8E4Q26Tq)
| [**Blogpost**](https://hazyresearch.stanford.edu/blog/2023-03-01-meerkat) -->

</div>

## What is this?

Blah...

<div align="center">
    <img src="assets/tradeoff-plots.png" height=70 alt="Cartridges logo"/>
</div>

**Table of contents**
- [Setup](#setup)
- [Generating Training Data with Self-Study](#generating-training-data-with-self-study)
- [Training a Cartridge](#training-a-Cartridge)
- [Analysis](#analysis)


## Setup

**Step 1:** Clone the repository and install the Python package.

```bash
git clone https://github.com/HazyResearch/cartridges && cd cartridges
pip install uv
uv pip install -e . 
```
Note: there *might* be some missing dependencies -- I haven't checked recently. 
If you run into missing dependencies, just add them to the `pyproject.toml` file and/or pip install them individually. 

**Step 2:** Set some environment variables

The codebase relies on your setting the following variables. We recommend adding them to your `~/.bashrc`, `~/.zshrc`, `DockerFile`, etc.

```bash
# path to your the directory where you cloned this repo
export CARTRIDGES_DIR=/path/to/cartridges

# path to a directory where you want to store outputs like models checkpoints and such
export CARTRIDGES_OUTPUT_DIR=/path/to/cartridges/outputs

# OPTIONAL: Your API keys for any cloud providers you want to use (don't need to set all)
export TOGETHER_API_KEY=...
export OPENAI_API_KEY=...
```


## Generating Training Data with Self-Study

For configuration of experiments, we use [Pydantic](https://docs.pydantic.dev/latest/) models.
Pydantic models are useful for defining the schema of the config and quickly ensuring that the config is valid at the beginning of the script. We also rely on `pydrantic`, which provides a few utilities for working with configs. 

To generate training data for a document, you can create a config file under `cartridges/configs/{your_name}/generate/` that looks like this (see `cartridges/configs/sabri/generate/m03d17_generate_longhealth_p01.py` for an example): 

```python 
import os
from pathlib import Path
import pydrantic

from cartridges.clients.together import TogetherClient
from cartridges.generate.generate_basic import GenerateDatasetConfig, GenerateSettings

# Make sure you have a valid API key for Together in your environment (e.g. put it in your ~/.bashrc)
client_config = TogetherClient.Config(
    model_name="meta-llama/Llama-3.2-3B-Instruct-Turbo",
)

file_name = Path(__file__).stem

config = GenerateConfig(
    name=file_name,
    convo_generator=SimpleQuestionFromChunk.Config(
        question_client=client_config,
        question_temperature=0.6,
        question_max_completion_tokens=256,
        answer_client=client_config,
        answer_temperature=0.0,
        answer_max_completion_tokens=256,
        chunker=SimpleCharacterChunker.Config(
            min_chunk_size_in_chars=500,
            max_chunk_size_in_chars=10_000,
        ),
        question_system_prompt_generator=QuestionSystemPromptWithEntireContext.Config(),
        answer_system_prompt_generator=AnswerSystemPromptWithChunk.Config(),
    ),
    document_title="Large Language Monkeys: Scaling Inference Compute with Repeated Sampling",
    document_path_or_url=str(MONKEYS.absolute()),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    # generate config
    num_samples=8192,
    batch_size=128,
    max_num_batches_in_parallel=20,
    wandb=WandBConfig(
        project="cartridges",
        entity="hazy-research",
    ),
)

if __name__ == "__main__":
    # Launch pydrantic CLI, which will parse arguments and run config.run() if desired.
    pydrantic.main([config])
```

Important parameters to set: 
- `document_path_or_url`: this should point to a url or a local file that contains the document for which you want to generate training data. Add any good documents you find to the `data/example_docs` directory.
- `question_client` and `answer_client`: update this config to change the models and model arguments (e.g. temperature) used for the question and answer clients. See `cartridges/clients` for all the available clients.

See all the parameters that control the generation in `cartridges.generate.generate_basic.GenerateDatasetConfig`.

Once you've created the file run it with: 
```bash
python cartridges/configs/generate/your_file_name.py
```

Once the run is complete it will write the results to a [feather file](https://pandas.pydata.org/docs/reference/api/pandas.read_feather.html) and print the path to the file to the terminal. The output will look like:

```bash
Running 1 configs
Wrote dataset to /path/to/output/dir/2025-03-02-14-29-27-m03d01_generate_bee_movie/df249742-3d4a-4a99-a2d9-896fbfac190f/dataset.feather
```

Explore the generations in a notebook with:
```python
import pandas as pd
df = pd.read_feather("/path/to/output/dir/2025-03-02-14-29-27-m03d01_generate_bee_movie/df249742-3d4a-4a99-a2d9-896fbfac190f/dataset.feather")
```




### Implementing a new data generation method
To implement a new data generation method:

1. Create a new file in the `cartridges/generate/{your_name}` directory (e.g., `generate_my_method.py`)
2. Subclass `BaseGenerateConfig` from `cartridges.generate.base` 
3. Implement the `_make_dataset` method that returns a `QADataset`
4. Create a config file in `cartridges/configs/{your_name}/generate/` that uses your new class and run it in the same way as above.

### Using Tokasaurus for data generation
We do most of our data generation with [Tokasaurus](https://github.com/jordan-benjamin/tokasaurus/tree/add-top-logprobs). 

If you have access to GPUs, you can run a server with:

1. Clone the repo: 
```bash
git clone https://github.com/jordan-benjamin/tokasaurus
cd tokasaurus
```
2. Checkout the `add-top-logprobs` branch: `git checkout --track origin/add-top-logprobs`
3. Install the package: `uv pip install -e .`
4. Start a server with:
```bash
tksrs model=meta-llama/Llama-3.2-3B-Instruct kv_cache_num_tokens='(512 * 1024)' max_top_logprobs=5
```
5. Update the config file to use a `TokasaurusClient` config:
```python
client_config = TokasaurusClient.Config(
    port=8001, # this should point to the port where the server is running
)
```

Alternatively, you can simply deploy a server on modal (or ask Sabri to deploy one for you):
```
modal deploy scratch/sabri/modal/deploy_llama_3b_modal.py
```
Then you update the config file to use the modal url:
```python
client_config = TokasaurusClient.Config(
    url="https://hazyresearch--tokasaurus-llama-3b-dp4-serve.modal.run/v1",
    model_name="llama-3.2-3b-instruct"  # this argument does not matter
)
```


### Finding the generated dataset in WandB
The data will also be saved to WandB as a feather file artifact. To find it go to the WandB artifact, go to the `cartridges` project in the UI and click on the "Artifacts" tab. 
You should see an entry on the left with the same name as you provided in the config. Click on it and select the version. (If you run the script multiple times, you'll see multiple versions.) 

To grab the path to the artifact, copy the value in the "Full Name" field shown below.

![image](static/dataset-artifact.png)

For example, here the full path is `hazy-research/cartridges/m03d17_generate_longhealth_p01:v0`. You'll need this path to train a Cartridge on the generated data. 



## Training a Cartridge 

To try launching a first experiment, you can try running on   
```
python cartridges/configs/sabri/train/m03d17_train_longhealth_p01.py
```

See `cartridges.train.TrainConfig` for the schema of the main config we use for training. 

Here is an example of a config file: 

```python 
import os
from pathlib import Path

import pydrantic

from cartridges.initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from cartridges.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from cartridges.config import HFModelConfig
from cartridges.datasets import CartridgeDataset
from cartridges.tasks.longhealth import LongHealthEvalDataset, LongHealthMultipleChoiceGenerateDataset
from cartridges.utils import WandBConfig

file_name = Path(__file__).stem
config = TrainConfig(
    name=f"{file_name}",
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
    ),
    dataset=CartridgeDataset.Config(
        data_sources=[
            # This is the path to the generated dataset we created above
            ("hazy-research/cartridges/m03d17_generate_longhealth_p01:v0", None),
        ],  
        is_wandb=True,
        label_type="logits",
        top_k_logits=20,
    ),
    generate_every_n_steps=64,
    generate_datasets=[
        GenerateDatasetConfig(
            name_for_wandb="multiple_choice_generations",
            dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                patient_ids=["patient_01"],
                max_questions=128,
                cot=True
            )
        ),
    ],
    eval_every_n_steps=16,
    eval_datasets=[
        EvalDatasetConfig(
            name_for_wandb="generated_questions",
            batch_size=16,
            dataset=CartridgeDataset.Config(
                data_sources=[
                    # This is the path to another test dataset we created with gpt4o using
                    # the approach as above
                    ("hazy-research/cartridges/m03d12_longhealth_p01_basic_qa_test:v0", None),
                ],
                is_wandb=True,
                label_type="tokens",
            ),
        )
    ],
    generate_max_new_tokens=512,
    kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
        max_tokens=2048
    ),
    loss_type="logits",
    save_every_n_steps=64,
    epochs=1,
    lr=5e-3,
    wandb=WandBConfig(
        project="cartridges",
        tags=["cache_tuning", "development"],
        entity="hazy-research",
    ),
    output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
    train_batch_size=6,
)

if __name__ == "__main__":
    pydrantic.main([config])
```

### Distributed data parallel training
To launch a data parallel training run, you can run:

```bash
torchrun --standalone --nproc_per_node=2 cartridges/configs/train/m03d09_train_memorization_and_qa_no_sum.py
```

## Analysis

### Chatting with a Cartridge 

If you have access to some idle GPUs on a cluster you can SSH into, just launch a streamlit app and make sure the port is forwarded to your local: 
```bash
streamlit run cartridges/analysis/dashboards/chat_w_cache.py
```
Our training scripts log the cache to WandB and the streamlit app just pulls it from there. So, no need  to run the app on the same machine you trained the model.

If you don't have GPUs, you can also deploy the app on Modal with: 
```bash
modal serve scratch/sabri/modal/streamlit_chat_w_cache.py
```

### Exploring generations in WandB

The documentation for the `wandb` query language, which you use to explore tables is shockingly bad. Here are some commands I use to look at the generations:
```javascript
// visualize one big table with all the generations across steps
runs.history.concat["test/results_step"]

// filter to a particular step
runs.history.concat["test/results_step"].table.rows.concat.filter((row) => row["global_step"] == 0)
runs.history.concat["eval_finance-ppl/table"].table.rows.concat.filter((row) => row["optimizer_step"] == 0)

```