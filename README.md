<div align="center">
    <img src="assets/banner.png" height=100 alt="Cartridges logo"/>

**Storing long contexts in tiny KV caches with self-study.**



<!-- ![GitHub Workflow Status](https://github.com/HazyResearch/meerkat/actions/workflows/.github/workflows/ci.yml/badge.svg) -->
[![GitHub](https://img.shields.io/github/license/HazyResearch/cartridges)](https://img.shields.io/github/license/HazyResearch/cartridges)
[![arXiv](https://img.shields.io/badge/arXiv-2402.18668-b31b1b.svg)](https://arxiv.org/abs/2506.06266)

</div>


**What is this?** This repository provides code for training a cartridge, a small KV cache representing amount of textual information. It uses a test-time training recipe called self-study.
The code is based on our paper *[Cartridges: Lightweight and general-purpose long context representations via self-study](https://arxiv.org/abs/2506.06266)*.

**tl;dr** When we put lots of text (*e.g.* a whole code repo) into a language model's context, generation cost soars because of the KV cache's size. *What if we trained a smaller KV cache for our documents offline?* Using a test-time training recipe called self-study, we show that this simple idea can improve throughput by 
26× while maintaining quality. (See our [blogpost](https://hazyresearch.stanford.edu/blog/2025-06-08-cartridges) for more.)


**Table of contents**
- [Setup](#setup)
- [Architecture Overview](#architecture-overview)
- [Synthesizing Training Data with Self-Study](#synthesizing-training-data-with-self-study)
- [Training a Cartridge](#training-a-Cartridge)
- [Serving Cartridges](#serving-Cartridges)
- [Acknowledgments and Citation](#acknowledgments-and-citation)


## Setup

**Step 1:** Clone the repository and install the Python package.

```bash
git clone https://github.com/HazyResearch/cartridges && cd cartridges
pip install uv
uv pip install -e . 
```

**Step 2:** Set some environment variables

The codebase relies on your setting the following variables. We recommend adding them to your `~/.bashrc`, `~/.zshrc`, `DockerFile`, etc.

```bash
# path to your the directory where you cloned this repo
export CARTRIDGES_DIR=/path/to/cartridges

# path to a directory where you want to store outputs like models checkpoints and such
export CARTRIDGES_OUTPUT_DIR=/path/to/cartridges/outputs

# the code in this repository is tightly integrated with wandb
# set your wandb project and entity here
export CARTRIDGES_WANDB_PROJECT=cartridges
export CARTRIDGES_WANDB_ENTITY=
```


## Running Self-Study

**What is self-study?** Self-study is an approach for training a model to understand a corpus of text. It works by generating synthetic conversations about a corpus of text and then training the model on those conversations with a context-distillation objective. The process consists of two AI agents in conversation with one another: one asks questions or makes requests about the content, and another responds using the provided context. 

**Quickstart**: Take a look at the scripts at `examples/arxiv/arxiv_synthesize.py` and `examples/arxiv/arxiv_train.py` for a basic example of how to synthesize training data and run context-distillation on the synthesized data. To run the synthesis script, you will need to spin up an inference server (either [Tokasaurus](https://github.com/ScalingIntelligence/tokasaurus) or [SGLang](https://github.com/sgl-project/sglang)) and set the `client` variable to point to it. [See below for more details on how to do this.](#step-3-prepare-an-inference-server)

Below we walk through the process of generating synthetic training data for a corpus of text. As a running example, we'll be training a cartridge on our [paper on Cartridges](https://arxiv.org/abs/2506.06266). How meta!
<!-- Here are the steps:
1. Synth
1. Configure resources that contain the data you want to store in the cartridge
2. Ensure you have an inference server running (either [Tokasaurus](https://github.com/ScalingIntelligence/tokasaurus) or [SGLang](https://github.com/ScalingIntelligence/tokasaurus)) and configure your client to point to it
3. Instantiate a `SynthesizeConfig` object that contains the parameters for the self-study process
4. Put it all together in one script and run it!
5. Run context-distillation (i.e. training) on the synthesized data -->

> **Note:** For configuration, we use [Pydantic](https://docs.pydantic.dev/latest/) models. Pydantic models are useful for defining the schema of the config and quickly ensuring that the config is valid at the beginning of a run. We also rely on [`pydrantic`](https://github.com/seyuboglu/pydrantic), which provides a few utilities for working with configs.


### Step 1: Synthesize training data
*Note: See `examples/arxiv/arxiv_synthesize.py` for the full example developed in this section.*

Below is the outline of a script for running the synthesis. It simply instantiates a [`SynthesizeConfig`](./cartridges/synthesize.py#L10) object and runs it with `pydrantic.main([config])`. *Note: Using `pydrantic.main` allow us to override the config on the command line like `python your_synthesis_script.py num_samples=1024`.*

The config has a couple of key fields missing: the resource, which controls what data is used , and a client of an inference server (*e.g.* SGLang or Tokasaurus). We'll cover those two below. 
There are many other configuration options we're not covering here, so refer to the [`SynthesizeConfig`](./cartridges/synthesize.py#L10) and [`SelfStudySynthesizer`](./cartridges/synthesizers/self_study.py#L10) for the full list.

```python
import pydrantic

from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer

resource_config = ...  # see 'Step 2: Configure Resources'
client_config = ...  # see 'Step 3: Prepare an Inference Server'

config = SynthesizeConfig(
    synthesizer=SelfStudySynthesizer.Config(
        client=client_config,
        resources=[resource_config],
    ),
    num_samples=512,
    name="cartridges-tutorial",
)

if __name__ == "__main__": 
    pydrantic.main([config])
```


#### Step 1.1: Configure Resources
A "resource" is an object that feeds context and seed prompts to a synthesizer.  For our example, we'll use the `LaTeXResource` type for a research paper.

```python 
from cartridges.resources.latex import LatexResource

resource_config = LaTeXResource.Config(
    arxiv_id="2506.06266",
    seed_prompts=[
        "structuring",
        "summarization",
        "question",
        "use_case",
        "creative",
    ],
    chunker=TokenChunker.Config(
        tokenizer=client.model_name,
        min_tokens_per_chunk=512,
        max_tokens_per_chunk=1024,
    ),
)
```

We provide several other basic resource types like `TextResource`, `FileTextResource`, `JSONResource`.

We're also adding some more specialized resource types like `SlackResource` and `GMailResource`.


#### Step 1.2: Prepare an Inference Server

Self-study requires an inference server to generate the synthetic conversations. We need to configure a [`Client`](./cartridges/clients/base.py#L10) object that points to the inference server. We support two options:
- [Tokasaurus](https://github.com/ScalingIntelligence/tokasaurus) (recommended) - We ran all of our experiments with Tokasaurus, which provides higher throughput generation and is easier to modify. 
- [SGLang](https://github.com/sgl-project/sglang) - We're also providing support for SGLang, but we have not tested it extensively.

<details>
<summary>
Option A: Modal Deployment (Tokasaurus)
</summary>

We found it easy to run data generation with Modal's serverless horizontal scaling.

For cloud deployment, you can deploy on Modal:
```bash
modal deploy infra/modal_deploy_tksrs.py
```

Then configure with the modal URL:
```python
from cartridges.clients.tokasaurus_batch import TokasaurusBatchClient

client_config = TokasaurusBatchClient.Config(
    url="https://your-modal-deployment-url.modal.run",
    model_name="meta-llama/Llama-3.2-3B-Instruct"
)
```
---
</details>


<details>
<summary>
Option B: Local deployment (Tokasaurus)
</summary>

If you have access to GPUs, you can run also run a local Tokasaurus server:

1. Clone and install Tokasaurus:
```bash
git clone https://github.com/ScalingIntelligence/tokasaurus
cd tokasaurus
git checkout --track origin/add-top-logprobs
uv pip install -e .
```

2. Start the server:
```bash
tksrs model=meta-llama/Llama-3.2-3B-Instruct kv_cache_num_tokens='(512 * 1024)' max_top_logprobs=5
```

3. Configure your client:
```python
from cartridges.clients.tokasaurus_batch import TokasaurusBatchClient

client_config = TokasaurusBatchClient.Config(
    port=8001,  # Default Tokasaurus port
    model_name="meta-llama/Llama-3.2-3B-Instruct"
)
```
</details>

<details>
<summary>
Option C: Modal deployment (SGLang)
</summary>

We found it easiest to run data generation with Modal because it provides serverless horizontal scaling.

For cloud deployment, you can deploy on Modal:
```bash
modal deploy infra/modal_deploy_sglang.py
```

Then configure with the modal URL:
```python
from cartridges.clients.sglang import SGLangClient

client_config = SGLangClient.Config(
    url="https://your-modal-deployment-url.modal.run",
    model_name="meta-llama/Llama-3.2-3B-Instruct"
)
```
</details>

<details>
<summary>
Option D: Local deployment (SGLang)
</summary>

1. Install and launch a SGLang server following the instructions [here](https://docs.sglang.ai/start/install.html).
2. Configure your client:
```python
from cartridges.clients.sglang import SGLangClient

client_config = SGLangClient.Config(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    url="http://localhost:8000",
)
```
</details>


#### Step 1.3: Run the Synthesis

Once you've created the script, run it with: 
```bash
python examples/arxiv/arxiv_synthesize.py
```

You can update the config on the command line like `python examples/arxiv/arxiv_synthesize.py num_samples=1024`.

Once the run is complete, it will save the results to a pickle file and print the path:
```bash
>>> Final output saved to /path/to/output/dir/artifact/dataset.pkl
```
Copy this path to your clipboard.

See [`TrainingExample`](./cartridges/structs.py#L10) for the schema of the output.


<details>
<summary>
<i>Exploring in synthesized dataset in the visualization UI</i>
</summary>

```python
import pickle
import pandas as pd

# Load the dataset
with open("/path/to/output/dir/artifact/dataset.pkl", "rb") as f:
    data = pickle.load(f)

rows = data["rows"]
resources = data["resources"]

# Convert to DataFrame for exploration
df = pd.DataFrame([
    {
        "num_messages": len(row.messages),
        "num_output_tokens": row.num_output_tokens,
        "seed_prompt": row.metadata.get("seed_prompt", ""),
        "conversation": "\n".join([f"{msg.role}: {msg.content}" for msg in row.messages])
    }
    for row in rows[:10]  # First 10 examples
])
```

</details>



<details>
<summary>
Exploring synthesized dataset in a DataFrame
</summary>

```python
import pickle
import pandas as pd

# Load the dataset
with open("/path/to/output/dir/artifact/dataset.pkl", "rb") as f:
    data = pickle.load(f)

rows = data["rows"]
resources = data["resources"]

# Convert to DataFrame for exploration
df = pd.DataFrame([
    {
        "num_messages": len(row.messages),
        "num_output_tokens": row.num_output_tokens,
        "seed_prompt": row.metadata.get("seed_prompt", ""),
        "conversation": "\n".join([f"{msg.role}: {msg.content}" for msg in row.messages])
    }
    for row in rows[:10]  # First 10 examples
])
```

</details>


### Step 2: Run context-distillation (i.e. training) on the synthesized data

*Note: See `examples/arxiv/arxiv_train.py` for the full script developed in this section.*


See `cartridges.train.TrainConfig` for the schema of the main config we use for training. Below we provide an example of a config file prefaced with notes describing each part of the config:

```python
import os
from pathlib import Path
import pydrantic

from cartridges.initialization.random import KVFromRandomText
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.train import TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import CartridgeTrainDataset


DATA_SOURCE = "/data/sabri/cartridges/2025-08-08-17-26-20-arxiv_synthesize/arxiv_synthesize_Qwen/Qwen3-4b_n8192-0/artifact/dataset.pkl"


config = TrainConfig(
    model=HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-4b",
        model_cls=FlexQwen3ForCausalLM,
    ),
    kv_cache_initializer=KVFromRandomText.Config(max_tokens=2048),
    
    lr=2e-2,
    epochs=1,
    global_batch_size=32,

    dataset=CartridgeTrainDataset.Config(
        data_sources=[(DATA_SOURCE, None)],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),

    save_every_n_steps=512,
    name="cartridges-tutorial-train",
)


if __name__ == "__main__":
    pydrantic.main(config)
```

### Data parallel training
To launch a data parallel training run, you can run:

```bash
torchrun --standalone --nproc_per_node=2 path/to/file.py
```


<!-- 
### Implementing a new data generation method
To implement a new data generation method:

1. Create a new file in the `cartridges/synthesizers/` directory (e.g., `my_synthesizer.py`)
2. Subclass `ConvoSynthesizer` from `cartridges.synthesizers.base` 
3. Implement the `sample_convos` method that returns a list of `TrainingExample` objects
4. Create a config that uses your new synthesizer class and run it in the same way as above. -->

<!-- 
### Finding the generated dataset in WandB
The data will also be saved to WandB as a pickle file artifact. To find it, go to the WandB project in the UI and click on the "Artifacts" tab. 
You should see an entry on the left with the same name as you provided in the config. Click on it and select the version. (If you run the script multiple times, you'll see multiple versions.) 

To grab the path to the artifact, copy the value in the "Full Name" field shown below.

![image](static/dataset-artifact.png)

For example, here the full path is `hazy-research/cartridges/m03d17_generate_longhealth_p01:v0`. You'll need this path to train a Cartridge on the generated data.
 -->


## Serving Cartridges

We describe two ways to serve and chat with a trained Cartridge: a simple, but slow way that just uses a pure PyTorch generation loop, and a faster one that uses a Tokasaurus server.

### Serving with Tokasuaurus [Fastest and recommended]
We've implemented (h/t @geoffreyangus) an integration with [Tokasaurus](https://github.com/ScalingIntelligence/tokasaurus), a simple LLM inference server optimized for high throughput. 

To run the Tokasaurus server, you will need to (install Tokasaurus from source)[], switch to the branch `geoff/cartridges`, and then follow the instructions [here](https://github.com/ScalingIntelligence/tokasaurus/tree/geoff/cartridges?tab=readme-ov-file#cartridges) to make API calls to the server.

We 

```python
client = TokasaurusClient(
    url="https://your-modal-deployment-url.modal.run",
    model_name="meta-llama/Llama-3.2-3B-Instruct"
)
```

```bash
streamlit run cartridges/analysis/dashboards/chat_w_cache.py
```

### Serving with Basic PyTorch [Easiest but slow]

```bash
streamlit run cartridges/analysis/dashboards/chat_w_cache.py
```

## Acknowledgments and Citation
There are tons of people and organizations who have supported this project. Below we shout out a few, but check out the the paper for a full list.

The compute for this project was provided by [Modal](https://modal.com/) — who made it super easy to scale out horizontally when running the synthetic data generation for self-study — and [Together](https://www.together.ai/) — who provided the compute for training the Cartridges on the synthetic data. [Prime Intellect](https://www.google.com/search?q=prime+intellect&oq=prime+intell&sourceid=chrome&ie=UTF-8&sei=dkNPaKfxNeq50PEPrdqiwA4), [Voltage Park](https://dashboard.voltagepark.com/), and [Azure](https://azure.microsoft.com/en-us/) through the HAI Grants program also contributed compute towards this project.


```bibtex
@article{eyuboglu2025cartridges,
  title={Cartridges: Lightweight and general-purpose long context representations via self-study},
  author={Eyuboglu, Sabri and Ehrlich, Ryan and Arora, Simran and Guha, Neel and Zinsley, Dylan and Liu, Emily and Tennien, Will and Rudra, Atri and Zou, James and Mirhoseini, Azalia and others},
  journal={arXiv preprint arXiv:2506.06266},
  year={2025}
}
```

