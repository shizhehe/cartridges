<div align="center">
    <img src="assets/banner.png" height=100 alt="Cartridges logo"/>

**Storing long contexts in tiny KV caches with self-study.**



<!-- ![GitHub Workflow Status](https://github.com/HazyResearch/meerkat/actions/workflows/.github/workflows/ci.yml/badge.svg) -->
[![GitHub](https://img.shields.io/github/license/HazyResearch/cartridges)](https://img.shields.io/github/license/HazyResearch/cartridges)
[![arXiv](https://img.shields.io/badge/arXiv-2402.18668-b31b1b.svg)](https://arxiv.org/abs/2506.06266)

</div>


**What is this?** This repository provides code for training a cartridge, a small KV cache representing a large dump of text, using a test-time training recipe called self-study.
The code is based on our paper *[Cartridges: Lightweight and general-purpose long context representations via self-study](https://arxiv.org/abs/2506.06266)*.

**tl;dr** When we put lots of text (*e.g.* a whole code repo) into a language model's context, generation cost soars because of the KV cache's size. *What if we trained a smaller KV cache for our documents offline?* Using a test-time training recipe called self-study, we show that this simple idea can improve throughput by 
26× while maintaining quality. (See our [blogpost](https://hazyresearch.stanford.edu/blog/2025-06-08-cartridges) for more.)


**Table of contents**
- [Setup](#setup)
- [Generating Training Data with Self-Study](#generating-training-data-with-self-study)
- [Training a Cartridge](#training-a-Cartridge)
- [Serving Cartridges](#serving-Cartridges)
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
```


## Synthesizing Training Data with Self-Study

**What is self-study?** Self-study is a test-time training approach where we generate synthetic conversations about a corpus of text. The process simulates two AI agents: one asks questions or makes requests about the content, and another responds using the provided context. This creates training data that teaches the model to efficiently compress and retrieve information from long contexts.

**Quickstart**: Take a look at the script at `scripts/longhealth_synthesize.py` for an example of how to generate training data with self-study. To actually run the script, you will need to spin up an inference server (either [Tokasaurus](https://github.com/ScalingIntelligence/tokasaurus) or [SGLang](https://github.com/ScalingIntelligence/tokasaurus)) and set the `client` variable to point to it.

Below we walk through the process of generating synthetic training data for a corpus of text in more detail. As a running example, we'll be training a cartridge on our [paper on Cartridges](https://arxiv.org/abs/2506.06266). How meta!
Here are the steps:
1. Create a `StructuredContext` object that contains the data you want to store in the cartridge
2. Ensure you have an inference server running (either [Tokasaurus](https://github.com/ScalingIntelligence/tokasaurus) or [SGLang](https://github.com/ScalingIntelligence/tokasaurus)) and configure your client to point to it
4. Instantiate a `SynthesizeConfig` object that contains the parameters for the self-study process
4. Put it all together in one script and run it!

> **Note:** For configuration, we use [Pydantic](https://docs.pydantic.dev/latest/) models. Pydantic models are useful for defining the schema of the config and quickly ensuring that the config is valid at the beginning of the script. We also rely on [`pydrantic`](https://github.com/seyuboglu/pydrantic), which provides a few utilities for working with configs.

### Step 1: Create a Context Object

A `StructuredContext` represents your corpus in a format that the self-study process can work with. We provide several built-in context types. For our example, we'll use the `TexDocument` context type. 

```python 
config = TexDocument.Config(
    arxiv_src_url="https://arxiv.org/src/2506.06266",
    main_file="main.tex"
)
```

We provide a few other context types including `HTMLDocument`, `TexDocument`.
Can also use an arbitrary JSON object as a context. 

### Step 2: Prepare an Inference Server

Self-study requires an inference server to generate the synthetic conversations. We support two options:
- [Tokasaurus](https://github.com/ScalingIntelligence/tokasaurus) (recommended) - We ran all of our experiments with Tokasaurus, which provides higher throughput generation and is easier to modify. 
- [SGLang](https://github.com/sgl-project/sglang) - We're also providing support for SGLang, but we have not tested it extensively.

<details>
<summary>
Option A: Modal Deployment (Tokasaurus)
</summary>

We found it easiest to run data generation with Modal because it provides serverless horizontal scaling.

For cloud deployment, you can deploy on Modal:
```bash
modal deploy infra/modal_deploy_tksrs.py
```

Then configure with the modal URL:
```python
from cartridges.clients.tokasaurus_batch import TokasaurusBatchClient

client = TokasaurusBatchClient.Config(
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

client = TokasaurusBatchClient.Config(
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

client = SGLangClient.Config(
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

client = SGLangClient.Config(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    url="http://localhost:8000",
)
```
</details>


### Step 3: Configure the Synthesizer


**Synthesizer Configuration**:
```python
from cartridges.synthesizers.self_study import SelfStudySynthesizer, SlicePromptSamplerWithChunks

synthesizer_config = SelfStudySynthesizer.Config(
    client=client,
    tokenizer="meta-llama/Llama-3.2-3B-Instruct",
    max_rounds=1,  # Number of conversation rounds
    
    # Configure the prompt sampler
    prompt_sampler=SlicePromptSamplerWithChunks.Config(
        # The 
        slices=["structuring", "summarization", "question", "use_case", "creative"],
        min_chunk_size=512,   # Minimum context chunk size in tokens
        max_chunk_size=4096,  # Maximum context chunk size in tokens
        desc="Below is a research paper on test-time training for long contexts."
    ),
    
    # Chain-of-thought reasoning
    prob_cot_a=0.2,  # Probability of using CoT for agent A
    
    # Tool usage (optional)
    use_tools=False,
    tools=[]
)
```

### Step 4: Putting it all together

We are now going to put all of the pieces together in a `SynthesizeConfig` object that configures the entire self-study process.

**Core Settings**:
- `num_samples`: Total number of training examples to generate
- `batch_size`: Number of training examples to generate per call to the inference server.
- `max_num_batches_in_parallel`: Number of batches to process concurrently. When using Modal, high values 

Here's a complete example script:

```python
import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus_batch import TokasaurusBatchClient
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer, SlicePromptSamplerWithChunks
from cartridges.utils import WandBConfig
from cartridges.tasks.longhealth.context import LongHealthStructuredContextConfig


client = TokasaurusBatchClient.Config(
    url="https://hazyresearch--tksrs-entry-capsules-3b-1xh100-min0-max64-serve.modal.run",
    ports=None,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

context = TexDocument.Config(
    arxiv_src_url="https://arxiv.org/src/2506.06266",
    main_file="main.tex"
)

config = SynthesizeConfig(
    context=context,
    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        tokenizer="meta-llama/Llama-3.2-3B-Instruct",
        max_rounds=1,
        prompt_sampler=SlicePromptSamplerWithChunks.Config(
            slices=["structuring", "summarization", "question", "use_case", "creative"],
            min_chunk_size=512,
            max_chunk_size=4096,
            desc=f"Below is a research paper on test-time training for long contexts."
        ),
        prob_cot_a=0.2,
        use_tools=False, 
        tools=[]
    ),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    num_samples=512,
    batch_size=16,
    max_num_batches_in_parallel=4,
    handle_exceptions=True,  # Continue if individual batches fail
    save_wandb_artifact=True,

    name="cartridges-tutorial",
    wandb=WandBConfig(project="cartridges", entity="hazy-research"),
)


if __name__ == "__main__": 
    pydrantic.main([config])
```

### Running the Synthesis

Once you've created the file, run it with: 
```bash
python your_synthesis_script.py
```
### Output Format

```python 
class TrainingExample(BaseModel):
    messages: List[Message]  # The conversation between agents (system, user, assistant format)
    token_ids: List[int]  # The token IDs for the response
    top_logprob_ids: List[List[int]]  # The top-k token predictions at each position
    top_logprob_logprobs: List[List[float]]  # The corresponding log probabilities
    metadata: Dict[str, Any]  # Information about tool usage, prompts, and generation process
```

Once the run is complete, it will save the results to a pickle file and print the path:
```bash
Final output saved to /path/to/output/dir/artifact/dataset.pkl
```

You can explore the generated data in a notebook:
```python
import pickle
import pandas as pd

# Load the dataset
with open("/path/to/output/dir/artifact/dataset.pkl", "rb") as f:
    data = pickle.load(f)

rows = data["rows"]
context = data["context"]

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

### Advanced Configuration

#### Custom Prompt Samplers

You can create custom prompt samplers for specialized conversation types:

```python
from cartridges.synthesizers.self_study import PromptSampler

class CustomPromptSampler(PromptSampler):
    class Config(PromptSampler.Config):
        domain_specific_prompts: List[str]
    
    def __call__(self, batch_idx: int, num_convos: int) -> tuple[str, List[str]]:
        # Sample context chunk
        context_chunk = self._sample_context_chunk()
        
        # Generate domain-specific seed prompts
        seed_prompts = [
            random.choice(self.config.domain_specific_prompts)
            for _ in range(num_convos)
        ]
        
        return context_chunk, seed_prompts
```

#### Tool Integration

You can enhance the self-study process with tools that allow agents to dynamically retrieve additional context:

```python
from cartridges.tools.base import Tool

# Define custom tools for information retrieval
tools = [
    SearchTool.Config(description="Search for specific information"),
    SummaryTool.Config(description="Generate summaries of sections")
]

synthesizer_config = SelfStudySynthesizer.Config(
    # ... other config ...
    use_tools=True,
    tools=tools
)
```

### Implementing a new data generation method
To implement a new data generation method:

1. Create a new file in the `cartridges/synthesizers/` directory (e.g., `my_synthesizer.py`)
2. Subclass `ConvoSynthesizer` from `cartridges.synthesizers.base` 
3. Implement the `sample_convos` method that returns a list of `TrainingExample` objects
4. Create a config that uses your new synthesizer class and run it in the same way as above.

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
The data will also be saved to WandB as a pickle file artifact. To find it, go to the WandB project in the UI and click on the "Artifacts" tab. 
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

## Serving Cartridges

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



