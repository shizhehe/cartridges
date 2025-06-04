import os
from pathlib import Path


import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus_batch import TokasaurusBatchClient
from cartridges.clients.sglang import SGLangClient
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer, SlicePromptSamplerWithChunks
from cartridges.utils import WandBConfig

from cartridges.tasks.longhealth.context import LongHealthStructuredContextConfig


client = TokasaurusBatchClient.Config(
    url="https://hazyresearch--tksrs-entry-capsules-3b-1xh100-min0-max64-serve.modal.run",
    ports=None,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

client = SGLangClient.Config(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    url="https://hazyresearch--sglang-llama-3-2-3b-instruct-h100-serve.modal.run",
)

NUM_PATIENTS = 10
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]

config = SynthesizeConfig(
    name=FormatStringVariable(f"{Path(__file__).stem}_{patients_str}_n{{num_samples}}"),
    run_id=FormatStringVariable("{name}"),
    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        tokenizer="meta-llama/Llama-3.2-3B-Instruct",
        max_rounds=1,
        prompt_sampler=SlicePromptSamplerWithChunks.Config(
            slices=[
                "structuring",
                "summarization",
                "question",
                "use_case",
                "creative",
            ],
            min_chunk_size=512,
            max_chunk_size=4096,
            desc=f"Below is a section of a patient's medical record. It is part of a larger corpus of medical records for {NUM_PATIENTS} different patients."
        ),
        prob_cot_a=0.2,
        use_tools=False, 
        tools=[]
    ),
    context=LongHealthStructuredContextConfig(patient_ids=patient_ids),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    num_samples=512,
    batch_size=16,
    max_num_batches_in_parallel=0,
    handle_exceptions=False,
    save_wandb_artifact=True,
    wandb=WandBConfig(
        project="cartridges",
        entity="hazy-research",
        tags=[f"longhealth", "generate", f"patients_{patients_str}"],
    ),
)


if __name__ == "__main__": 
    pydrantic.main([config])
