from capsules.clients.anthropic import AnthropicClient
from capsules.configs.common_evals.finance_evals import FinanceEvalMetadata
from capsules.configs.common_evals.finance_evals import (
    ANSWER_SYSTEM_PROMPT_TEMPLATE,
    QUESTION_SYSTEM_PROMPT_TEMPLATE,
)
from capsules.datasets import CapsuleDataset, CapsuleGenerateDataset
from capsules.generate.context_convo_generators.claude_evals.anthropic_q_and_a import (
    AnthropicQandA,
)
from capsules.generate.run import GenerateConfig
from capsules.tasks.finance import FinanceBenchContextConfig
from capsules.train import EvalDatasetConfig, GenerateDatasetConfig
from capsules.transforms import ConvoTransformConfig
from capsules.utils.wandb import WandBConfig


import os

MODEL_TYPE = "claude_qa"


def get_config(
    eval_metadata: FinanceEvalMetadata,
    doc_name: str,
    num_samples: int,
    version_tag: str,
) -> GenerateConfig:
    return GenerateConfig(
        name=eval_metadata.get_artifact_name(
            doc_name, num_samples, version_tag, MODEL_TYPE
        ),
        convo_generator=AnthropicQandA.Config(
            question_system_prompt_template=QUESTION_SYSTEM_PROMPT_TEMPLATE,
            answer_system_prompt_template=ANSWER_SYSTEM_PROMPT_TEMPLATE,
            instructions=eval_metadata.question_prompt_template,
            answer_prompt_template=eval_metadata.answer_prompt_template,
            client=AnthropicClient.Config(),
            question_temperature=1.0,
        ),
        context=FinanceBenchContextConfig(doc_names=[doc_name], force_single_doc=True),
        output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
        # generate config
        num_samples=num_samples,
        batch_size=num_samples,
        max_num_batches_in_parallel=1,
        wandb=WandBConfig(
            project="capsules",
            entity="hazy-research",
        ),
    )


def get_evals(
    evals: list[FinanceEvalMetadata],
    doc_name: str,
    num_samples: int,
    version_tag: str,
    batch_size: int,
    transforms: list[ConvoTransformConfig] = None,
) -> tuple[list[GenerateDatasetConfig], list[EvalDatasetConfig]]:
    generate_configs = []
    eval_configs = []

    for eval_metadata in evals:
        artifact_name = f"hazy-research/capsules/{eval_metadata.get_artifact_name( doc_name, num_samples,version_tag,MODEL_TYPE)}:latest"
        dataset_config_kwargs = dict(
            data_sources=[
                (
                    artifact_name,
                    None,
                )
            ],
            is_wandb=True,
            label_type="tokens",
            convo_transforms=transforms,
        )
        name_for_wandb = f"_anthropic_qa_{doc_name}_tag_{eval_metadata.tag}"

        eval_configs.append(
            EvalDatasetConfig(
                local_batch_size=batch_size,
                dataset=CapsuleDataset.Config(**dataset_config_kwargs),
                name_for_wandb=f"eval_{name_for_wandb}",
                only_eval_rank_0=True,  # RE: eventually, 'lets' fix, but for now, let's not get hozed!
            )
        )

        generate_configs.append(
            GenerateDatasetConfig(
                dataset=CapsuleGenerateDataset.Config(**dataset_config_kwargs),
                name_for_wandb=f"generate_{name_for_wandb}",
            )
        )

    return generate_configs, eval_configs

