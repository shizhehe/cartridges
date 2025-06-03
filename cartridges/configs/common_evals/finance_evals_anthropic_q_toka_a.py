import os
from capsules.clients.tokasaurus import TokasaurusClient
from capsules.configs.common_evals.finance_evals import FinanceEvalMetadata
from capsules.generate.context_convo_generators.claude_evals.model_eval_from_existing import (
    GenerateAnswersToExistingQuestions,
)
from capsules.generate.context_convo_generators.system_prompts import (
    AnswerSystemPromptWithEntireContextTruncated,
)
from capsules.generate.run import GenerateConfig
from capsules.tasks.finance import FinanceBenchContextConfig
from capsules.utils.wandb import WandBConfig
from capsules.configs.common_evals import finance_evals_anthropic

MODEL_TYPE = "llama3b"

answer_client_config = TokasaurusClient.Config(
    url="http://localhost:8012/v1", model_name="not_empty"
)


def get_config(
    eval_metadata: FinanceEvalMetadata,
    doc_name: str,
    num_samples: int,
    version_tag: str,
):
    return GenerateConfig(
        name=eval_metadata.get_artifact_name(
            doc_name, num_samples, version_tag, f"claude_q_{MODEL_TYPE}_a"
        ),
        convo_generator=GenerateAnswersToExistingQuestions.Config(
            answer_prompt_template=eval_metadata.answer_prompt_template,
            answer_system_prompt_generator=AnswerSystemPromptWithEntireContextTruncated.Config(
                max_chars=400_000,
            ),
            answer_client=answer_client_config,
            answer_temperature=1.0,
            directory_or_artifact=f"{eval_metadata.get_artifact_name(
                doc_name,
                num_samples,
                version_tag,
                finance_evals_anthropic.MODEL_TYPE,
            )}:latest",
            is_wandb=True,
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
