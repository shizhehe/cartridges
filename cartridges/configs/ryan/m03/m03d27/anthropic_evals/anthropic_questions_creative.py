import os
from pathlib import Path


import pydrantic


from capsules.clients.anthropic import AnthropicClient
from capsules.clients.tokasaurus import TokasaurusClient

from capsules.generate.context_convo_generators.claude_evals.anthropic_q_client_answer import AnthropicQClientAnswer
from capsules.generate.context_convo_generators.system_prompts import (
    AnswerSystemPromptWithEntireContext,
    AnswerSystemPromptWithEntireContextTruncated,
)
from capsules.generate.run import GenerateConfig
from capsules.generate.chunk import SimpleCharacterChunker
from capsules.tasks.finance import FinanceBenchContextConfig
from capsules.utils import WandBConfig

question_client_config = TokasaurusClient.Config(
    url="http://localhost:8012/v1", model_name="not_empty"
)
answer_client_config = TokasaurusClient.Config(
    url="http://localhost:8012/v1", model_name="not_empty"
)

ANSWER_PROMPT_TEMPLATE = """{question}"""

DOC_NAME = "AMD_2022_10K"

file_name = Path(__file__).stem

INSTR_CREATIVE = """Please generate a question about the document to test someone's ability to comprehend the content of the document. This question specifically should be focused on their ability to generalize the information about the document to a strange question of sorts.

This question shouldn't be a standard question about this kind of document, it should ask to do something abnormal and creative, like writing a poem about a financial document."""

config = GenerateConfig(
    name=file_name + "_higher_temp",
    convo_generator=AnthropicQClientAnswer.Config(
        instructions=INSTR_CREATIVE,
        client=AnthropicClient.Config(),
        temperature=1.0,
        answer_client=answer_client_config,
        answer_temperature=0.0,
        answer_max_completion_tokens=384,
        answer_prompt_template=ANSWER_PROMPT_TEMPLATE,
        answer_system_prompt_generator=AnswerSystemPromptWithEntireContext.Config(),#AnswerSystemPromptWithEntireContextTruncated.Config(
            #max_chars=200_000,
        #),
    ),
    context=FinanceBenchContextConfig(doc_names=[DOC_NAME], force_single_doc=True),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    # generate config
    num_samples=50,
    batch_size=50,
    max_num_batches_in_parallel=1,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
    ),
)


if __name__ == "__main__":
    # Launch pydrantic CLI, which will parse arguments and run config.run() if desired.
    pydrantic.main([config])
