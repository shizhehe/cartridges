import random

import pydrantic
from capsules.configs.common_evals.finance_evals import FinanceEvals
from capsules.configs.common_evals.finance_evals_anthropic_q_toka_a import get_config

DOC_NAME = "AMD_2022_10K"

configs = {}
for eval_metadata in FinanceEvals:
    print(f"Generating {eval_metadata.tag} questions for {DOC_NAME}...")

    configs[eval_metadata.tag] = get_config(
        eval_metadata,
        DOC_NAME,
        num_samples=16,
        version_tag="v1",
    )


if __name__ == "__main__":
    # Launch pydrantic CLI, which will parse arguments and run config.run() if desired.
    random.seed(42)
    for tag, config in configs.items():
        print(f"Running {tag} questions for {DOC_NAME}...")
        pydrantic.main([config])
