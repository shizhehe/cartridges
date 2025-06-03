import pydrantic

from capsules.analysis.figures.slice_barplots_grouped import SliceBarplots, PlotRunSetConfig


ntp = PlotRunSetConfig(
    wandb_run_ids=["hazy-research/capsules/rnw4tecy"],
    hue="ntp"
)

cartridge = PlotRunSetConfig(
    wandb_run_ids=["hazy-research/capsules/rqq82dys"],
    hue="cartridge"
)

icl = PlotRunSetConfig(
    wandb_run_ids=["hazy-research/capsules/69zpdjh7"],
    hue="icl"
)

DATASET_TO_LABEL = {
    "factual": "Factual Questions",
    "disjoint": "Questions require disjoint info.",
    "reasoning": "Tasks with math. reasoning",
    "creative": "Creative tasks",
    "synthesize": "Tasks requiring synthesis",
    "structure": "Data-structuring tasks",
    "finance-memorization": "Next-Token Prediction",
}


def modify_dataset(dataset: str) -> str:
    if dataset.startswith("eval"):
        return dataset.split("tag_")[-1]
    return dataset

plot = SliceBarplots.Config(
    run_sets=[ntp, cartridge, icl],
    metric="loss",
    hue_order=["ntp", "cartridge", "icl"],
    modify_dataset=modify_dataset,
    score_label="←  log(perplexity)",
    excluded_datasets=["finance-ppl-gt", "memorization", "counting", "coding"],
    dataset_to_label=DATASET_TO_LABEL,
)

if __name__ == "__main__":
    pydrantic.main([plot])

