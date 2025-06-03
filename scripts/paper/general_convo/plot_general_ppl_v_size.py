



from cartridges.analysis.figures.quality_v_cache_size_w_groups import (
    PlotRunSetConfig,
    QualityVsCacheSizePlotter,
)
import pydrantic

from cartridges.utils.cache_size import MODEL_TO_CACHE_SIZE_FN


def modify_dataset(dataset: str) -> str:
    if dataset.startswith("eval"):
        return dataset.split("tag_")[-1]
    return dataset

def convert_x_to_bytes(row, x):
    model_name = row["model.pretrained_model_name_or_path"]
    x_in_bytes = MODEL_TO_CACHE_SIZE_FN[model_name](x)
    return x_in_bytes / (1e9)

def convert_x_to_bytes_baseline(row, x):
    model_name = "meta-llama/Llama-3.2-3B-Instruct" #row["generator.client.model_name"]
    x_in_bytes = MODEL_TO_CACHE_SIZE_FN[model_name](x)
    return x_in_bytes / (1e9)


DATASET_TO_LABEL = {
    "factual": "Factual Questions",
    "disjoint": "Questions require disjoint info.",
    "reasoning": "Tasks with math. reasoning",
    "creative": "Creative tasks",
    "synthesize": "Tasks requiring synthesis",
    "structure": "Data-structuring tasks",
    "finance-memorization": "Next-Token Prediction",
}


cartridge = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/Cartridges/8q2ooljx", # 4096
        "hazy-research/Cartridges/1dwmf6v5", # 2048
        "hazy-research/Cartridges/li740c5l", # 1024
        "hazy-research/Cartridges/tawoblj9", # 512
        "hazy-research/Cartridges/ebbr94lu", # 8192
    ],
    hue="cartridge",
    modify_x=lambda row, x: convert_x_to_bytes(row, x + row["num_trainable_tokens"]),
)

ntp = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/Cartridges/1qn4kqvs", # 2048
        "hazy-research/Cartridges/4jgp59jw", # 4096
        "hazy-research/Cartridges/5a0pc86r", # 1024
        "hazy-research/Cartridges/n8xdpyji", # 512
        "hazy-research/Cartridges/s6rc1xgz", # 8192
    ],
    hue="ntp",
    modify_x=lambda row, x: convert_x_to_bytes(row, x + row["num_trainable_tokens"]),
    step=64
)

truncate = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/Cartridges/1qn4kqvs", # 2048
        "hazy-research/Cartridges/4jgp59jw", # 4096
        "hazy-research/Cartridges/5a0pc86r", # 1024
        "hazy-research/Cartridges/n8xdpyji", # 512
        "hazy-research/Cartridges/s6rc1xgz", # 8192
    ],
    hue="first_k_tokens",
    modify_x=lambda row, x: convert_x_to_bytes(row, x + row["num_trainable_tokens"]),
    step=0
)

baseline_runs = PlotRunSetConfig(
    wandb_run_ids=["hazy-research/Cartridges/69zpdjh7"],
    hue="icl",
    baseline=True,
    modify_x=lambda row, x: convert_x_to_bytes(row, x + row["num_trainable_tokens"]),
    )

run_sets = [
    cartridge,
    ntp,
    baseline_runs,
    truncate,

    # kv_compression_expected_attention
]

avg_plot = QualityVsCacheSizePlotter.Config(
    run_sets=run_sets,
    y_metric="loss",
    dataset_type="eval",
    show_baseline_points=True,
    y_label="log (Perplexity) (←)",
    x_label="Cache Size (GB)",
    hue_label="Method",
    run_id="avg_plot",
    legend_loc="best",
    excluded_datasets=["finance-ppl-gt", "memorization", "counting", "coding"],

    # y_min=0.0,
    y_max=3,
)

if __name__ == "__main__":
    pydrantic.main([
        avg_plot
    ])


