
import pydrantic

from cartridges.analysis.figures.quality_v_compute_ablation import QualityVsComputePlotter,PlotRunSetConfig
from cartridges.utils.cache_size import MODEL_TO_CACHE_SIZE_FN


def convert_x_to_bytes(row, x):
    model_name = row["model.pretrained_model_name_or_path"]
    x_in_bytes = MODEL_TO_CACHE_SIZE_FN[model_name](x)
    return x_in_bytes / (1e9)

def convert_x_to_bytes_baseline(row, x):
    model_name = row["generator.client.model_name"]
    x_in_bytes = MODEL_TO_CACHE_SIZE_FN[model_name](x)
    return x_in_bytes / (1e9)

IDXS = [-3, -1]

cache_tuning = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/Cartridges/wauoq23f", # 2048
        "hazy-research/Cartridges/xd1b7oi7", # 8192
        
    ],
    hue=lambda row: f"cfirst_k_tokens_{row['num_trainable_tokens'] + 1}",
)


random_vec = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/Cartridges/l4c70ken", # 2048
        "hazy-research/Cartridges/216fvin7", # 8192
    ],
    hue=lambda row: f"brandom_vec_{row['num_trainable_tokens'] + 1}",
    modify_x=lambda row, x: convert_x_to_bytes(row, x + row["num_trainable_tokens"]),
    filter=lambda row: row["state"] == "finished",
)

random_tokens = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/Cartridges/hhzv7xb5", # 2048
        "hazy-research/Cartridges/ijibddb0", # 8192
    ],
    hue=lambda row: f"arandom_tokens_{row['num_trainable_tokens'] + 1}",
    modify_x=lambda row, x: convert_x_to_bytes(row, x + row["num_trainable_tokens"]),
    filter=lambda row: row["state"] == "finished",
)


run_sets = [
    cache_tuning,
    random_vec,
    random_tokens,
]

configs = []


avg_plot = QualityVsComputePlotter.Config(
    run_sets=run_sets,
    y_metric="score",
    dataset_type="generate",
    include_datasets=["longhealth_mc"],
    y_label="Accuracy",
    x_label="Train Steps",
    hue_label="Method",
    bin_size=768,
    run_id="avg_plot",
    legend_loc="best",
)
configs.append(avg_plot)

if __name__ == "__main__":
    pydrantic.main(configs)
