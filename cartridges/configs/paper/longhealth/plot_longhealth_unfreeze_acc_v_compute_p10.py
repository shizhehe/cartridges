
import pydrantic

from capsules.analysis.figures.quality_v_compute_ablation import QualityVsComputePlotter,PlotRunSetConfig
from capsules.utils.cache_size import MODEL_TO_CACHE_SIZE_FN
from capsules.configs.paper.longhealth.plot_longhealth_lora_acc_v_size_p10 import cache_tuning, lora

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
    wandb_run_ids=["hazy-research/capsules/xoajifmc"],
    hue=lambda row: f"bfreeze_prefix-{row['num_trainable_tokens']}",
    modify_x=lambda row, x: row["generate_mmlu/score"],
    filter=lambda row: row["state"] == "finished",
)

cache_tuning_unfreeze = PlotRunSetConfig(
    wandb_run_ids=["hazy-research/capsules/jv73wvdm"],
    hue=lambda row: f"aunfreeze_prefix-{row['num_trainable_tokens']}",
    modify_x=lambda row, x: row["generate_mmlu/score"],
    filter=lambda row: row["state"] == "finished",
)


run_sets = [
    cache_tuning,
    cache_tuning_unfreeze,
]

configs = []

for excluded in ["mmlu", "longhealth_mc"]:

    avg_plot = QualityVsComputePlotter.Config(
        run_sets=run_sets,
        y_metric="score",
        dataset_type="generate",
        excluded_datasets=[excluded],
        y_label="Accuracy",
        x_label="Train Steps",
        hue_label="Method",
        bin_size=512,
        run_id="avg_plot",
        legend_loc="best",
    )
    configs.append(avg_plot)

if __name__ == "__main__":
    pydrantic.main(configs)
