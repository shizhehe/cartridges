
import pydrantic

from capsules.analysis.figures.quality_v_compute_ablation import QualityVsComputePlotter,PlotRunSetConfig
from capsules.utils.cache_size import MODEL_TO_CACHE_SIZE_FN
from capsules.configs.paper.qasper.plot_qasper_lora_ppl_v_size import cache_tuning, lora

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
    wandb_run_ids=[cache_tuning.wandb_run_ids[i] for i in IDXS],
    hue=lambda row: f"prefix-{row['num_trainable_tokens']:04d}",
    shade=lambda row: row["num_trainable_tokens"],
    modify_x=lambda row, x: row["generate_mmlu/score"],
    filter=lambda row: row["state"] == "finished",
)

lora = PlotRunSetConfig(
    wandb_run_ids=[lora.wandb_run_ids[i] for i in IDXS],
    hue=lambda row: f"lora-{row['model.peft.r']:04d}",
    shade=lambda row: row["model.peft.r"],
    modify_x=lambda row, x: row["generate_mmlu/score"],
    filter=lambda row: row["state"] == "finished",

)


run_sets = [
    cache_tuning,
    lora,
]

configs = []


avg_plot = QualityVsComputePlotter.Config(
    run_sets=run_sets,
    y_metric="score",
    dataset_type="generate",
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
