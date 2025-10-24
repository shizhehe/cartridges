import math
from capsules.analysis.figures.quality_v_cache_size_w_groups import (
    PlotRunSetConfig,
    QualityVsCacheSizePlotter,
)
import pydrantic

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

cache_tuning = PlotRunSetConfig(
    wandb_run_ids=cache_tuning.wandb_run_ids,
    hue="cartridge",
    shade=lambda row: - math.log(row["num_trainable_tokens"], 2),
    modify_x=lambda row, x: row["eval_qasper_rewrite/macro_loss"],
    filter=lambda row: row["state"] == "finished",
)

lora = PlotRunSetConfig(
    wandb_run_ids=lora.wandb_run_ids,
    hue="lora_rank",
    shade=lambda row: -math.log(row["model.peft.r"], 2),
    modify_x=lambda row, x: row["eval_qasper_rewrite/macro_loss"],
    filter=lambda row: row["state"] == "finished",

)

icl = PlotRunSetConfig(
    raw_data=[
        {
            "generate_mmlu/num_system_and_user_tokens": 0,
            "generate_mmlu/score": 0.53125, # ve4ztpi
            "eval_qasper_rewrite/macro_loss": 1.25, # gcc2a7pa
        }
    ],
    hue="icl",
    modify_x=lambda row, x: row["eval_qasper_rewrite/macro_loss"],
)


icl_empty = PlotRunSetConfig(
    raw_data=[
        {
            "generate_mmlu/num_system_and_user_tokens": 0,
            "generate_mmlu/score": 0.580, # lu7wc8rb
            "eval_qasper_rewrite/macro_loss": 0, # ryq2d61k
        }
    ],
    hue="icl_empty",
    modify_x=lambda row, x: row["eval_qasper_rewrite/macro_loss"],
    baseline=True
)


run_sets = [
    cache_tuning,
    lora,
    icl,
    icl_empty
]

configs = []

avg_plot = QualityVsCacheSizePlotter.Config(
    run_sets=run_sets,
    y_metric="score",
    dataset_type="generate",
    excluded_datasets=[],
    x_label="log (Perplexity) (←)",
    y_label="Accuracy (MMLU)",
    hue_label="Method",
    run_id="avg_plot",
    legend_loc="best",
    x_scale="linear",
    show_lines=False,
    show_baseline_points=False,
)
configs.append(avg_plot)

if __name__ == "__main__":
    pydrantic.main(configs)
