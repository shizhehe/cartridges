import math
from cartridges.analysis.figures.quality_v_cache_size_w_groups import (
    PlotRunSetConfig,
    QualityVsCacheSizePlotter,
)
import pydrantic

from cartridges.utils.cache_size import MODEL_TO_CACHE_SIZE_FN

def convert_x_to_bytes(row, x):
    model_name = row["model.pretrained_model_name_or_path"]
    x_in_bytes = MODEL_TO_CACHE_SIZE_FN[model_name](x)
    return x_in_bytes / (1e9)

def convert_x_to_bytes_baseline(row, x):
    model_name = row["generator.client.model_name"]
    x_in_bytes = MODEL_TO_CACHE_SIZE_FN[model_name](x)
    return x_in_bytes / (1e9)

cache_tuning = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/Cartridges/uyer83th", # 256
        "hazy-research/Cartridges/tlluhdrl", # 512
        "hazy-research/Cartridges/tc8gem0p", # 1024
        "hazy-research/Cartridges/x6si1jpf", # 2048
        "hazy-research/Cartridges/b8k71rp3", # 4096
    ],
    hue="cartridge",
    modify_x=lambda row, x: convert_x_to_bytes(row, x + row["num_trainable_tokens"]),
    shade=lambda row: - math.log(row["num_trainable_tokens"], 2),
    filter=lambda row: row["state"] == "finished",
)

lora = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/Cartridges/6s54xbxw", # 51
        "hazy-research/Cartridges/tps67d46", # 102
        "hazy-research/Cartridges/b2ibz0i0", # 204
        "hazy-research/Cartridges/rkhvr0y5", # 408
        "hazy-research/Cartridges/mwhzbxc8", # 816
        # "hazy-research/Cartridges/", # 1632
    ],
    hue="lora_rank",
    modify_x=lambda row, x: convert_x_to_bytes(row, x) + row["num_trainable_params"] * 2 / 1e9,
    shade=lambda row: - math.log(row["model.peft.r"], 2),
    filter=lambda row: row["state"] == "finished",

)

# icl_baseline = PlotRunSetConfig(
#     wandb_run_ids=["hazy-research/Cartridges/gcc2a7pa"],
#     hue="icl",
#     baseline=True,
#     modify_x=lambda row, x: convert_x_to_bytes_baseline(row, x),
# )

# kv_cache_compression_expected = PlotRunSetConfig(
#     launch_ids=["2025-05-11-00-06-06-llama_3b"],
#     hue="KV Cache Compression (Expected)",
#     modify_x=lambda row, x: row[f"generate_longhealth_mc/kv_cache_size_bytes"] / 1e9,
# )


run_sets = [
    cache_tuning,
    lora,
]

configs = []

mmlu_plot = QualityVsCacheSizePlotter.Config(
    run_sets=run_sets,
    y_metric="score",
    dataset_type="generate",
    excluded_datasets=["mmtob-ek-test", "mmtob-ke-test"],
    show_baseline_points=True,
    x_scale_base=2,
    y_label="Accuracy",
    x_label="Cache Size (GB)",
    hue_label="Method",
    run_id="mmlu_plot",
    legend_loc="best",
)
configs.append(mmlu_plot)

ke_plot = QualityVsCacheSizePlotter.Config(
    run_sets=run_sets,
    y_metric="batch_score",
    dataset_type="generate",
    excluded_datasets=["mmtob-ek-test", "mmlu"],
    show_baseline_points=True,
    x_scale_base=2,
    y_label="ChRF",
    x_label="Cache Size (GB)",
    hue_label="Method",
    run_id="ke_plot",
    legend_loc="best",
)
configs.append(ke_plot)

if __name__ == "__main__":
    pydrantic.main(configs)
