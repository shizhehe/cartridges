from cartridges.analysis.figures.quality_v_cache_size_w_groups import (
    PlotRunSetConfig,
    QualityVsCacheSizePlotter,
)
import pydrantic

from cartridges.utils.cache_size import MODEL_TO_CACHE_SIZE_FN

def convert_x_to_bytes(row, x):
    model_name = row["model.pretrained_model_name_or_path"]
    x_in_bytes = MODEL_TO_CACHE_SIZE_FN[model_name.lower()](x)
    return x_in_bytes / (1e9)

def convert_x_to_bytes_baseline(row, x):
    model_name = "meta-llama/Llama-3.2-3B-Instruct" #row["generator.client.model_name"]
    x_in_bytes = MODEL_TO_CACHE_SIZE_FN[model_name.lower()](x)
    return x_in_bytes / (1e9)

cache_tuning = PlotRunSetConfig(
    wandb_run_ids=[
        # OLD WITHOUT MACRO
        # "hazy-research/capsules/uexjfegh",  # 128
        # "hazy-research/capsules/on2vi932",  # 256
        # "hazy-research/capsules/98l6voho",  # 512
        # "hazy-research/capsules/wk3speis",  # 1024
        # "hazy-research/capsules/mx62gf6d",  # 2048
        # "hazy-research/capsules/gc72d8fn", # 4096
        # "hazy-research/capsules//zn73e3ej",  # 8192

        "hazy-research/capsules/6cy15cyw",  # 128
        "hazy-research/capsules/3y6svmxn",  # 256
        # "hazy-research/capsules/",  # 512
        "hazy-research/capsules/tzhe958g",  # 1024
        "hazy-research/capsules/bcyg44th",  # 2048
        "hazy-research/capsules/8auiop80", # 4096
        "hazy-research/capsules/q1sebt5s",  # 8192
  
    ],
    hue="cartridge",
    modify_x=lambda row, x: convert_x_to_bytes(row, x + row["num_trainable_tokens"]),
)

baseline_runs = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/capsules/c5bkriun",
    ],
    hue="icl",
    baseline=True,
    modify_x=lambda row, x: convert_x_to_bytes(row, x),
)

kv_compression_duo = PlotRunSetConfig(
    launch_ids=["2025-05-11-23-25-21-pplbaseline_qasper_icl_kv_duo"],
    hue="duo",
    modify_x=lambda row, x: row["kv_size_bytes"] / (1e9),
)


# kv_compression_expected_attention = PlotRunSetConfig(
#     launch_ids=["2025-05-12-10-59-45-pplbaseline_qasper_icl_kv_duo"],
#     hue="duo",
#     modify_x=lambda row, x: row["kv_size_bytes"] / (1e9),
# )

firstk_runs = PlotRunSetConfig(
    launch_ids=[
        "2025-05-12-16-16-29-pplbaseline_qasper_firstk"
    ],
    hue="first_k_tokens",
    modify_x=lambda row, x: convert_x_to_bytes(row, x),
)

run_sets = [
    cache_tuning,
    baseline_runs,
    kv_compression_duo, 
    firstk_runs,
    # kv_compression_expected_attention
]

avg_plot = QualityVsCacheSizePlotter.Config(
    run_sets=run_sets,
    y_metric="macro_loss",
    dataset_type="eval",
    excluded_datasets=[],
    show_baseline_points=True,
    y_label="log (Perplexity) (←)",
    x_label="Cache Size (GB)",
    hue_label="Method",
    run_id="avg_plot",
    legend_loc="best",
    # y_min=0.0,
    y_max=2.28,
)

if __name__ == "__main__":
    pydrantic.main([
        avg_plot
    ])





