
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
    model_name = "meta-llama/Llama-3.2-3B-Instruct" #row["generator.client.model_name"]
    x_in_bytes = MODEL_TO_CACHE_SIZE_FN[model_name](x)
    return x_in_bytes / (1e9)

# ---- begin amd ----
amd_cache_tuning = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/Cartridges/7rswu4j2", # 512
        "hazy-research/Cartridges/xxzeonzo", # 1024
        "hazy-research/Cartridges/1llykntp", # 2048
        "hazy-research/Cartridges/x1m29og7", # 4096
        "hazy-research/Cartridges/bor45311", # 8192
    ],
    group=lambda row: row["num_trainable_tokens"],
    hue="Cache Tuning",
    modify_x=lambda row, x: convert_x_to_bytes(row, x + row["num_trainable_tokens"]),
)

amd_baseline_runs = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/Cartridges/oa6bgna6"       # AMD
    ],
    group="icl",
    hue="ICL Baseline",
    baseline=True,
    modify_x=lambda row, x: convert_x_to_bytes(row, x + row["num_trainable_tokens"]),
)
# ---- end amd ----

# ---- begin amex ----
amex_cache_tuning = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/Cartridges/jljxpbwy", # 512
        "hazy-research/Cartridges/xhsirr3q", # 1024
        "hazy-research/Cartridges/e5lz0ovj", # 2048
        "hazy-research/Cartridges/bwfr4jum", # 4096
        "hazy-research/Cartridges/l9kexozg"
    ],
    group=lambda row: row["num_trainable_tokens"],
    hue="Cache Tuning",
    modify_x=lambda row, x: convert_x_to_bytes(row, x + row["num_trainable_tokens"]),
)

amex_baseline_runs = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/Cartridges/6hayclix"       
    ],
    group="icl",
    hue="ICL Baseline",
    baseline=True,
    modify_x=lambda row, x: convert_x_to_bytes(row, x + row["num_trainable_tokens"]),
)
# ---- end amex ----

# ---- begin boeing ----
boeing_cache_tuning = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/Cartridges/9ct54wed", # 512
        "hazy-research/Cartridges/sc2bhdsx", # 1024
        "hazy-research/Cartridges/48yw22k9", # 2048
        "hazy-research/Cartridges/ax4cg2aa", # 4096
        "hazy-research/Cartridges/tkhajsus", # 8192
    ],
    group=lambda row: row["num_trainable_tokens"],
    hue="Cache Tuning",
    modify_x=lambda row, x: convert_x_to_bytes(row, x + row["num_trainable_tokens"]),
)

boeing_baseline_runs = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/Cartridges/bqjoo55z", # 512
    ],
    group="icl",
    hue="ICL Baseline",
    baseline=True,
    modify_x=lambda row, x: convert_x_to_bytes(row, x + row["num_trainable_tokens"]),
)
# ---- end boeing ----

# ---- begin pepsico ----
pepsico_cache_tuning = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/Cartridges/xq3wqffq", # 512
        "hazy-research/Cartridges/daz444xv", # 1024
        "hazy-research/Cartridges/o05yguu1", # 2048
        "hazy-research/Cartridges/amavku4n", # 4096
        "hazy-research/Cartridges/bdpaqkg8", # 8192
    ],
    group=lambda row: row["num_trainable_tokens"],
    hue="Cache Tuning",
    modify_x=lambda row, x: convert_x_to_bytes(row, x + row["num_trainable_tokens"]),
)

pepsico_baseline_runs = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/Cartridges/zp9x8bri",
    ],
    group="icl",
    hue="ICL Baseline",
    baseline=True,
    modify_x=lambda row, x: convert_x_to_bytes(row, x + row["num_trainable_tokens"]),
)
# ---- end pepsico ----
run_sets = [
    amd_cache_tuning,
    amd_baseline_runs,
    amex_cache_tuning,
    amex_baseline_runs,
    boeing_cache_tuning,
    boeing_baseline_runs,
    pepsico_cache_tuning,
    pepsico_baseline_runs,
]

avg_plot = QualityVsCacheSizePlotter.Config(
    run_sets=run_sets,
    y_metric="loss",
    dataset_type="eval",
    excluded_datasets=[],
    show_baseline_points=True,
    y_label="log(perplexity) (←)",
    x_label="Cache Size (GB)",
    hue_label="Method",
    run_id="avg_plot",
    legend_loc="best",
)

if __name__ == "__main__":
    pydrantic.main([
        avg_plot
    ])





