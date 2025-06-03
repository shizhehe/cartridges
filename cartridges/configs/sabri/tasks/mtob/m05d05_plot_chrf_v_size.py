from capsules.analysis.figures.quality_v_cache_size_per_dataset import (
    QualityVsCacheSizePerDatasetPlotter,
)
from capsules.analysis.figures.quality_v_cache_size import (
    PlotRunSetConfig,
    QualityVsCacheSizePlotter,
)
import pydrantic

from capsules.utils.cache_size import MODEL_TO_CACHE_SIZE_FN

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

        
        "hazy-research/capsules/lmipg82t", # 512
        "hazy-research/capsules/58krekz4", # 1024
        "hazy-research/capsules/lsw41yuc", # 8192
        # "hazy-research/capsules/msad7zj8", # 2048
        # "hazy-research/capsules/hltr6yg3", # 4096
        # "hazy-research/capsules/c5tyjmep", # 8192
        
    ],
    hue="Cache Tuning",
    modify_x=lambda row, x: convert_x_to_bytes(row, x + row["num_trainable_tokens"]),
    step=384
)

# openai_runs = PlotRunSetConfig(
#     launch_ids=[
#         "2025-04-23-09-18-08-m04d23_genbaseline_longhealth_rag_px",
#         "2025-04-23-10-24-57-m04d23_genbaseline_longhealth_rag_px",
#         "2025-04-23-10-53-58-m04d23_genbaseline_longhealth_rag_px"
#     ],
#     hue="RAG Baseline",
#     modify_x=lambda row, x: convert_x_to_bytes(row, x),
# )

# baseline_runs = PlotRunSetConfig(
#     wandb_run_ids=["hazy-research/capsules/6as2n9bx"],
#     hue="ICL Baseline",
#     baseline=True,
#     modify_x=lambda row, x: convert_x_to_bytes_baseline(row, x),
# )

baseline_runs = PlotRunSetConfig(
    raw_data=[
        {
            "generate_mmtob-ek-test/batch_score": 38.37397030895699,
            "generate_mmtob-ke-test/batch_score": 36.41735615395569,
            "generate_mmtob-ek-test/num_system_and_user_tokens": 47964,
            "generate_mmtob-ke-test/num_system_and_user_tokens": 47964,
            "generator.client.model_name": "meta-llama/Llama-3.1-8B-Instruct",
        },
    ],
    hue="ICL Baseline (Llama-3.1-8B)",
    baseline=True,
    modify_x=lambda row, x: convert_x_to_bytes_baseline(row, x),
)


run_sets = [
    cache_tuning,
    # cache_tuning_old,
    baseline_runs, 
    # bm_25_runs,
    # openai_runs,
]

task = "ke"
task_to_title = {
    "ek": "English -> Kalamang",
    "ke": "Kalamang -> English",
}

avg_plot = QualityVsCacheSizePlotter.Config(
    run_sets=run_sets,
    y_metric="batch_score",
    dataset_type="generate",
    include_datasets=[f"mmtob-{task}-test"],
    show_baseline_points=True,
    y_label="ChRF",
    x_label="Cache Size (GB)",
    hue_label="Method",
    run_id="avg_plot",
    legend_loc="best",
    title=f"MTOB {task_to_title[task]}"
)

if __name__ == "__main__":
    pydrantic.main([
        # dataset_plots, 
        avg_plot
    ])
