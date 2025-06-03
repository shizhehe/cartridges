from cartridges.analysis.figures.quality_v_compute import (
    PlotRunSetConfig,
    QualityVsComputePlotter,
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
        f"hazy-research/Cartridges/{id_}"
        for id_ in [
            "646poxbf",  # 128
            "y46db7xl",  # 256
            # "1qjpv32m",  # 512
            "vpe6vmjq",  # 1024
            # "f3cnm3ki",  # 2048
            "45ihqmce",  # 4096
        ]
    ],
    hue=lambda row: row['num_trainable_tokens'],
)

icl_baseline = PlotRunSetConfig(
    wandb_run_ids=["hazy-research/Cartridges/ryq2d61k"],
    hue="icl",
    baseline=True,
    modify_x=lambda row, x: convert_x_to_bytes_baseline(row, x),
)

prompt_compression_summary = PlotRunSetConfig(
    launch_ids=["2025-05-12-16-29-32-genbaseline_mtob_summary"],
    hue="summary",
    baseline=True,
)

run_sets = [
    cache_tuning,
    icl_baseline, 
    prompt_compression_summary
]


task = "ke"
task_to_title = {
    "ek": "English -> Kalamang",
    "ke": "Kalamang -> English",
}


avg_plot = QualityVsComputePlotter.Config(
    run_sets=run_sets,
    y_metric="batch_score",
    dataset_type="generate",
    include_datasets=[f"mmtob-{task}-test"],
    y_label="ChRF",
    x_label="Train Steps",
    hue_label="Method",
    bin_size=511,
    run_id="avg_plot",
    legend_loc="best",
)

if __name__ == "__main__":
    pydrantic.main([
        # dataset_plots, 
        avg_plot
    ])
