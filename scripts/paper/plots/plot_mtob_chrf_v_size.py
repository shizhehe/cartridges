
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
        f"hazy-research/Cartridges/{id_}"
        for id_ in [
            "646poxbf",  # 128
            "y46db7xl",  # 256
            "1qjpv32m",  # 512
            "vpe6vmjq",  # 1024
            "f3cnm3ki",  # 2048
            "45ihqmce",  # 4096

            
            
            
            
            
            # old no latex 
            # "ezuefyu3",
            # "vy99nipj",
            # "aib5ansu",
            # "0qkkedxn",
            # "zzrkdiju",
            # "zx2q79x2"

        ]


        # "hazy-research/Cartridges/rvgf53wu",  # 512
        # "hazy-research/Cartridges/ijzzbgr4",  # 2048
        # "hazy-research/Cartridges/lsw41yuc",  # 8192

        # "hazy-research/Cartridges/w8uq1kw8",  # 8192


        # "hazy-research/Cartridges/msad7zj8", # 2048
        # "hazy-research/Cartridges/hltr6yg3", # 4096
        # "hazy-research/Cartridges/c5tyjmep", # 8192
    ],
    hue="cartridge",
    modify_x=lambda row, x: convert_x_to_bytes(row, x + row["num_trainable_tokens"]),
    step=2047,
)



medium_baseline_runs = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/Cartridges/ryq2d61k",
    ],
    hue="icl",
    baseline=True,
    modify_x=lambda row, x: convert_x_to_bytes_baseline(row, x),
)


full_baseline_runs = PlotRunSetConfig(
    wandb_run_ids=[
        "hazy-research/Cartridges/9g38scf8",
    ],
    hue="ICL (Full Book)",
    baseline=True,
    modify_x=lambda row, x: convert_x_to_bytes_baseline(row, x),
)




task = "ke"
task_to_title = {
    "ek": "English -> Kalamang",
    "ke": "Kalamang -> English",
}



kv_cache_compression_duo_on_the_fly = PlotRunSetConfig(
    launch_ids=['2025-05-11-13-18-13-llama_8b_ke_duo'],
    hue="duo",
    modify_x=lambda row, x: row[f"generate_mmtob-{task}-test/kv_cache_size_bytes"] / 1e9,
)

prompt_compression_firstk = PlotRunSetConfig(
    launch_ids=["2025-05-12-14-23-21-genbaseline_mtob_firstk"],
    hue="first_k_tokens",
    modify_x=lambda row, x: convert_x_to_bytes_baseline(row, x),
)

prompt_compression_summary = PlotRunSetConfig(
    launch_ids=["2025-05-12-16-29-32-genbaseline_mtob_summary"],
    hue="summary",
    modify_x=lambda row, x: convert_x_to_bytes_baseline(row, x),
)


# kv_cache_compression_expected = PlotRunSetConfig(
#     launch_ids=['2025-05-10-16-09-07-llama_8b_ke_expected'],
#     hue="KV Cache Compression (Expected Attention)",
#     modify_x=lambda row, x: row[f"generate_mmtob-{task}-test/kv_cache_size_bytes"] / 1e9,
# )

run_sets = [
    cache_tuning,
    # cache_tuning_old,
    medium_baseline_runs,
    full_baseline_runs,
    # bm_25_runs,
    # openai_runs,
    # kv_cache_compression_duo,
    # kv_cache_compression_expected,
    kv_cache_compression_duo_on_the_fly,
    prompt_compression_firstk,
    prompt_compression_summary,
]



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
    title=f"MTOB {task_to_title[task]}",
)

if __name__ == "__main__":
    pydrantic.main(
        [
            # dataset_plots,
            avg_plot
        ]
    )
