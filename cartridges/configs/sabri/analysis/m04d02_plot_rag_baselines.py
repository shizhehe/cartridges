from capsules.analysis.figures.quality_v_cache_size_per_dataset import QualityVsCacheSizePerDatasetPlotter
from capsules.analysis.figures.quality_v_cache_size import PlotRunSetConfig, QualityVsCacheSizePlotter
import pydrantic


cache_runs = PlotRunSetConfig(
    launch_ids=[
        "2025-04-02-00-06-03-m04d01_train_amd_cache_new_data_ref_dist",
        "2025-04-01-23-49-06-m04d01_train_amd_cache_new_data_ref_dist",
        "2025-04-02-14-04-12-m04d01_train_amd_cache_new_data_ref_dist",
        "2025-04-02-14-34-11-m04d01_train_amd_cache_new_data_ref_dist"
    ],
    hue="Cache Tuning",
    modify_x=lambda row, x: x + row["kv_cache_initializer.max_tokens"],
)

baseline_runs = PlotRunSetConfig(
    launch_ids=["2025-04-01-12-56-11-m03d30_amd_anthropic_icl_baseline"],
    hue="ICL Baseline",
    baseline=True
)

rag_runs = PlotRunSetConfig(
    launch_ids=["2025-04-01-11-34-11-m03d31_eval_rag_baseline_amd"],
    hue=lambda row: "RAG (chunk size: " + str(row["eval_datasets.1.dataset.convo_transforms.0.max_tokens_per_chunk"]) + ")",
    filter=lambda row: row["eval_datasets.1.dataset.convo_transforms.0.max_tokens_per_chunk"] > 128
)

run_sets = [
    cache_runs, 
    baseline_runs, 
    rag_runs
]

dataset_plots = QualityVsCacheSizePerDatasetPlotter.Config(
    run_sets=run_sets,
    y_metric="perplexity",
    excluded_datasets=["eval__anthropic_qa_AMD_2022_10K_tag_memorization", "mmlu", "eval__anthropic_qa_AMD_2022_10K_tag_coding", "finance-ppl-gt"],
    show_baseline_points=True,
    
    y_label="Perplexity",
    x_label="Cache Size (tokens)",
    hue_label="Method",
    run_id="dataset_plots"
)

avg_plot = QualityVsCacheSizePlotter.Config(
    run_sets=run_sets,
    y_metric="perplexity",
    excluded_datasets=["eval__anthropic_qa_AMD_2022_10K_tag_memorization", "mmlu", "eval__anthropic_qa_AMD_2022_10K_tag_coding", "finance-ppl-gt"],
    show_baseline_points=True,
    
    y_label="Perplexity",
    x_label="Cache Size (tokens)",
    hue_label="Method",
    run_id="avg_plot"
)

if __name__ == "__main__":
    pydrantic.main(
        [
            dataset_plots,
            avg_plot
        ]
    )
