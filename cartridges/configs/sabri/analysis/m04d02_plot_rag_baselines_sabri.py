from capsules.analysis.figures.quality_v_cache_size_per_dataset import QualityVsCacheSizePerDatasetPlotter
from capsules.analysis.figures.quality_v_cache_size import PlotRunSetConfig, QualityVsCacheSizePlotter
import pydrantic


cache_runs = PlotRunSetConfig(
    run_ids=[
        # "46427f86-8121-4841-8568-16748c144e95",
        # "98780fe5-9070-4266-a29c-f1a12d9bbe35",
        # "0a4f0e32-8b93-4670-9a4f-ccee4a3b5e34",
        # "4fe7739d-a916-43df-9d32-71e87f2cd9c5"
        "ab89fb66-13d8-4ce5-967a-4424dc51598c"
    ],
    hue="Cache Tuning (Yesterday)",
    modify_x=lambda row, x: x + row["kv_cache_initializer.max_tokens"],
)

cache_runs_today = PlotRunSetConfig(
    run_ids=[
        "53960c86-c599-46c5-98da-d3289996b4eb",
        "23b79c84-c5dd-43f0-b2ef-21afe4574b1f"
    ],
    hue="Cache Tuning (Today)",
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
    # cache_runs_today,
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