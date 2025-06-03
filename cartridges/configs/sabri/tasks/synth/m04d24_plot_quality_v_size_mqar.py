from capsules.analysis.figures.synth_quality_v_size import (
    QualityVsCacheSizePlotter,
    PlotRunSetConfig,
)
import pydrantic

pretrained_runs = PlotRunSetConfig(
    launch_ids=[
        "2025-04-24-15-22-51-m04d24_mqar"
    ],
    hue=lambda row: row["model.name"],
    modify_x=lambda row, x: x
)


run_sets = [
    pretrained_runs,
]

avg_plot = QualityVsCacheSizePlotter.Config(
    run_sets=run_sets,
    y_metric="valid/accuracy",

    show_baseline_points=True,
    y_label="Accuracy",
    x_label="State Size (bytes)",
    hue_label="Method",
    run_id="avg_plot",
    legend_loc="best",
    show_lines=False,
)

if __name__ == "__main__":
    pydrantic.main([
        # dataset_plots, 
        avg_plot
    ])
