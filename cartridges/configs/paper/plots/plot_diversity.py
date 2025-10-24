import pydrantic

from capsules.analysis.figures.slice_barplots import SliceBarplots, PlotRunSetConfig


ntp = PlotRunSetConfig(
    wandb_run_ids=["hazy-research/capsules/rnw4tecy"],
    hue="ntp"
)

cartridge = PlotRunSetConfig(
    wandb_run_ids=["hazy-research/capsules/rqq82dys"],
    hue="cartridge"
)

icl = PlotRunSetConfig(
    wandb_run_ids=["hazy-research/capsules/69zpdjh7"],
    hue="icl"
)

DATASET_TO_LABEL = {
    "factual": "Factual Questions",
    "disjoint": "Questions require disjoint info.",
    "reasoning": "Tasks with math. reasoning",
    "creative": "Creative tasks",
    "synthesize": "Tasks requiring synthesis",
    "structure": "Data-structuring tasks",
    "finance-memorization": "Next-Token Prediction",
}


def modify_dataset(dataset: str) -> str:
    if dataset.startswith("eval"):
        return dataset.split("tag_")[-1]
    return dataset

plot = SliceBarplots.Config(
    run_sets=[ntp, cartridge, icl],
    metric="loss",
    hue_order=["ntp", "cartridge", "icl"],
    modify_dataset=modify_dataset,
    score_label="←  log(perplexity)",
    excluded_datasets=["finance-ppl-gt", "memorization", "counting", "coding"],
    dataset_to_label=DATASET_TO_LABEL,
)

if __name__ == "__main__":
    pydrantic.main([plot])


# selected_slices = [
#     "factual", 
#     "synthesize", 
#     "structure",
#     # "memorization", 
#     "reasoning", "creative", 
#     # "disjoint",
#     "eval finance-memorization", 
#     # "eval finance-ppl-gt"
# ]

# fontsize = 20
# sns.set_theme(style="whitegrid")
# plt.rcParams.update({"font.size": fontsize})

# colors = ["#7CB9BC", "#8D69B8", "#DE836C", "#68AC5A", "#E59852"]
# expt_names = list(experiments.keys())
# expt2color = {expt_name: colors[i] for i, expt_name in enumerate(expt_names)}


# def plot_with_optional_axis_break(
#     metric: str,
#     tags: List[str],
#     expt_names: List[str],
#     all_scores: dict,
#     expt2color: dict,
#     output_path: str,
#     break_threshold: float,
#     upper_ylim: tuple,
#     lower_ylim: tuple,
#     title: str,
#     ylabel: str,
#     scale: str = "raw"  # or "k" for thousands
# ):
#     x = np.arange(len(tags))
#     width = 0.90 / len(expt_names)

#     # Check if any values exceed the threshold
#     show_break = any(
#         all_scores[expt].get(tag, 0.0) > break_threshold
#         for expt in expt_names for tag in tags
#     )

#     if show_break:
#         fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(18, 6),
#                                                 gridspec_kw={'height_ratios': [1, 3]})
#         ax_top.grid(True, axis='y')
#         ax_bottom.grid(True, axis='y')
#     else:
#         plt.figure(figsize=(10, 6))
#         ax_bottom = plt.gca()
#         ax_bottom.grid(True, axis='y')

#     for i, expt_name in enumerate(expt_names):
#         offsets = x + (i - len(expt_names)/2) * width + width/2
#         values = [all_scores[expt_name].get(tag, 0.0) for tag in tags]
#         color = expt2color[expt_name]

#         # Plot bars
#         if show_break:
#             ax_bottom.bar(offsets, values, width, label=expt_name, alpha=0.7, color=color)
#             ax_top.bar(offsets, values, width, alpha=0.7, color=color)
#         else:
#             ax_bottom.bar(offsets, values, width, label=expt_name, alpha=0.7, color=color)

#         for j, v in enumerate(values):
#             label = f"{v:.1f}" if scale == "raw" else f"{v/1000:.1f}K"
#             if show_break and v > break_threshold:
#                 ax_top.text(offsets[j], v + (upper_ylim[1] - upper_ylim[0]) * 0.05, label,
#                             ha='center', va='bottom', fontsize=fontsize)
#             else:
#                 ax_bottom.text(offsets[j], v + (lower_ylim[1] - lower_ylim[0]) * 0.02, label,
#                                ha='center', va='bottom', fontsize=fontsize)

#     if show_break:
#         ax_top.set_ylim(upper_ylim)
#         ax_bottom.set_ylim(lower_ylim)

#         ax_top.spines['bottom'].set_visible(False)
#         ax_bottom.spines['top'].set_visible(False)
#         ax_top.tick_params(labeltop=False)
#         ax_bottom.xaxis.tick_bottom()

#         # Diagonal lines for axis break
#         d = .01
#         kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
#         ax_top.plot((-d, +d), (-d, +d), **kwargs)
#         ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

#         kwargs.update(transform=ax_bottom.transAxes)
#         ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
#         ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

#         ax_top.set_title(title, fontsize=fontsize)
#     else:
#         ax_bottom.set_title(title, fontsize=fontsize)

#     if "perplexity" in metric:
#         ax_bottom.set_xticks(x)

#     tags = [tag.replace("Eval ", "").replace("finance-", "").replace("_", " ").replace("memorization", "memorize").capitalize() for tag in tags]
#     ax_bottom.set_xticklabels(tags, rotation=0, ha="right", fontsize=fontsize)
#     ax_bottom.set_ylabel(ylabel, fontsize=fontsize)

#     # y-axis font
#     ax_bottom.tick_params(axis='y', labelsize=fontsize)
#     ax_top.tick_params(axis='y', labelsize=fontsize)

#     if "perplexity" not in metric:
#         bottom_pos = 0.17
#     else:
#         bottom_pos = 0.17
#     if show_break:
#         plt.subplots_adjust(hspace=0.15, bottom=bottom_pos)
#     else:
#         plt.subplots_adjust(bottom=bottom_pos)

#     # Add legend *after* layout
#     fig.legend(
#         loc="lower center",
#         bbox_to_anchor=(0.5, -0.0),
#         ncol=len(expt_names),
#         fontsize=fontsize,
#         frameon=False
#     )
#     plt.savefig(output_path, bbox_inches="tight")
#     plt.show()


# for metric in ['perplexity', 'num_trainable_tokens']:
#     all_scores = {}
#     all_tags = set()

#     for expt_name, wandb_run_path in experiments.items():
#         scores = fetch_runs(wandb_run_ids=[wandb_run_path])
#         metric_cols = [col for col in scores.columns if metric in col]
#         col2score = {}
#         for col in metric_cols:
#             col_name = col.split("/")[0].split("tag")[-1].strip("_").replace("_", " ")
#             if 'train' in col_name.split() or (col_name.lower() not in selected_slices and metric == 'perplexity'):
#                 continue
#             col_name = col_name[0].upper() + col_name[1:]
#             score = scores[col].iloc[-1]
#             col2score[col_name] = score
#             all_tags.add(col_name)
#         all_scores[expt_name] = col2score

#     tags = sorted(all_tags)
#     preferred_order = ["Factual", "Reasoning", "Creative", "Memorize", "Eval_finance-memorization"]
#     tags = [t for t in preferred_order if t in all_tags] + [t for t in all_tags if t not in preferred_order]
#     print(tags)
#     expt_names = list(experiments.keys())
#     output_path = f"capsules/configs/simran/tasks/amd/analyze/{metric}_scores.png"

#     if metric == 'perplexity':
#         plot_with_optional_axis_break(
#             metric,
#             tags,
#             expt_names,
#             all_scores,
#             expt2color,
#             output_path,
#             break_threshold=25,
#             upper_ylim=(50, 70),
#             lower_ylim=(0, 25),
#             title="Finance Evaluations",
#             ylabel="Perplexity",
#             scale="raw"
#         )
#     else:  # num_trainable_tokens
#         plot_with_optional_axis_break(
#             metric,
#             tags,
#             expt_names,
#             all_scores,
#             expt2color,
#             output_path,
#             break_threshold=10000,
#             upper_ylim=(90000, 110000),
#             lower_ylim=(0, 10000),
#             title="Cache Size",
#             ylabel="Number of Tokens",
#             scale="k"
#         )


        


