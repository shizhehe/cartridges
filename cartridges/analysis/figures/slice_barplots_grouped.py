from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from typing import Any, Counter, Dict, List, Literal


from typing import Callable, List, Optional, Union
from pydrantic import BaseConfig
from tqdm import tqdm

from cartridges.utils.cache_size import MODEL_TO_CACHE_SIZE_FN
from cartridges.utils.wandb import fetch_wandb_runs

from .base import Plotter, PlotterConfig
from ..utils import BACKUP_COLORS, COLORS, HUE_TO_COLOR, HUE_TO_NAME, HUE_TO_ZORDER

class PlotRunSetConfig(BaseConfig):
    # we will take the union of the run_ids, launch_ids, and wandb_run_ids
    run_ids: Optional[Union[str, List[str]]] = None
    launch_ids: Optional[Union[str, List[str]]] = None
    # this is the unique identifier for the run in wandb which is found at the end of
    # of the run path (e.g. hazy-research/cartridges/rfdhxjn6) in the overview tab of the
    # run page
    wandb_run_ids: Optional[Union[str, List[str]]] = None

    # HACK: or we can just pass in the raw data
    raw_data: Optional[List[Dict[str, str | float | int]]] = None

    # the name is what is used for the group of the plot (things in the same group will be averaged together)
    # can either just pass a name or a function that takes the config and returns a name
    group: Optional[Union[str, Callable[[BaseConfig], str]]] = None
    
    # the name is what is used for the hue of the 
    # can either just pass a name or a function that takes the config and returns a name
    hue: Optional[Union[str, Callable[[dict], str]]] = None

    filter: Optional[Callable[[dict], bool]] = None


    step: Optional[int] = None



@dataclass
class PlotDataRow:
    hue: Optional[str]
    group: Optional[str]
    score: float
    dataset: str

    run_set_config: Optional[PlotRunSetConfig] = None


class SliceBarplots(Plotter):

    class Config(PlotterConfig):
        run_sets: List[PlotRunSetConfig]

        metric: str = "perplexity"
        hue_order: Optional[List[str]] = None
        dataset_type: Literal["eval", "generate"] = "eval"

        modify_dataset: Optional[Callable[[str], str]] = None
        dataset_to_label: Optional[Dict[str, str]] = None
        excluded_datasets: Optional[List[str]] = None
        include_datasets: Optional[List[str]] = None
        
        score_label: str = "Perplexity"
        hue_label: Optional[str] = None
        title: Optional[str] = None


    def __init__(self, config: Config):
        self.config = config
        assert not (self.config.include_datasets is not None and self.config.excluded_datasets is not None), "Cannot include and exclude datasets at the same time"


    def _prepare_data(self) -> List[PlotDataRow]:

        all_data: List[PlotDataRow] = [] # Renamed and moved outside the loop
        datasets: Optional[List[str]] = None

        for rs_config in tqdm(self.config.run_sets, desc="Fetching and processing runs"):
            if rs_config.raw_data is not None:
                run_set = rs_config.raw_data
            else:
                df = self._fetch_runs(
                    launch_ids=rs_config.launch_ids, 
                    run_ids=rs_config.run_ids, 
                    wandb_run_ids=rs_config.wandb_run_ids,
                    step=rs_config.step,
                )

                if df is None or df.empty:
                    print(f"Warning: No data found for run set config {rs_config}.")
                    continue

                run_set = df.to_dict(orient="records")

            # determine what datasets were evaluated for this run set
            current_datasets = [
                col.split("/")[0][len(f"{self.config.dataset_type}_"):]
                for col in df.columns
                if col.endswith(f"/{self.config.metric}") and col.startswith(f"{self.config.dataset_type}_")
            ]
            


            if self.config.excluded_datasets is not None:
                current_datasets = [
                    ds for ds in current_datasets if self.config.modify_dataset(ds) not in self.config.excluded_datasets
                ]
            if self.config.include_datasets is not None:
                current_datasets = [
                    ds for ds in current_datasets if self.config.modify_dataset(ds) in self.config.include_datasets
                ]

            # Ensure consistent datasets across all run_sets
            if datasets is None:
                datasets = [self.config.modify_dataset(ds) for ds in current_datasets]
            elif set(datasets) != set([self.config.modify_dataset(ds) for ds in current_datasets]):
                raise ValueError(f"Dataset mismatch detected. Run sets must {self.config.dataset_type}uate on the same datasets. Found datasets: {sorted(list(set(datasets)))} and {sorted(list(set(current_datasets)))}")
                
            for run in run_set:
                if rs_config.filter is not None and not rs_config.filter(run):
                    continue

                for dataset in current_datasets:

                    value = run[f"{self.config.dataset_type}_{dataset}/{self.config.metric}"]
                    # breakpoint()

                    if rs_config.hue is None:
                        hue = lambda row: row["name"]
                    elif isinstance(rs_config.hue, str):
                        hue = rs_config.hue
                    else:
                        hue = rs_config.hue(run)
                    
                    if rs_config.group is None:
                        group = None
                    elif isinstance(rs_config.group, str):
                        group = rs_config.group
                    else:
                        group = rs_config.group(run)

                    row = PlotDataRow(
                        hue=hue,
                        group=group,
                        dataset=self.config.modify_dataset(dataset),
                        score=value,
                        run_set_config=rs_config,
                    )
                    all_data.append(row)
        

        return all_data

    def plot(self):
        """Generates the plot."""
        plot_data = self._prepare_data()

        self._viz_data_table(plot_data)


        if not plot_data:
            print("No data available to plot.")
            return None
        return self._plot(plot_data)

    def _viz_data_table(self, runs: List[PlotDataRow]):
        df = pd.DataFrame([vars(run) for run in runs]).drop(columns=["run_set_config"])

        from tabulate import tabulate
        headers = list(df.columns)
        headers[headers.index('score')] = f"score ({self.config.score_label})"
        headers[headers.index('dataset')] = f"dataset"
        headers[headers.index('hue')] = f"hue ({self.config.hue_label})"
        pretty_table = tabulate(df, headers=headers, tablefmt="psql", showindex=False)
        print(pretty_table)


    def _plot(self, runs: List[PlotDataRow]):
        import numpy as np
        import matplotlib.ticker as mticker

        # Get all unique hues and datasets
        hues = sorted(set(run.hue for run in runs))
        datasets = set(run.dataset for run in runs)
        if self.config.dataset_to_label is not None:
            assert set(datasets) == set(self.config.dataset_to_label.keys()), "All datasets must be in dataset_to_label"
            datasets = list(self.config.dataset_to_label.keys())
        else:
            datasets = sorted(datasets)

        if self.config.hue_order is not None:
            assert set(self.config.hue_order) == set(hues), "Hue order must be a subset of the hues"
            hues = self.config.hue_order

        # Build a mapping: dataset -> {hue -> score}
        from collections import defaultdict
        dataset_to_hue_to_score = defaultdict(dict)
        for run in runs:
            dataset_to_hue_to_score[run.dataset][run.hue] = run.score

        # Compute global min and max score (ignoring NaNs)
        all_scores = []
        for run in runs:
            if run.score is not None and not np.isnan(run.score):
                all_scores.append(run.score)
        if all_scores:
            global_min = min(all_scores)
            global_max = max(all_scores)
        else:
            global_min = 0
            global_max = 1

        # Bar width and positions
        n_datasets = len(datasets)
        n_hues = len(hues)
        bar_height = 0.8 / n_hues  # so bars don't overlap

        fig, ax = plt.subplots(figsize=(2.5, 0.5 * n_datasets + 1))

        y_ticks = np.arange(n_datasets)
        for i, hue in enumerate(hues):
            # For each dataset, get the score for this hue (or np.nan if missing)
            scores = [dataset_to_hue_to_score[ds].get(hue, np.nan) for ds in datasets]
            color = HUE_TO_COLOR.get(hue, f"C{i}")
            # Offset bars for each hue
            y_offsets = y_ticks - 0.4 + i * bar_height + bar_height / 2
            bars = ax.barh(
                y_offsets,
                scores,
                height=bar_height,
                color=color,
                edgecolor='black',
                linewidth=1.2,
                label=HUE_TO_NAME.get(hue, str(hue)),
            )
            # # Optionally, show values on bars
            # for j, (score, y) in enumerate(zip(scores, y_offsets)):
            #     if not np.isnan(score):
            #         ax.text(score, y, f"{score:.2f}", va='center', ha='left', fontsize=9)

        # Set y-ticks and labels to dataset names
        ax.set_yticks(y_ticks)
        if self.config.dataset_to_label is not None:
            ax.set_yticklabels([self.config.dataset_to_label[ds] for ds in datasets])
        else:
            ax.set_yticklabels(datasets)

        ax.set_xlabel(self.config.score_label)
        ax.set_title(self.config.title if self.config.title else "")
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax.set_xlim(global_min, global_max + 0.02)
        sns.despine(ax=ax)

        # Legend for hues
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            # If not auto-generated, create manually
            handles = [
                mpatches.Patch(
                    color=HUE_TO_COLOR.get(hue, f"C{i}"),
                    label=HUE_TO_NAME.get(hue, str(hue))
                )
                for i, hue in enumerate(hues)
            ]
            labels = [HUE_TO_NAME.get(hue, str(hue)) for hue in hues]
        ax.legend(handles, labels, title=self.config.hue_label, bbox_to_anchor=(1.01, 1), loc='upper left')

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(right=0.75)

        return fig
