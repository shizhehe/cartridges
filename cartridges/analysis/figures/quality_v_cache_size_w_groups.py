from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import math
import os
import uuid
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from typing import Any, Counter, Dict, List, Literal, Tuple
from pydrantic import ObjectConfig, RunConfig

from typing import Callable, List, Optional, Union
from pydrantic import BaseConfig
from tqdm import tqdm

from capsules.utils.cache_size import MODEL_TO_CACHE_SIZE_FN
from capsules.utils.wandb import fetch_wandb_runs

from .base import Plotter, PlotterConfig
from ..utils import BACKUP_COLORS, COLORS, HUE_TO_COLOR, HUE_TO_GRADIENTS, HUE_TO_NAME, HUE_TO_ZORDER

class PlotRunSetConfig(BaseConfig):
    # we will take the union of the run_ids, launch_ids, and wandb_run_ids
    run_ids: Optional[Union[str, List[str]]] = None
    launch_ids: Optional[Union[str, List[str]]] = None
    # this is the unique identifier for the run in wandb which is found at the end of
    # of the run path (e.g. hazy-research/capsules/rfdhxjn6) in the overview tab of the
    # run page
    wandb_run_ids: Optional[Union[str, List[str]]] = None

    # HACK: or we can just pass in the raw data
    raw_data: Optional[List[Dict[str, str | float | int]]] = None

    # can modify the x value set in the plot config
    modify_x: Optional[Union[Callable[[dict, float], float]]] = None

    # the name is what is used for the group of the plot (things in the same group will be averaged together)
    # can either just pass a name or a function that takes the config and returns a name
    group: Optional[Union[str, Callable[[BaseConfig], str]]] = None
    
    # the name is what is used for the hue of the 
    # can either just pass a name or a function that takes the config and returns a name
    hue: Optional[Union[str, Callable[[dict], str]]] = None

    # the name is what is used for the hue of the 
    # can either just pass a name or a function that takes the config and returns a name
    shade: Optional[Union[float, Callable[[dict], float]]] = None

    # # runs with the same shape will have the same marker
    # # if different runs in the same group have different shapes, then the shape for the
    # # average point will be the most common shape in the group
    # shape: Optional[Union[str, Callable[[dict], str]]] = None 

    filter: Optional[Callable[[dict], bool]] = None

    # if True, then a horizontal line is drawn at the accuracy of the run
    baseline: bool = False

    step: Optional[int] = None



@dataclass
class PlotDataRow:
    hue: Optional[str]
    group: Optional[str]
    x: Any
    y: float

    
    baseline: bool
    run_set_config: Optional[PlotRunSetConfig] = None
    shade: Optional[float] = None



class QualityVsCacheSizePlotter(Plotter):

    class Config(PlotterConfig):
        run_sets: List[PlotRunSetConfig]

        y_metric: str = "perplexity"
        dataset_type: Literal["eval", "generate"] = "eval"
        excluded_datasets: Optional[List[str]] = None
        include_datasets: Optional[List[str]] = None

        # If False, then we don't actually plot the scatter point for baseline run sets
        # we only plot the horizontal line. This is useful for when the x value of the
        # the baseline is far to the right, and we want to zoom in on the other runs.
        show_baseline_points: bool = True

        x_scale: Literal["log", "linear"] = "log"
        x_scale_base: int = 10
        y_min: Optional[float] = None
        y_max: Optional[float] = None

        x_label: Optional[str] = None
        y_label: str = "Perplexity"
        hue_label: Optional[str] = None
        title: Optional[str] = None

        # plot customizations
        marker_size: float = 64
        show_lines: bool = True
        legend_loc: Literal["right", "bottom", "best"] = "right"


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
                if col.endswith(f"/{self.config.y_metric}") and col.startswith(f"{self.config.dataset_type}_")
            ]
            if self.config.excluded_datasets is not None:
                current_datasets = [
                    ds for ds in current_datasets if ds not in self.config.excluded_datasets
                ]
            if self.config.include_datasets is not None:
                current_datasets = [
                    ds for ds in current_datasets if ds in self.config.include_datasets
                ]

            # Ensure consistent datasets across all run_sets
            if datasets is None:
                datasets = current_datasets
            elif set(datasets) != set(current_datasets):
                raise ValueError(f"Dataset mismatch detected. Run sets must {self.config.dataset_type}uate on the same datasets. Found datasets: {sorted(list(set(datasets)))} and {sorted(list(set(current_datasets)))}")
                
            for run in run_set:
                if rs_config.filter is not None and not rs_config.filter(run):
                    continue

                for metric in [self.config.y_metric, "num_system_and_user_tokens"]:
                    values = [
                        run[f"{self.config.dataset_type}_{dataset}/{metric}"]
                        for dataset in datasets if 'memorization' not in dataset
                    ]
                    run[f"{self.config.dataset_type}_avg/{metric}"] = sum(values) / len(values)

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

                if rs_config.shade is None:
                    shade = None
                elif isinstance(rs_config.shade, str):
                    shade = rs_config.shade
                else:
                    shade = rs_config.shade(run)
                x = run[f"{self.config.dataset_type}_avg/num_system_and_user_tokens"]
                if rs_config.modify_x is not None:
                    x = rs_config.modify_x(run, x)

                row = PlotDataRow(
                    hue=hue,
                    group=group,
                    x=x,
                    y=run[f"{self.config.dataset_type}_avg/{self.config.y_metric}"],
                    baseline=rs_config.baseline,
                    shade=shade,
                )
                all_data.append(row)
        print(datasets)

        return all_data

    def plot(self):
        """Generates the plot."""
        plot_data = self._prepare_data()

        self._viz_data_table(plot_data)
        self._seaborn_sanity_check(plot_data)

        if not plot_data:
            print("No data available to plot.")
            return None
        return self._plot(plot_data)

    def _viz_data_table(self, runs: List[PlotDataRow]):
        df = pd.DataFrame([vars(run) for run in runs]).drop(columns=["run_set_config"])

        from tabulate import tabulate
        headers = list(df.columns)
        headers[headers.index('x')] = f"x ({self.config.x_label})"
        headers[headers.index('y')] = f"y ({self.config.y_label})"
        headers[headers.index('hue')] = f"hue ({self.config.hue_label})"
        pretty_table = tabulate(df, headers=headers, tablefmt="psql", showindex=False)
        print(pretty_table)
    
    def _seaborn_sanity_check(self, runs: List[PlotDataRow]):
        # SE (04/02): The main plot function below was mostly coded by o1, and I've found its
        # very good at editing itself and doing cool annotations and shape control, which
        # we don't get with seaborn.  
        # However, we should be careful though that it is plotting correctly, since 
        # those bugs would be hard to catch. So I'm also plotting with seaborn to sanity check.
        # that is outputting something reasonable.
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size as needed
        ax.grid(True, linestyle='--', alpha=0.5)
        df = pd.DataFrame([vars(run) for run in runs]).drop(columns=["run_set_config"])
        sns.relplot(
            x="x",
            y="y",
            hue="hue",
            data=df,
            kind="line",
            legend=False,
            marker="o",
            ax=ax,
        )
        path = os.path.join(self.config.run_dir, "seaborn_sanity_check.png")
        plt.savefig(path)
        print(f"Saved seaborn sanity check to {path}")


    def _plot(self, runs: List[PlotDataRow]):

        import numpy as np
        import matplotlib.ticker as mticker

        fig, ax = plt.subplots(figsize=(5, 3))  # Adjust figure size as needed
        ax.grid(True, linestyle='--', alpha=0.5)

        # Set x-axis to log scale
        if self.config.x_scale == "log":
            ax.set_xscale(self.config.x_scale, base=self.config.x_scale_base)
        else:
            ax.set_xscale(self.config.x_scale)

        # Group runs by hue
        hue_to_runs: Dict[str, List[PlotDataRow]] = defaultdict(list)
        for run in runs:
            hue_to_runs[run.hue].append(run)
        if len(COLORS) < len(hue_to_runs):
            raise ValueError(f"Not enough colors for {len(hue_to_runs)} hues")
        hue_to_color = {hue: color for hue, color in zip(hue_to_runs.keys(), COLORS)}
       

        # Group runs by group
        hue_to_pts: Dict[str, List[PlotDataRow]] = defaultdict(list)
        for hue, hue_runs in hue_to_runs.items():
            color = hue_to_color[hue]

            group_to_runs: Dict[str, List[PlotDataRow]] = defaultdict(list)
            for row in hue_runs:
                if row.group is None:
                    group_to_runs[uuid.uuid4()].append(row)
                else:
                    group_to_runs[row.group].append(row)

            for group, group_runs in group_to_runs.items():
                if len(group_runs) == 1:
                    hue_to_pts[hue].append(group_runs[0])
                    continue
                else:
                    assert all(p.baseline == group_runs[0].baseline for p in group_runs), "All runs in a group must have the same baseline value"
                    hue_to_pts[hue].append(
                        PlotDataRow(
                            hue=hue,
                            group=group,
                            x=sum(p.x for p in group_runs) / len(group_runs),
                            y=sum(p.y for p in group_runs) / len(group_runs),
                            shade=sum(p.shade for p in group_runs) / len(group_runs) if group_runs[0].shade is not None else None,
                            baseline=group_runs[0].baseline,
                        )
                    )
                

        # Get default color cycle from matplotlib
        unique_hues = list(hue_to_pts.keys())
        hue_to_color = {}
        for i, hue in enumerate(unique_hues):
            if hue in HUE_TO_COLOR:
                hue_to_color[hue] = HUE_TO_COLOR[hue]
            else:
                hue_to_color[hue] = BACKUP_COLORS[i % len(BACKUP_COLORS)]

        hue_to_cmap = {}
        for hue, gradient in HUE_TO_GRADIENTS.items():
            shades = [pt.shade for pt in hue_to_runs[hue] if pt.shade is not None]
            if len(shades) > 0:
                min_shade = min(shades)
                max_shade = max(shades)
                cmap = LinearSegmentedColormap.from_list(f"{hue}_gradient", [gradient[0], gradient[1]], N=256)
                hue_to_cmap[hue] = lambda shade, cmap=cmap, min_shade=min_shade, max_shade=max_shade: cmap((shade - min_shade) / (max_shade - min_shade))

        # Plot lines and scatter points for each hue group
        for hue, pts in hue_to_pts.items():
            zorder = HUE_TO_ZORDER.get(hue, 50)

            # if False, then we don't actually plot the scatter point for baseline run sets
            # we only plot
            if not self.config.show_baseline_points:
                pts = [pt for pt in pts if not pt.baseline]
            # sort points by x if possible
            try:
                sorted_pts = sorted(pts, key=lambda p: p.x)
            except Exception:
                sorted_pts = pts

            color = hue_to_color[hue]


            # Plot points grouped by marker type
            xs = [p.x for p in sorted_pts]
            ys = [p.y for p in sorted_pts]
            shades = [p.shade for p in sorted_pts]
            new_xs, new_ys = [], []
            for x, y, shade in zip(xs, ys, shades):
                if self.config.y_min is not None and y < self.config.y_min:
                    y = self.config.y_min
                    marker = "X"
                elif self.config.y_max is not None and y > self.config.y_max:
                    y = self.config.y_max
                    marker = "X"
                else:
                    marker = "o"
                    new_xs.append(x)
                    new_ys.append(y)
                
                curr_color = hue_to_cmap[hue](shade) if shade is not None else color
                print(shade, curr_color)
                ax.scatter(x, y, 
                    # s= [x / 100 for x in xs], 
                    s=self.config.marker_size, 
                    edgecolors='black', linewidths=1,
                    marker=marker, color=curr_color, zorder=zorder
                )
            
            if len(sorted_pts) > 1 and self.config.show_lines:
                ax.plot(new_xs, new_ys, color=color, linestyle='-', linewidth=2, zorder=50)

        # Draw horizontal dashed lines for baseline run sets
        for hue, pts in hue_to_pts.items():
            baseline_pts = [pt for pt in pts if pt.baseline]
            if baseline_pts:
                baseline_y = sum(pt.y for pt in baseline_pts) / len(baseline_pts)
                color = "black"  # hue_to_color[hue]
                ax.axhline(baseline_y, linestyle='--', color=color, linewidth=1.5, zorder=40)

        # Set labels and title
        if self.config.x_label:
            ax.set_xlabel(self.config.x_label)
        else:
            ax.set_xlabel("X")
        ax.set_ylabel(self.config.y_label)
        if self.config.title:
            ax.set_title(self.config.title)

        # Add more xticks and express them not in scientific notation
        # Collect all x values
        if self.config.x_scale == "log":
            all_x = [p.x for p in runs if p.x is not None]
            if all_x:
                min_x = min(all_x)
                max_x = max(all_x)
                # Generate more ticks in log space
                # Find the order of magnitude range
                log_min = np.floor(np.log10(min_x))
                log_max = np.ceil(np.log10(max_x))
                # Generate ticks at each 0.1 log step for more granularity
                ticks = np.logspace(log_min, log_max, num=int((log_max-log_min) * 2)+1)
                # Only keep ticks within the data range
                ticks = [tick for tick in ticks if min_x <= tick <= max_x]
                # Add the min and max if not present
                if min_x not in ticks:
                    ticks = [min_x] + ticks
                if max_x not in ticks:
                    ticks = ticks + [max_x]
                ax.set_xticks(ticks)
                # Format ticks as plain numbers, not scientific notation
                ax.get_xaxis().set_major_formatter(mticker.FuncFormatter(lambda x, _: '{:,.2f}'.format(x)))
                # Optionally, rotate for readability
                plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

        
        import matplotlib.lines as mlines
        legend_handles = [
            mlines.Line2D(
                [], [], 
                color='w', 
                marker='o', 
                markerfacecolor=color, 
                markeredgecolor='black',
                markersize=np.sqrt(self.config.marker_size) if self.config.marker_size > 0 else 8,
                linestyle='None',
                label=HUE_TO_NAME.get(hue, str(hue))
            )
            for hue, color in hue_to_color.items()
        ]
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                title=self.config.hue_label or "Hue",
                bbox_to_anchor=(1.04, 1),
                loc='upper left',
                borderaxespad=0.
            )

        # Adjust layout to prevent legend overlapping with labels/title if necessary
        # plt.tight_layout() might be called later, or adjust manually:
        plt.subplots_adjust(right=0.75) # Example: Adjust right margin to make space for legend; value might need tuning
        sns.despine(ax=ax)

        return ax
