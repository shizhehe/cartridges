from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import math
import os
import uuid
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
    
    # the name is what is used for the hue of the 
    # can either just pass a name or a function that takes the config and returns a name
    hue: Optional[Union[str, Callable[[dict], str]]] = None

    # runs with the same shape will have the same marker
    # if different runs in the same group have different shapes, then the shape for the
    # average point will be the most common shape in the group
    shape: Optional[Union[str, Callable[[dict], str]]] = None 

    filter: Optional[Callable[[dict], bool]] = None

    # if True, then a horizontal line is drawn at the accuracy of the run
    baseline: bool = False

    step: Optional[int] = None



@dataclass
class PlotDataRow:
    hue: Optional[str]
    shape: Optional[str]
    x: Any
    y: float
    run_set_config: PlotRunSetConfig


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
                breakpoint()
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
                # breakpoint()

                if rs_config.hue is None:
                    hue = lambda row: row["name"]
                elif isinstance(rs_config.hue, str):
                    hue = rs_config.hue
                else:
                    hue = rs_config.hue(run)

                if rs_config.shape is None:
                    shape = None
                elif isinstance(rs_config.shape, str):
                    shape = rs_config.shape
                else:
                    shape = rs_config.shape(run)

                x = run[f"{self.config.dataset_type}_avg/num_system_and_user_tokens"]
                if rs_config.modify_x is not None:
                    x = rs_config.modify_x(run, x)

                row = PlotDataRow(
                    shape=shape,
                    hue=hue,
                    x=x,
                    y=run[f"{self.config.dataset_type}_avg/{self.config.y_metric}"],
                    run_set_config=rs_config,
                )
                all_data.append(row)
        

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
        from collections import defaultdict
        hue_groups: Dict[str, List[PlotDataRow]] = defaultdict(list)
        for pt in runs:
            hue_groups[pt.hue].append(pt)

        # Get default color cycle from matplotlib
        default_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['b', 'g', 'r', 'c', 'm', 'y', 'k'])
        unique_hues = list(hue_groups.keys())
        hue_color_mapping = {}
        for i, hue_val in enumerate(unique_hues):
            hue_color_mapping[hue_val] = default_colors[i % len(default_colors)]

        # Create marker mapping for shapes
        unique_shapes = set()
        for pt in runs:
            if pt.shape is None:
                unique_shapes.add("default")
            else:
                unique_shapes.add(pt.shape)
        shape_marker_mapping = {}
        for s in unique_shapes:
            if s == "default":
                shape_marker_mapping[s] = "o"
            else:
                shape_marker_mapping[s] = s  # assume valid matplotlib marker

        # Plot lines and scatter points for each hue group
        for hue_val, pts in hue_groups.items():

            # if False, then we don't actually plot the scatter point for baseline run sets
            # we only plot
            if not self.config.show_baseline_points:
                pts = [pt for pt in pts if not pt.run_set_config.baseline]
            # sort points by x if possible
            try:
                sorted_pts = sorted(pts, key=lambda p: p.x)
            except Exception:
                sorted_pts = pts

            x_vals = [p.x for p in sorted_pts]
            y_vals = [p.y for p in sorted_pts]
            color = hue_color_mapping[hue_val]

            if len(sorted_pts) > 1 and self.config.show_lines:
                ax.plot(x_vals, y_vals, color=color, linestyle='-', linewidth=2, zorder=50)

            # Plot points grouped by marker type
            marker_groups = defaultdict(list)
            for p in sorted_pts:
                marker = shape_marker_mapping[p.shape] if p.shape is not None else shape_marker_mapping["default"]
                marker_groups[marker].append(p)
            for marker, pts_marker in marker_groups.items():
                xs = [p.x for p in pts_marker]
                ys = [p.y for p in pts_marker]
                ax.scatter(xs, ys, 
                    # s= [x / 100 for x in xs], 
                    s=self.config.marker_size, 
                    edgecolors='black', linewidths=1,
                           marker=marker, color=color, zorder=100)

        # Draw horizontal dashed lines for baseline run sets
        for hue_val, pts in hue_groups.items():
            baseline_pts = [pt for pt in pts if pt.run_set_config.baseline]
            if baseline_pts:
                baseline_y = sum(pt.y for pt in baseline_pts) / len(baseline_pts)
                ax.axhline(baseline_y, linestyle='--', color=hue_color_mapping[hue_val], linewidth=2, zorder=40)

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

        # Create legend handles based on hue and potentially shape
        legend_handles = []
        # Use the hue groups already computed
        sorted_hues = sorted(hue_groups.keys())

        for hue_val in sorted_hues:
            pts = hue_groups[hue_val]
            if not pts:
                continue

            # Determine the marker shape for this hue group
            # Use the most common shape within the group for the legend marker
            shapes_in_group = [pt.shape for pt in pts if pt.shape]
            if shapes_in_group:
                # Find the most frequent shape in this group
                shape_counts = Counter(shapes_in_group)
                most_common_shape_val = shape_counts.most_common(1)[0][0]
                marker = shape_mapping.get(most_common_shape_val, 'o') # Default to 'o' if shape not in mapping
            else:
                # Default marker if no shape is specified for any point in the group
                marker = 'o' # Use the default marker used for averages without shape

            color = hue_color_mapping[hue_val]

            # Determine marker size for the legend entry
            # Use sqrt of scatter size 's' for Line2D 'markersize'
            marker_size_for_legend = 8 # Default markersize
            if self.config.marker_size is not None and self.config.marker_size > 0:
                 # Ensure marker_size is not negative before sqrt
                 marker_size_for_legend = np.sqrt(self.config.marker_size)


            # Create a legend entry using Line2D
            handle = Line2D([0], [0], marker=marker, color='w', # Use 'w' or 'none' for line color if only marker is desired
                            label=str(hue_val), # Ensure label is string
                            markersize=marker_size_for_legend,
                            markerfacecolor=color,
                            markeredgecolor='black', # Match scatter plot style
                            linewidth=1, # Match scatter plot style edge width
                            linestyle='None') # No line connecting points in legend
            legend_handles.append(handle)

        # Add the legend outside the plot area if there are handles
        if legend_handles:
            # Place the legend outside the plot to the right
            # Adjust bbox_to_anchor and loc as needed for desired placement
            # loc='upper left' places the upper left corner of the legend at bbox_to_anchor
            ax.legend(handles=legend_handles,
                      title=self.config.hue_label or "Hue", # Use hue_label from config if available
                      bbox_to_anchor=(1.04, 1), # Position legend outside the plot (x=1.04 means 4% right of the axes width)
                      loc='upper left', # Anchor point of the legend box
                      borderaxespad=0.) # Padding between axes and legend border

        # Adjust layout to prevent legend overlapping with labels/title if necessary
        # plt.tight_layout() might be called later, or adjust manually:
        plt.subplots_adjust(right=0.75) # Example: Adjust right margin to make space for legend; value might need tuning
        sns.despine(ax=ax)

        return ax
