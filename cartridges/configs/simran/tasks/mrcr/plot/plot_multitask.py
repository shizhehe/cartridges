from collections import defaultdict
from dataclasses import dataclass
import os
import uuid
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from typing import Any, Counter, Dict, List, Literal, Tuple

from typing import Callable, List, Optional, Union
from pydrantic import BaseConfig
from tqdm import tqdm

from .base import Plotter, PlotterConfig


class PlotRunSetConfig(BaseConfig):
    # we will take the union of the run_ids, launch_ids, and wandb_run_ids
    run_ids: Optional[Union[str, List[str]]] = None
    launch_ids: Optional[Union[str, List[str]]] = None
    # this is the unique identifier for the run in wandb which is found at the end of
    # of the run path (e.g. hazy-research/capsules/rfdhxjn6) in the overview tab of the
    # run page
    wandb_run_ids: Optional[Union[str, List[str]]] = None

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


@dataclass
class PlotDataRow:
    hue: Optional[str]
    shape: Optional[str]
    x: Any
    y: float
    run_set_config: PlotRunSetConfig
    color: Optional[str] = None


class QualityVsCacheSizePlotter(Plotter):

    class Config(PlotterConfig):
        run_sets: List[PlotRunSetConfig]

        y_metric: str = "perplexity"
        dataset_type: Literal["eval", "generate"] = "eval"
        excluded_datasets: Optional[List[str]] = None

        # If False, then we don't actually plot the scatter point for baseline run sets
        # we only plot the horizontal line. This is useful for when the x value of the
        # the baseline is far to the right, and we want to zoom in on the other runs.
        show_baseline_points: bool = True

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


    def _prepare_data(self) -> List[PlotDataRow]:

        all_data: List[PlotDataRow] = [] # Renamed and moved outside the loop
        

        for rs_config in tqdm(self.config.run_sets, desc="Fetching and processing runs"):
            datasets: Optional[List[str]] = None

            df = self._fetch_runs(
                launch_ids=rs_config.launch_ids, 
                run_ids=rs_config.run_ids, 
                wandb_run_ids=rs_config.wandb_run_ids
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

            # Ensure consistent datasets across all run_sets
            if datasets is None:
                datasets = current_datasets
            elif set(datasets) != set(current_datasets):
                # raise ValueError(f"Dataset mismatch detected. Run sets must {self.config.dataset_type}uate on the same datasets. Found datasets: {sorted(list(set(datasets)))} and {sorted(list(set(current_datasets)))}")
                pass
                
            if 1: #"icl" not in rs_config.hue.lower():

                for run in run_set:
                    if rs_config.filter is not None and not rs_config.filter(run):
                        continue

                    for dataset in datasets:

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

                        x = run[f"{self.config.dataset_type}_{dataset}/num_system_and_user_tokens"]
                        if rs_config.modify_x is not None:
                            x = rs_config.modify_x(run, x)

                        color = "blue"
                        if "task" not in dataset:
                            dataset_str = "Task 1"
                        elif "task2" in dataset:
                            dataset_str = "Task 2"
                        elif "task3" in dataset:
                            dataset_str = "Task 3"


                        row = PlotDataRow(
                            shape=shape,
                            hue=f"{hue} ({dataset_str})",
                            x=x,
                            y=run[f"{self.config.dataset_type}_{dataset}/{self.config.y_metric}"],
                            run_set_config=rs_config,
                            color=color,
                        )
                        all_data.append(row)
        
            else:

                values = []
                for run in run_set:
                    if rs_config.filter is not None and not rs_config.filter(run):
                        continue
                    for metric in [self.config.y_metric]:
                        for dataset in datasets:
                            values.append(run[f"{self.config.dataset_type}_{dataset}/{metric}"])
                            # remove nan
                            values = [v for v in values if v is not None and not pd.isna(v)]
                run[f"{self.config.dataset_type}_avg/{metric}"] = sum(values) / len(values)

                num_tokens = []
                for run in run_set:
                    if rs_config.filter is not None and not rs_config.filter(run):
                        continue
                    for metric in ["num_system_and_user_tokens"]:
                        for dataset in datasets:
                            num_tokens.append(run[f"{self.config.dataset_type}_{dataset}/{metric}"])
                            # remove nan
                            num_tokens = [v for v in num_tokens if v is not None and not pd.isna(v)]
                run[f"{self.config.dataset_type}_avg/{metric}"] = sum(num_tokens) / len(num_tokens)


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
            legend=True,
            marker="o",
            ax=ax,
        )
        path = os.path.join(self.config.run_dir, "seaborn_sanity_check.png")
        plt.savefig(path)
        print(f"Saved seaborn sanity check to {path}")


    def _plot(self, runs: List[PlotDataRow]):
        
        fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size as needed
        ax.grid(True, linestyle='--', alpha=0.5)

        # Group runs by hue
        from collections import defaultdict
        hue_groups: Dict[str, List[PlotDataRow]] = defaultdict(list)
        for pt in runs:
            hue_groups[pt.hue].append(pt)

        # Get default color cycle from matplotlib
        # default_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['b', 'g', 'r', 'c', 'm', 'y', 'k'])

        default_colors = plt.rcParams['axes.prop_cycle'].by_key().get(
            'color', ['b', 'b', 'b', 'b', 'b', 'y', 'k']
        )
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

            # breakpoint()

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
                ax.scatter(xs, ys, s=self.config.marker_size, edgecolors='black', linewidths=1,
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

        # Create legend for hues
        hue_handles = []
        hue_labels = []
        for hue_val, color in hue_color_mapping.items():
            label = str(hue_val) if hue_val is not None else "default"
            handle = Line2D([0], [0], marker='o', color='black', linestyle='-', markersize=8,
                              markerfacecolor=color, markeredgecolor='black')
            hue_handles.append(handle)
            hue_labels.append(label)
        if hue_handles:
            leg1 = ax.legend(hue_handles, hue_labels, title="Hue", loc=self.config.legend_loc)
            ax.add_artist(leg1)

        # Create legend for shapes if more than one unique shape exists
        if len(shape_marker_mapping) > 1:
            shape_handles = []
            shape_labels = []
            for shape_val, marker in shape_marker_mapping.items():
                label = shape_val if shape_val != "default" else "default"
                handle = Line2D([0], [0], marker=marker, color='w', linestyle='None', markerfacecolor='gray',
                                markeredgecolor='black', markersize=8)
                shape_handles.append(handle)
                shape_labels.append(label)
            ax.legend(shape_handles, shape_labels, title="Shape", loc="upper left")
        
        sns.despine(ax=ax)

        return ax
