from collections import defaultdict
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
from pydantic import Field
from tqdm import tqdm

from capsules.analysis.tables.base import Table, TableConfig
from capsules.generate.run import load_run, BaseTask


def get_task_name(config: BaseTask.Config):
    return config.target.name


MODEL_TO_MACRO = {
    "gpt-4o": "\gpt",
    "meta-llama/llama-3.2-3b-instruct": "\llamathree",
    "meta-llama/llama-3.1-8b-instruct": "\llamaeight",
    "meta-llama/llama-3.2-1b-instruct": "\llamaone",
    "meta-llama/llama-3.2": "\llamathreetwo",
    "meta-llama/llama-3.1": "\llamathreeone",
    "qwen/qwen2.5-3b-instruct": "\qwenthree",
    "qwen/qwen2.5-7b-instruct": "\qwenseven",
    "qwen/qwen2.5-1.5b-instruct": "\qwenone",
    "qwen2.5": "\qwen",
    "-": "---"
}
TASK_TO_MACRO = {
    "FinanceBench": r"\finance",
    "LongHealth": "\longhealth",
    "QASPER": "\qasper",
    "Macro Avg.": "Macro Avg."
}

PROTOCOL_TO_MACRO = {
    "Remote Only": "Remote Only",
    "Edge Only": "Local Only",
    "Naive": "\\naive",
    "Minions": "\system",
}


class ColumnSpec(BaseConfig):
    name: str
    latex_name: str
    format: Callable[[Any], str] = lambda x: x  

class TableRunConfig(BaseConfig):
    run_dir: Union[str, List[str]]

    # if True, then the run_dir is recursively searched for runs
    recursive: bool = False

    # the name of the protocol
    protocol: Optional[Union[str, Callable[[BaseConfig], str]]] = None

    # the name of the edge model
    edge_model: Optional[Union[str, Callable[[BaseConfig], str]]] = None

    # remote model
    remote_model: Optional[Union[str, Callable[[BaseConfig], str]]] = "gpt-4o"

    # the name of the task the run was for
    # this is useful for computing macro/micro averages across tasks
    task: Optional[Union[str, Callable[[BaseConfig], str]]] = get_task_name

    # hack to deal with the fact that some runs return edge usage in the place of 
    # remote usage
    flip_edge_and_remote: bool = False


def _recursively_find_run_dirs(base_dir: str):
    run_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if "config.yaml" in files:
            run_dirs.append(root)
    return run_dirs

class TradeoffTable(Table):
    class Config(TableConfig):
        runs: List[TableRunConfig]
        column_specs: List[ColumnSpec]

        index_cols: List[str] = Field(default_factory=lambda: ["protocol", "edge_model", "remote_model"])
        pivot_cols: List[str] = Field(default_factory=lambda: ["task"])

        dividers_on: List[str] = Field(default_factory=lambda: ["protocol"])
        ordering: Dict[str, List[str]] = Field(default_factory=lambda: {"protocol": ["Remote Only", "Edge Only", "Naive", "Minions"]})

        # what to do if we encounter runs that are incomplete (e.g. not all problems are recorded)
        incomplete_runs: Literal["exclude", "include", "error"] = "error"
        include_task_avg: bool = False

        save_data: bool = False


    def __init__(self, config: Config):
        self.config = config
        self.column_specs = {
            column_spec.name: column_spec
            for column_spec in config.column_specs
        }

    def _get_value(
        self, 
        table_run_onfig: TableRunConfig, 
        original_config: BaseConfig,
        key: Literal["task", "group", "hue"]
    ):
        item: Union[str, Callable[[BaseConfig], Any]] = getattr(table_run_onfig, key)
        if isinstance(item, str):
            return item
        elif item is None:
            return None
        elif isinstance(item, Callable):
            return item(original_config)
    
    @lru_cache(maxsize=10)
    def _prepare_data(self) -> List[Dict[str, Any]]:
        
    def _prepare_data(self) -> List[PlotDataRow]:

        all_data: List[PlotDataRow] = [] # Renamed and moved outside the loop
        datasets: Optional[List[str]] = None

        for rs_config in tqdm(self.config.run_sets, desc="Fetching and processing runs"):
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
                raise ValueError(f"Dataset mismatch detected. Run sets must {self.config.dataset_type}uate on the same datasets. Found datasets: {sorted(list(set(datasets)))} and {sorted(list(set(current_datasets)))}")
                
            for run in run_set:
                if rs_config.filter is not None and not rs_config.filter(run):
                    continue

                for metric in [self.config.y_metric, "num_system_and_user_tokens"]:
                    values = [
                        run[f"{self.config.dataset_type}_{dataset}/{metric}"]
                        for dataset in datasets
                    ]
                    run[f"{self.config.dataset_type}_avg/{metric}"] = sum(values) / len(values)

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
        
    
    def _compute_averages(
        self, 
        runs: List[Dict[str, Any]],
        key: Literal["x", "accuracy"]
    ) -> float:
        if self.config.task_avg == "micro":
            return sum(run[key] for run in runs) / len(runs)
        elif self.config.task_avg == "macro":
            task_to_runs = defaultdict(list)
            for run in runs:
                task_to_runs[run['task']].append(run)
            
            task_avgs = [
                sum(run[key] for run in task_runs) / len(task_runs)
                for task, task_runs in task_to_runs.items()
            ]
            return sum(task_avgs) / len(task_avgs)
        else:
            raise ValueError(f"Invalid task_avg value: {self.config.task_avg}")

    def build(self):
        data = self._prepare_data()
        df = pd.DataFrame(data)
        df = df[list(self.column_specs.keys()) + self.config.pivot_cols]

        if self.config.include_task_avg:
            avg_df = (df
                .groupby(["protocol", "edge_model", "remote_model", "task"]).mean()
                .groupby(["protocol", "edge_model", "remote_model"]).mean()
                .reset_index()
            )
            avg_df["task"] = "Macro Avg."
            df = pd.concat([avg_df, df])
    
        grouped_df = df.groupby(self.config.index_cols + self.config.pivot_cols).mean().reset_index()
        pivot_df = grouped_df.pivot(index=self.config.index_cols, columns=self.config.pivot_cols)
        breakpoint()

        if self.config.save_data:
            pivot_df.reset_index().to_csv(os.path.join(self.config.run_dir, "pivot_df.csv"), index=False)
        

        # Flip the order of the MultiIndex on the columns
        # pivot_df = pivot_df.swaplevel(axis=1).sort_index(axis=1)
        return self._to_latex(pivot_df)

    def _to_latex(self, pivot_df):
        sns.despine()
        sns.set_theme(style="whitegrid")

        """
        Construct LaTeX code for a 2-level column MultiIndex and (potentially) multi-level row index DataFrame
        without using `to_latex`. This version also adds headers for the index columns themselves.

        Example usage:
            latex_str = produce_latex_table(pivot_df)
            print(latex_str)
        """
        # Flip the order of the MultiIndex on the columns
        pivot_df = pivot_df.swaplevel(axis=1).sort_index(axis=1)
        if self.config.include_task_avg:
            tasks = [task for task in pivot_df.columns.levels[0] if task != "Macro Avg."]
            pivot_df = pivot_df[["Macro Avg."] + tasks]
        

        import pandas as pd

        # Add ordering code
        for col, order in self.config.ordering.items():
            if col in pivot_df.index.names:
                pivot_df = pivot_df.reindex(order, level=col)
        
        # Sanity check: we assume columns is a 2-level MultiIndex.
        if not hasattr(pivot_df.columns, "levels") or len(pivot_df.columns.levels) != 2:
            raise ValueError("This function expects a DataFrame with a two-level columns MultiIndex.")

        # Gather the column MultiIndex levels.
        top_level = pivot_df.columns.levels[0]  # e.g., tasks/stat groups
        if self.config.include_task_avg:
            top_level = ["Macro Avg."] + [task for task in top_level if task != "Macro Avg."]
        sub_level = pivot_df.columns.levels[1]  # e.g., metrics or sub-task columns

        # In typical usage, pivot_df.columns is something like:
        # MultiIndex([('finance', 'accuracy'), ('finance', 'f1'), ('longhealth', 'accuracy'), ... ],
        #            names=[None, None])
        # We'll obtain every (top, sub) pair in the order they appear.
        col_pairs = pivot_df.columns.tolist()  # e.g. [("finance", "accuracy"), ("finance", "f1"), ...]

        # Sort column pairs to match column spec order for sub-columns
        spec_order = list(self.column_specs.keys())
        def get_sub_col_order(col_pair):
            sub_col = col_pair[1]
            try:
                return spec_order.index(sub_col)
            except ValueError:
                return len(spec_order)  # Put unspecified columns at the end
                
        col_pairs.sort(key=lambda x: (top_level.index(x[0]), get_sub_col_order(x)))

        # Compute how many sub-columns each top-level group has, to align with \multicolumn.
        top_counts = {lvl: 0 for lvl in top_level}
        for (tl, _) in col_pairs:
            top_counts[tl] += 1

        # The row index might also be multi-level. We'll figure out how many levels it has:
        row_index_depth = pivot_df.index.nlevels  
        # Typically row_index_depth = len(["protocol","edge_model","remote_model"]) = 3
        index_headers = [nm if nm else "" for nm in pivot_df.index.names]  # Possibly user-supplied names

        # Number of data columns in the table:
        num_data_cols = len(col_pairs)

        # Build the column alignment string:
        #  - "l" for each row index level (so row_index_depth times)
        #  - "c" for each data column (num_data_cols times)
        alignment = " ".join(["l"] * row_index_depth + ["c"] * num_data_cols)

        lines = []
        lines.append(r"\begin{tabular}{" + alignment + "}")
        lines.append(r"\toprule")

        #
        # 1) Top header row with the row-index-level column names,
        #    plus the grouped top-level column headers for the data columns
        #
        # e.g., if index_headers = ["protocol","edge_model","remote_model"],
        # we create a single cell for each, then \multicolumn for each top-level column group.
        #
        header_top_parts = []
        for nm in self.config.index_cols:
            if nm in self.column_specs:
                latex_name = self.column_specs[nm].latex_name
            else:
                latex_name = nm
            header_top_parts.append(r"\multicolumn{1}{c}{" + latex_name + "}")


        for tl in top_level:
            if tl in TASK_TO_MACRO:
                latex_name = TASK_TO_MACRO[tl]
            else:
                latex_name = tl
            count = top_counts[tl]
            header_top_parts.append(r"\multicolumn{" + str(count) + r"}{c}{" + latex_name + "}")

        lines.append(" & ".join(header_top_parts) + r" \\")
        # Draw a partial horizontal line across the columns for the data portion:
        # lines.append(r"\cline{" + str(row_index_depth + 1) + "-" + str(row_index_depth + num_data_cols) + "}")

        #
        # 2) Header row for the sub-level columns (the second level in the pivoted DataFrame)
        #
        # For the row-index columns (i.e., protocol, edge_model, remote_model), we just leave blank because 
        # they're labeled in the step above. Then each sub-level column label is placed in order.
        #
        header_sub_parts = [""] * row_index_depth
        for (top, sub) in col_pairs:
            if sub in self.column_specs:
                name = self.column_specs[sub].latex_name
            else:
                name = sub
            header_sub_parts.append(name)

        lines.append(" & ".join(header_sub_parts) + r" \\")
        lines.append(r"\hline")

        #
        # 3) Table body
        #
        # For each row in pivot_df, we:
        #   - Pull out the row index level values (e.g., protocol="Naive Protocol", edge_model="Llama2-7b", etc.)
        #   - Then each cell is the data for the corresponding (top, sub) pair in col_pairs.
        #
        curr_divider_values = {key: None for key in self.config.dividers_on}
        for row_idx, row_series in pivot_df.iterrows():
            # row_idx is a tuple if the row index is multi-level
            row_index_values = []
            divide = False
            for col, val in zip(self.config.index_cols, row_idx):
                if col in self.config.dividers_on:
                    if (curr_divider_values[col] is not None) and (curr_divider_values[col] != val):
                        divide = True
                    curr_divider_values[col] = val
                if col in self.column_specs:
                    row_index_values.append(self.column_specs[col].format(val))
                else:
                    row_index_values.append(str(val))
            if divide:
                lines.append(r"\midrule")
            data_values = []
            for colpair in col_pairs:
                val = row_series[colpair]
                col_name = colpair[1]  # Get the sub-level column name
                if col_name in self.column_specs:
                    formatted_val = self.column_specs[col_name].format(val)
                else:
                    formatted_val = val

                if formatted_val is None or (isinstance(formatted_val, float) and pd.isna(formatted_val)):
                    data_values.append("--")
                else:
                    data_values.append(str(formatted_val))

            line_parts = row_index_values + data_values
            lines.append(" & ".join(line_parts) + r"\\")

        lines.append(r"\hline")
        lines.append(r"\end{tabular}")

        return "\n".join(lines)