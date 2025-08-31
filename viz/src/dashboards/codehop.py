from functools import partial
import wandb
import pandas as pd
from dashboards.base import Dashboard, registry, Slice


def slice_by_depth(df: pd.DataFrame, table: str) -> list[Slice]:
    dfs = df.groupby("call_chain_depth")

    return [
        Slice(
            name=f"Call Chain Depth {depth}", 
            df=df,
            metrics={
                f"{table}/score": df["score"].mean(),
            }
        )
        for depth, df in dfs
    ]

def slice_by_file(df: pd.DataFrame, table: str) -> list[Slice]:
    dfs = df.groupby("file_name")
    return [
        Slice(
            name=f"File {file}", 
            df=df,
            metrics={
                f"{table}/score": df["score"].mean(),  
            }
        )
        for file, df in dfs
    ]


def slice_by_file_level(df: pd.DataFrame, table: str) -> list[Slice]:
    dfs = df.groupby("file_level")
    return [
        Slice(
            name=f"File Level {file}", 
            df=df,
            metrics={
                f"{table}/score": df["score"].mean(),  
            }
        )
        for file, df in dfs
    ]


for table, name in [
    ("generate_codehop", "CodeHop"),
    ("generate_codehop_w_ctx", "CodeHop_w_Ctx"),
]:
    registry.register(Dashboard(
        name=f"{name}",
        filters={"$and": [{"tags": "codehop"}, {"tags": "train"}]},
        table=f"{table}/table",
        score_metric=f"{table}/score",
        step="train/optimizer_step",
        slice_fns=[
            partial(slice_by_depth, table=table),
            partial(slice_by_file_level, table=table),
            partial(slice_by_file, table=table),
        ]
    ))
        