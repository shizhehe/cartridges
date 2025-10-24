import os
from typing import List, Literal, Optional
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from pydrantic import ObjectConfig, RunConfig

from cartridges.utils.wandb import fetch_wandb_runs


class PlotterConfig(RunConfig, ObjectConfig):
    _pass_as_config: bool = True

    output_types: List[Literal["png", "pdf"]] = ["pdf", "png"]

    output_dir: str = os.path.join(os.getenv("CARTRIDGES_OUTPUT_DIR"), "cartridges/analysis/figures/outputs")


    def run(self):
        # this is important for keeping the legend from being cut off
        plt.tight_layout()
        plotter = self.instantiate()
        plotter.plot()
        for output_type in self.output_types:
            path = os.path.join(self.run_dir, f"plot.{output_type}")
            plt.savefig(path, bbox_inches="tight")
            print(f"Saved plot to {path}")
    
    


class Plotter(ABC):

    def __init__(self, config: PlotterConfig):
        self.config = config

    @abstractmethod
    def _prepare_data(self):
        raise NotImplementedError("Subclass must implement prepare_data method")

    @abstractmethod
    def plot(self):
        raise NotImplementedError("Subclass must implement plot method")

    def _fetch_runs(
        self,
        launch_ids: Optional[List[str]],
        run_ids: Optional[List[str]],
        wandb_run_ids: Optional[List[str]],
        step: Optional[str] = None,
        return_steps: bool = False,
        step_keys: Optional[List[str]] = None,
    ):
        if launch_ids is None and run_ids is None and wandb_run_ids is None:
            raise ValueError("At least one of launch_ids, run_ids, or wandb_run_ids must be provided")
        launch_ids = launch_ids or []
        run_ids = run_ids or []

        if wandb_run_ids is not None:
            wandb_run_ids = [x.split("/")[-1] for x in wandb_run_ids]
        else:
            wandb_run_ids = []
        
        df, steps = fetch_wandb_runs(
            filters={
                "$or": [
                    {"name": {"$in": wandb_run_ids}},
                    {"config.run_id": {"$in": run_ids}},
                    {"config.launch_id": {"$in": launch_ids}},
                ]
            },
            step=step,
            return_steps=return_steps,
            step_keys=step_keys,
        )
        if return_steps:
            return df, steps
        else:
            return df