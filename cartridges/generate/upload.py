
import argparse
import os
from pathlib import Path
import time 

import wandb

from capsules.generate.generate_training import GenerateTrainingConfig
from capsules.utils import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Process a directory path.")
    parser.add_argument('run_dir', type=str, help='Path to the directory')
    args = parser.parse_args()

    run_dir = args.run_dir
    config_path = os.path.join(run_dir, "config.yaml")
    print(f"Loading config from {config_path}")
    config: GenerateTrainingConfig = GenerateTrainingConfig.from_yaml(config_path)

    # we need to find the wandb run id using the local run id we use in the path
    api = wandb.Api()
    runs = api.runs(
        config.wandb.project, 
        filters={
            f"config.run_id": {"$in": [config.run_id]}
        }
    )
    wandb_run_id = runs[0].id

    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        id=wandb_run_id,
        resume="must"
    )

    t0 = time.time()
    output_dir = Path(run_dir) / "artifact"
    artifact = wandb.Artifact(name=config.name, type="dataset")
    artifact.add_dir(local_path=str(output_dir.absolute()), name="dataset")
    wandb.log_artifact(artifact)
    artifact.wait()
    logger.info(
        f"Saved dataset to wandb as artifact {artifact.name}, took {time.time() - t0:.1f}s"
    )


    wandb.finish()


if __name__ == "__main__":
    main()
