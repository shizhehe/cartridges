import asyncio
import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.clients.base import CartridgeConfig
from cartridges.data.chunkers import TokenChunker
from cartridges.data.code.resources import PythonRepositoryResource 
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.utils.wandb import WandBConfig


# DATASET_DIR = "/data/sabri/cartridges/2025-08-20-10-12-26-make_codehop/codehop-nf4-nm3-dc2-v4-fn36-0/repo-e2c17c"
DATASET_DIR = "/data/sabri/cartridges/2025-08-20-10-26-45-make_codehop/codehop-nf14-nm1-dc2-v4-fn36-0/repo-df0b47"


config =PythonRepositoryResource.Config(
    path=DATASET_DIR,
    included_extensions=[".py"],
    recursive=True,
)

resource = config.instantiate()

async def main():
    await resource.setup()
    breakpoint()

if __name__ == "__main__":
    asyncio.run(main())
   

