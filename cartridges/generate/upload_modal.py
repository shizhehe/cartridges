
import argparse
import os
from pathlib import Path
import time 

import modal

from capsules.generate.generate_training import GenerateTrainingConfig
from capsules.utils import get_logger

logger = get_logger(__name__)

vol = modal.Volume.from_name("capsules-datasets")

# DATA_SOURCES = [
#     ("/data/sabri/code/capsules/output/2025-04-22-15-10-19-m04d22_generate_longhealth_generated_reformat_w_toc_npatients/58acef64-5991-4174-8f6c-25de7a817596/artifact/dataset.pkl", None),
#     ("/data/sabri/code/capsules/output/2025-04-22-19-28-13-m04d22_generate_longhealth_generated_reformat_w_toc_npatients/75cec0ba-ab2b-4542-a114-99fb679b44eb/artifact/dataset.pkl", None),
#     ("/data/sabri/code/capsules/output/2025-04-22-20-40-40-m04d22_generate_longhealth_generated_reformat_w_toc_npatients/cc2c97e1-eb6a-467a-b86b-b541c148fed0/artifact/dataset.pkl", None),
#     ("/data/sabri/code/capsules/output/2025-04-22-21-32-05-m04d22_generate_longhealth_generated_reformat_w_toc_npatients/559b979b-c2b4-44dd-98b8-6be995510561/artifact/dataset.pkl", None),
# ]

DATA_SOURCES = [
    # generic prompts
    # ("/data/sabri/code/capsules/output/2025-05-01-13-19-08-m05d01_generate_longhealth_auto/38dd60df-e262-4fde-a2e2-197756461a71/artifact/dataset.pkl", None),
    # ("/data/sabri/code/capsules/output/2025-05-01-13-58-59-m05d01_generate_longhealth_auto/71dbbcf6-9788-4fa0-9fea-ae35ad947663/artifact/dataset.pkl", None),

    # 4096 max context length, long_health prompts 
    ("/data/sabri/code/capsules/output/2025-05-03-12-43-00-m05d03_generate_longhealth_auto_healthprompt/b128659b-4b18-48bf-8b98-5a3f2615bcca/artifact/dataset.pkl", None),
    ("/data/sabri/code/capsules/output/2025-05-03-13-00-10-m05d03_generate_longhealth_auto_healthprompt/bfbf8857-f4ba-41f9-81da-7e5ea8d300c6/artifact/dataset.pkl", None),
    ("/data/sabri/code/capsules/output/2025-05-03-13-16-45-m05d03_generate_longhealth_auto_healthprompt/8d067664-f598-4d91-94c3-6c9af5e14ee9/artifact/dataset.pkl", None),
    ("/data/sabri/code/capsules/output/2025-05-03-14-52-54-m05d03_generate_longhealth_auto_healthprompt/2fce1115-854c-4cb6-a5f7-411c75bbe58e/artifact/dataset.pkl", None),


    # 16384 max context length, long_health prompts 
    ("/data/sabri/code/capsules/output/2025-05-03-15-31-59-m05d03_generate_longhealth_auto_healthprompt/8634d8f0-e8be-4565-99d1-5a8744e3444b/artifact/dataset.pkl", None),
    ("/data/sabri/code/capsules/output/2025-05-03-16-18-33-m05d03_generate_longhealth_auto_healthprompt/d49d9081-51f6-4408-a486-2a8b2e443a36/artifact/dataset.pkl", None),
]

def main():
    parser = argparse.ArgumentParser(description="Process a directory path.")
    parser.add_argument(
        '--path', 
        type=str, 
        help='Path to the directory or a path to train config', 
        default="",
        required=False,
    )
    args = parser.parse_args()

    if args.path != "":
        run_dirs = [args.path]
    else:
        run_dirs = [
            path.replace("artifact/dataset.pkl", "")
            for path, _ in DATA_SOURCES
        ]


    volume = modal.Volume.from_name("capsules-datasets")
    
   
    logger.info(f"Uploading {run_dirs} to modal")
    t0 = time.time()
    with volume.batch_upload() as batch:
        for run_dir in run_dirs:
            batch.put_directory(run_dir, run_dir)
    logger.info(f"Uploaded {run_dirs} to modal in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
