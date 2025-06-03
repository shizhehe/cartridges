import pydrantic
from capsules.configs.ryan.m04d04.datagen.refdist.refdist import refdist_config


name = "m04d04_amd_sliding_window_memorization"
config = refdist_config(
    input_artifact=f"hazy-research/capsules/{name}:latest",
    output_artifact=f"{name}_refdist",
)
if __name__ == "__main__":
    pydrantic.main([config])
