# import os
# from pathlib import Path
# import pydrantic
# from pydrantic.variables import FormatStringVariable

# from capsules.clients.together import TogetherClient
# from capsules.make_training_dataset_for_document import GenerateDatasetConfig


# file_name = Path(__file__).stem
# config = GenerateDatasetConfig(
#     question_client=TogetherClient.Config(
#         model_name="meta-llama/Llama-3.2-3B-Instruct",
#     ),
#     answer_client=TogetherClient.Config(
#         model_name="meta-llama/Llama-3.2-3B-Instruct",
#     ),
#     document_path_or_url="https://gist.githubusercontent.com/MattIPv4/045239bc27b16b2bcf7a3a9a4648c08a/raw/2411e31293a35f3e565f61e7490a806d4720ea7e/bee%2520movie%2520script"
# )

# if __name__ == "__main__":
#     # Launch pydrantic CLI, which will parse arguments and run config.run() if desired.
#     pydrantic.main([config])
