
import pydrantic
from monkeys_next_section_pred_trans import trans_config
config = trans_config("simple English, so that it is easy for a non-expert to understand. Include explanations of concepts if necessary")

if __name__ == "__main__":
    # Launch pydrantic CLI, which will parse arguments and run config.run() if desired.
    pydrantic.main([config])
