
import pydrantic
from monkeys_next_section_pred_trans import trans_config
config = trans_config("your own words, not using the same phrasing or words as the original.")

if __name__ == "__main__":
    # Launch pydrantic CLI, which will parse arguments and run config.run() if desired.
    pydrantic.main([config])
