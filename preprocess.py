import json
import argparse

from data_helpers.data_preprocessor import preprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', default="./config/data_preprocessor_config.json",
        type=str, help='Data preprocessor config file path.'
    )
    parser.add_argument(
        '-r', '--root_dir', default="/home/l7secu/workspace/project/FOTS-torch/",
        type=str, help='Data preprocessor config file path.'
    )
    parser.add_argument(
        '-m', '--mode', default="multi",
        type=str, help='Data preprocessor config file path.'
    )
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as f:
            config = json.load(f)
        preprocess(args.root_dir,config)
    else:
        print("Invalid data preprocessing configuration file provided.")
