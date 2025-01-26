import argparse
import json
import sys
from utils import train, test
import tensorflow as tf
import os

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run script for train or test with a config JSON.")
    parser.add_argument("--config_path", default="config/config.json", type=str, help="Path to the configuration JSON file.")
    parser.add_argument("--mode", default="train", type=str, choices=["train", "test"], help="Mode to run: 'train' or 'test'.")
    return parser.parse_args()


def main():
    physical_devices = tf.config.list_physical_devices('GPU')
    print("GPU is available: ", len(physical_devices) > 0)
    print(tf.__version__)  # 检查 TensorFlow 版本
    print(tf.test.is_built_with_cuda())  # 检查是否支持 CUDA
    print(tf.test.is_built_with_gpu_support())  # 检查是否支持 GPU

    args = parse_args()

    try:
        with open(args.config_path, "r") as f:
            config = json.load(f)
        universal_config, train_config, test_config = config["universal"], config["train"], config["test"]
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON file '{args.config_path}': {e}")
        sys.exit(1)

    if not os.path.exists(os.path.join("log", universal_config["exp_name"])):
        os.makedirs(os.path.join("log", universal_config["exp_name"]))

    # Call the appropriate function based on the mode
    if args.mode == "train":
        train(universal_config, train_config)
    elif args.mode == "test":
        test(universal_config, test_config)
    else:
        print(f"Error: Invalid mode '{args.mode}'. Use 'train' or 'test'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
