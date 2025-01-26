'''
    模型演示 Deploy Demo, 最后补充
'''
import numpy as np
from keras.models import load_model
import os
import argparse
import json
import joblib

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Inference tlv model with a config JSON.")
    parser.add_argument("--config_path", default="config/config.json", type=str, help="Path to the configuration JSON file.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    with open(args.config_path) as f:
        config = json.load(f)
    universal_config = config['universal']
    inference_config = config['inference']
    threshold = inference_config['threshold']
    # Load model and scaler
    vae = load_model(os.path.join('model_pth', universal_config['model_name'] + '.h5'))
    scaler_path = os.path.join('data', inference_config["scaler_name"] + '.joblib') # this file usually doesn't need change
    scaler = joblib.load(scaler_path)

    # Random input data
    random_input = np.random.randn(64, 6)
    std_input = scaler.transform(random_input)
    output = vae.predict(np.array([std_input]), batch_size=1)

    # Use normal weight adjustment simply
    error = np.mean(np.square(std_input - output))
    print(f'Error: {error}')

    if error > threshold:
        print(f'Error: {error} > threshold: {threshold}. Anomaly data appears!')
    else:
        print(f'Error: {error} <= threshold: {threshold}. Normal data, it is ok!')