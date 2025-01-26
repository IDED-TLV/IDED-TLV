import numpy as np
from dataset.dataset import MyDataset
from models.transformer.TransformerVAE.transformer_lstm_vae import TransformerVAE


'''
    Entrypoint for train
    Get training data and train model
'''
def train(universal_config, train_config):
    anomaly_dataset = MyDataset(universal_config)
    x_train, x_threshold, x_test, y_test = anomaly_dataset.get_ad_data()
    print(f'Total len: {len(x_train) + len(x_threshold) + len(x_test)}\nx_train len: {len(x_train)}\nx_threshold len: {len(x_threshold)}\n'
          f'x_test len: {len(x_test)}, anomaly x_test len: {np.sum(y_test)}\nLoad data successfully!')

    vae = TransformerVAE(universal_config, train_config)
    vae.train(X_train=x_train, X_valid=x_threshold)

