from __future__ import print_function
import os
import tensorflow as tf
from keras import Input, Model
from keras.layers import Dense, LSTM, Lambda
import warnings
from keras.losses import mean_squared_error as mse
from models.transformer.TransformerVAE.transformer_block import TransformerBlock
from models.transformer.TransformerVAE.positional_embedding import PositionalEmbedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import json

warnings.filterwarnings("ignore")

class TransformerVAE:
    def __init__(self, universal_config, train_config):
        super(TransformerVAE, self).__init__()
        keys_to_include_universal = ["exp_name", "model_name", "time_step", "feature_size"]
        for key in keys_to_include_universal:
            setattr(self, key, universal_config[key])

        keys_to_include_train = ["batch_size", "epoch", "lr", "beta", "latent_dim", "dense_dim", "num_heads", "num_layers"]
        for key in keys_to_include_train:
            if key in train_config:
                setattr(self, key, train_config[key])

        # Input and position embedding
        inputs = Input(shape=(self.time_step, self.feature_size))
        positional_embedding = PositionalEmbedding(d_model=self.feature_size, max_len=self.time_step)
        encoder = TransformerBlock(embed_dim=self.feature_size, dense_dim=self.dense_dim, num_heads=self.num_heads)
        x = positional_embedding.call(inputs)
        # encoding
        for _ in range(self.num_layers):
            x = encoder.call(x)

        # reparameters
        mean = Dense(self.latent_dim)(x)
        log_var = Dense(self.latent_dim)(x)
        z = self.reparameterize(mean, log_var)

        # decoding
        decoded = LSTM(self.latent_dim, return_sequences=True)(z)
        outputs = Dense(self.feature_size)(decoded)

        # VAE
        self.model = Model(inputs, outputs)

        recon_loss = tf.reduce_mean(tf.reduce_sum(mse(inputs, outputs), axis=-1))
        kl_loss = - 0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=-1)
        vae_loss = recon_loss + self.beta * kl_loss
        self.model.add_loss(vae_loss)
        optimizer = Adam(learning_rate=self.lr)
        self.model.compile(optimizer=optimizer)

    def reparameterize(self, mean, log_var):
        epsilon = tf.random.normal(shape=(tf.shape(mean)[0], tf.shape(mean)[1], tf.shape(mean)[2]))
        return epsilon * tf.exp(log_var) + mean

    def train(self, X_train, X_valid):
        early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='min')
        model_checkpoint = ModelCheckpoint(
            filepath=os.path.join('model_pth', self.model_name + '.h5'),
            monitor='val_loss',
            save_best_only=True,  # ����val_loss�и���ʱ�ű���ģ��
            mode='min',
            verbose=1
        )
        train_info = self.model.fit(X_train, X_train, batch_size=self.batch_size, epochs=self.epoch, shuffle=True,
                       callbacks=[early_stop, model_checkpoint], validation_data=(X_valid, X_valid))

        with open(os.path.join('log', self.exp_name, 'train_log.json'), 'w') as f:
            json.dump(train_info.history, f, indent=4)

    def predict(self, X, batch_size):
        return self.model.predict(X, batch_size=batch_size)

