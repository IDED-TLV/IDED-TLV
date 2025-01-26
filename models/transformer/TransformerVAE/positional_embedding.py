import keras
from keras.layers import Layer
import tensorflow as tf
import numpy as np
import math


class PositionalEmbedding(Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = np.zeros((max_len, d_model), dtype=np.float32)

        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        if d_model % 2 == 1:
            # 如果d_model是奇数，最后一列使用余弦编码
            pe[:, -1] = np.cos(position * div_term[-1]).squeeze()
            # 剩下的偶数索引列使用余弦编码
            pe[:, 1:-1:2] = np.cos(position * div_term[:-1])
        else:
            # 如果d_model是偶数，所有偶数索引列使用余弦编码
            pe[:, 1::2] = np.cos(position * div_term)

        pe = pe[np.newaxis, ...]
        self.pe = tf.constant(pe, dtype=tf.float32)

    def call(self, x):
        # `x` is expected to be of shape [batch_size, seq_length]
        # We use broadcasting to apply the positional encoding to each sequence in the batch.
        seq_length = x.shape[1]
        return x + self.pe[:, :seq_length, :]
