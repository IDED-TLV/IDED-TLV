import keras
from keras import Sequential
from keras.layers import MultiHeadAttention, Dense, LayerNormalization, Layer


class TransformerBlock(Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        # Multi-Head Attention层
        self.attention = MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        # Feed Forward层
        self.dense_proj = Sequential(
            [Dense(dense_dim, activation="relu"),
             Dense(embed_dim), ]
        )
        # Add&Norm层1
        self.layernorm_1 = LayerNormalization()
        # Add&Norm层2
        self.layernorm_2 = LayerNormalization()

    def call(self, inputs):
        # 首先经过Multi-Head Attention层
        attention_output = self.attention(inputs, inputs)
        # 经过Add&Norm层1
        proj_input = self.layernorm_1(inputs + attention_output)
        # 经过Feed Forward层
        proj_output = self.dense_proj(proj_input)
        # 经过Add&Norm层2
        outputs = self.layernorm_2(proj_input + proj_output)
        return outputs
