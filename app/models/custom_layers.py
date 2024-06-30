import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, GlobalAveragePooling2D, Conv2D, Reshape, multiply
from tensorflow.keras.initializers import Constant


# Define custom layers
class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, input_dim, ratio=8, name=None):
        super(ChannelAttention, self).__init__(name=name)
        self.avg_pool = GlobalAveragePooling2D()
        self.fc1 = Dense(input_dim // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True,
                         bias_initializer='zeros')
        self.fc2 = Dense(input_dim, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
        self.sigmoid = tf.keras.activations.sigmoid

    def call(self, inputs):
        avg_pooled = self.avg_pool(inputs)
        avg_pooled = Reshape((1, 1, avg_pooled.shape[1]))(avg_pooled)
        avg_out = self.fc2(self.fc1(avg_pooled))
        return self.sigmoid(avg_out)


class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size=7, name=None):
        super(SpatialAttention, self).__init__(name=name)
        self.conv = Conv2D(1, kernel_size, padding='same', activation='sigmoid', kernel_initializer='he_normal',
                           use_bias=False)

    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_out = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_out, max_out], axis=-1)
        return self.conv(concat)


class DualAttention(tf.keras.layers.Layer):
    def __init__(self, input_dim, ratio=8, kernel_size=7, name=None):
        super(DualAttention, self).__init__(name=name)
        self.channel_attention = ChannelAttention(input_dim, ratio, name=f"{name}_channel_attention" if name else None)
        self.spatial_attention = SpatialAttention(kernel_size, name=f"{name}_spatial_attention" if name else None)

    def call(self, inputs):
        ca = self.channel_attention(inputs)
        ca_out = multiply([inputs, ca])
        sa = self.spatial_attention(ca_out)
        return multiply([ca_out, sa])


class LayerScale(Layer):
    def __init__(self, init_values, **kwargs):
        # Extract 'projection_dim' if it is in kwargs and discard it
        kwargs.pop('projection_dim', None)
        super(LayerScale, self).__init__(**kwargs)
        self.init_values = init_values

    def build(self, input_shape):
        self.scale = self.add_weight(name='scale', shape=(input_shape[-1],), initializer=Constant(self.init_values),
                                     trainable=True)
        super(LayerScale, self).build(input_shape)

    def call(self, inputs):
        return inputs * self.scale

