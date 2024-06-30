import tensorflow as tf
from app.models.custom_layers import LayerScale, ChannelAttention, SpatialAttention, DualAttention
from app.utils.config import Config


def load_model():
    with tf.keras.utils.custom_object_scope({
        'LayerScale': LayerScale,
        'ChannelAttention': ChannelAttention,
        'SpatialAttention': SpatialAttention,
        'DualAttention': DualAttention
    }):
        model = tf.keras.models.load_model(Config.CHECKPOINT_PATH)
    return model
