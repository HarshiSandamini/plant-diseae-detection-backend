import os
import tensorflow as tf
from keras import Model
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import cv2


# from app.models.custom_layers import LayerScale, ChannelAttention, SpatialAttention, DualAttention
# from app.config import Config

# checkpoint_path = Config.CHECKPOINT_PATH
#
# with tf.keras.utils.custom_object_scope({
#     'LayerScale': LayerScale,
#     'ChannelAttention': ChannelAttention,
#     'SpatialAttention': SpatialAttention,
#     'DualAttention': DualAttention}):
#     model = load_model(checkpoint_path)


@tf.function
def get_intermediate_output(model, img_array):
    attention_layer = model.get_layer('dual_attention_layer')
    intermediate_model = Model(inputs=model.input, outputs=attention_layer.output)
    return intermediate_model(img_array, training=False)


def visualize_attention(image_path, model):
    # Load image
    img = load_img(image_path, target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    # Get the class labels
    class_labels = ["0", "1", "2", "3"]

    # Get intermediate layer output
    intermediate_output = get_intermediate_output(model, img_array)

    # Average the intermediate output across channels
    attention_map = np.mean(intermediate_output[0], axis=-1)

    # Normalize the attention map
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

    # Resize the attention map to match the input image size
    attention_map = cv2.resize(attention_map, (299, 299))

    # Overlay the attention map on the original image
    overlay = img_to_array(load_img(image_path, target_size=(299, 299)))
    overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min())
    attention_overlay = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
    attention_overlay = cv2.cvtColor(attention_overlay, cv2.COLOR_BGR2RGB)
    overlay = 0.6 * overlay + 0.4 * attention_overlay
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Plot original image, attention map, and overlay
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(load_img(image_path))
    ax[0].set_title(f"Original Image")
    ax[0].axis('off')

    ax[1].imshow(attention_map, cmap='jet')
    ax[1].set_title("Attention Map")
    ax[1].axis('off')

    ax[2].imshow(overlay)
    ax[2].set_title(f"Prediction: {class_labels[predicted_class]}")
    ax[2].axis('off')

    plt.show()
