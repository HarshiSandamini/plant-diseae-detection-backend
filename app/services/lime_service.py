from keras.src.applications.convnext import preprocess_input

from lime import lime_image

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# # Load the model
# checkpoint_path = Config.CHECKPOINT_PATH
# with tf.keras.utils.custom_object_scope({
#     'LayerScale': LayerScale,
#     'ChannelAttention': ChannelAttention,
#     'SpatialAttention': SpatialAttention,
#     'DualAttention': DualAttention
# }):
#     model = load_model(checkpoint_path)

explainer = lime_image.LimeImageExplainer()

model = None


def predict(image):
    image = preprocess_input(image)
    return model.predict(image)


def get_lime_explanation(file_path, cnn_model):
    global model
    model = cnn_model
    img = load_img(file_path, target_size=(299, 299))
    img_array = img_to_array(img)
    explanation = explainer.explain_instance(img_array.astype('double'), predict, top_labels=4, hide_color=0,
                                             num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10,
                                                hide_rest=False)
    plt.imshow(temp)
    plt.show()
