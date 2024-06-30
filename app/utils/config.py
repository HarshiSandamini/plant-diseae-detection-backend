import os

class Config:
    # Adjust the path to ensure it correctly locates the model file from your script's location
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CHECKPOINT_PATH = 'checkpoints/final_model.h5'
    IMAGE_UPLOAD = 'input_images'
