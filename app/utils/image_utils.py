import os
import shutil

import numpy as np
from fastapi import UploadFile
from keras.src.utils import load_img, img_to_array
from app.utils.config import Config

upload_dir = Config.IMAGE_UPLOAD


def read_imagefile(file) -> np.ndarray:
    image = load_img(file, target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image


def save_image(upload_file: UploadFile) -> str:
    file_path = os.path.join(upload_dir, upload_file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return file_path


def is_valid_image(file: UploadFile) -> bool:
    valid_image_extensions = [".jpg", ".jpeg", ".png"]
    ext = os.path.splitext(file.filename)[1].lower()
    return ext in valid_image_extensions
