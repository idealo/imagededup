from keras.models import Model
from keras.applications import MobileNet as CNN
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten
from pathlib import Path
from typing import Tuple

import numpy as np


class Cnn:
    def __init__(self) -> None:
        self.model_full = CNN(include_top=True)
        x = Flatten()(self.model_full.layers[-3].output)
        self.model = Model(inputs=self.model_full.input, outputs=[x])
        self.TARGET_SIZE = (224, 224)
        self.BATCH_SIZE = 64
        self.IMG_DIR = '/Users/tanuj.jain/Documents/dedup-data/Transformed_dataset/'

    def generator(self, sub_dir: Path) -> ImageDataGenerator:
        img_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

        img_batches_rets = img_gen.flow_from_directory(
            directory=self.IMG_DIR,
            target_size=self.TARGET_SIZE,
            batch_size=self.BATCH_SIZE,
            color_mode='rgb',
            shuffle=False,
            classes=[sub_dir],
            class_mode=None
        )
        return img_batches_rets

    def generate_features(self, sub_dir: Path) -> Tuple[np.ndarray, dict]:
        image_generator = self.generator(sub_dir)
        feat_vec = self.model.predict_generator(image_generator, len(image_generator), verbose=1)
        file_mapping = dict(zip(range(len(image_generator.filenames)), image_generator.filenames))
        return feat_vec, file_mapping