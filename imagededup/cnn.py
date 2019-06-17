from imagededup.retrieval import CosEval
from imagededup.logger import return_logger
from keras.models import Model
from keras.applications import MobileNet as ConvNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten
from pathlib import Path, PosixPath
from typing import Tuple
from PIL import Image
import os
import numpy as np

# TODO: Add a function for deleting detected duplicates
# TODO: Write tests


class CNN:
    def __init__(self) -> None:
        self.model_full = ConvNet(include_top=True)
        x = Flatten()(self.model_full.layers[-3].output)  # hard-coded slice at -3 (conv layer closest to output layer)
        self.model = Model(inputs=self.model_full.input, outputs=[x])
        self.TARGET_SIZE = (224, 224)
        self.BATCH_SIZE = 64
        self.file_mapping = None
        self.feat_vec = None
        self.logger = return_logger(__name__, os.getcwd())
        self.logger.info('Initialized: MobileNet pretrained on ImageNet dataset sliced at Convolutional layer closest'
                         'to o/p layer')

    def _image_preprocess(self, pillow_image: Image) -> np.ndarray:
        im_res = pillow_image.resize(self.TARGET_SIZE)
        im_arr = np.array(im_res)
        return im_arr

    def _convert_to_array(self, path_image: None) -> np.ndarray:
        try:
            if isinstance(path_image, Path):
                im = Image.open(path_image)
            elif isinstance(path_image, np.ndarray):
                im = path_image.astype('uint8')  # fromarray can't take float32/64
                im = Image.fromarray(im)
            else:
                raise Exception
            im_arr = self._image_preprocess(im)
            return im_arr
        except Exception:
            print('Check Input Format! Input should be either a Path Variable or a numpy array!')
            raise

    def cnn_image(self, path_image: str) -> np.ndarray:
        im_arr = self._convert_to_array(path_image)
        im_arr_proc = preprocess_input(im_arr)
        im_arr_shaped = np.array(im_arr_proc)[np.newaxis, :]
        return self.model.predict(im_arr_shaped)

    @staticmethod
    def _get_sub_dir(path_dir: Path) -> str:
        return path_dir.parts[-1]

    @staticmethod
    def _get_parent_dir(path_dir: Path) -> PosixPath:
        return path_dir.parent

    def _generator(self, path_dir: Path) -> ImageDataGenerator:
        sub_dir = self._get_sub_dir(path_dir)
        img_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
        parent_dir = self._get_parent_dir(path_dir)

        img_batches = img_gen.flow_from_directory(
            directory=parent_dir,
            target_size=self.TARGET_SIZE,
            batch_size=self.BATCH_SIZE,
            color_mode='rgb',
            shuffle=False,
            classes=[sub_dir],
            class_mode=None
        )
        return img_batches

    def cnn_dir(self, path_dir: Path) -> Tuple[np.ndarray, dict]:
        self.logger.info('Start: Image feature generation')
        image_generator = self._generator(path_dir)
        self.feat_vec = self.model.predict_generator(image_generator, len(image_generator), verbose=1)
        self.logger.info('Completed: Image feature generation')
        filenames = [i.split('/')[-1] for i in image_generator.filenames]
        dict_file_feature = {filenames[i]: self.feat_vec[i] for i in range(len(filenames))}
        self.file_mapping = dict(zip(range(len(image_generator.filenames)), filenames))
        return dict_file_feature

    def find_duplicates(self, path_dir: PosixPath, threshold: float = 0.8) -> dict:
        """Takes in path of the directory on which duplicates are to be detected.
        Returns dictionary containing key as filename and value as a list of duplicate filenames"""
        _ = self.cnn_dir(path_dir)
        self.logger.info('Start: Evaluating similarity for getting duplicates')
        dict_ret = CosEval(self.feat_vec, self.feat_vec).get_retrievals_at_thresh(file_mapping_query=self.file_mapping,
                                                                                  file_mapping_ret=self.file_mapping,
                                                                                  thresh=threshold)
        self.logger.info('End: Evaluating similarity for getting duplicates')
        return dict_ret
