from imagededup.retrieval import CosEval
from imagededup.logger import return_logger
from keras.models import Model
from keras.applications import MobileNet as ConvNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten
from pathlib import PosixPath
from typing import Tuple, Dict, List
from PIL import Image
import os
import numpy as np

# TODO: Add a function for deleting detected duplicates (How?)
# TODO: check whether a valid path is given as input at every function


class CNN:
    def __init__(self) -> None:
        model_full = ConvNet(include_top=True)
        x = Flatten()(model_full.layers[-3].output)  # hard-coded slice at -3 (conv layer closest to output layer)
        self.model = Model(inputs=model_full.input, outputs=[x])
        self.TARGET_SIZE = (224, 224)
        self.BATCH_SIZE = 64
        self.logger = return_logger(__name__, os.getcwd())
        self.logger.info('Initialized: MobileNet pretrained on ImageNet dataset sliced at Convolutional layer closest'
                         'to o/p layer')
        self.result_score = None  # {query_filename: {retrieval_filename:score, ...}, ..}

    def _image_preprocess(self, pillow_image: Image) -> np.ndarray:
        im_res = pillow_image.resize(self.TARGET_SIZE)
        im_arr = np.array(im_res)
        return im_arr

    def _convert_to_array(self, path_image=None) -> np.ndarray:
        if isinstance(path_image, PosixPath):
            im = Image.open(path_image)
        elif isinstance(path_image, np.ndarray):
            im = path_image.astype('uint8')  # fromarray can't take float32/64
            im = Image.fromarray(im)
        im_arr = self._image_preprocess(im)
        return im_arr

    def cnn_image(self, path_image: str) -> np.ndarray:
        im_arr = self._convert_to_array(path_image)
        im_arr_proc = preprocess_input(im_arr)
        im_arr_shaped = np.array(im_arr_proc)[np.newaxis, :]
        return self.model.predict(im_arr_shaped)

    @staticmethod
    def _get_sub_dir(path_dir: PosixPath) -> str:
        return path_dir.parts[-1]

    @staticmethod
    def _get_parent_dir(path_dir: PosixPath) -> PosixPath:
        return path_dir.parent

    def _generator(self, path_dir: PosixPath) -> ImageDataGenerator:
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

    def cnn_dir(self, path_dir: PosixPath) -> Dict[str, np.ndarray]:
        self.logger.info('Start: Image feature generation')
        image_generator = self._generator(path_dir)
        feat_vec = self.model.predict_generator(image_generator, len(image_generator), verbose=1)
        self.logger.info('Completed: Image feature generation')
        filenames = [i.split('/')[-1] for i in image_generator.filenames]
        dict_file_feature = {filenames[i]: feat_vec[i] for i in range(len(filenames))}
        return dict_file_feature

    @staticmethod
    def _get_file_mapping_feat_vec(dict_file_feature: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[int, str]]:
        # order the dictionary to ensure consistent mapping between filenames and order of rows vector in similarity
        # matrix
        keys_in_order = [i for i in dict_file_feature]
        feat_vec_in = np.array([dict_file_feature[i] for i in keys_in_order])
        filenames_generated = keys_in_order
        filemapping_generated = dict(zip(range(len(filenames_generated)), filenames_generated))
        return feat_vec_in, filemapping_generated

    @staticmethod
    def _get_only_filenames(dict_of_dict: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
        dict_ret = {}
        for k, v in dict_of_dict.items():
            dict_ret[k] = list(v.keys())
        return dict_ret

    def _find_duplicates_dict(self, dict_file_feature: Dict[str, np.ndarray], threshold: float = 0.8, scores: bool = False):
        """Takes in dictionary {filename: vector}, detects duplicates and
                returns dictionary containing key as filename and value as a list of duplicate filenames"""
        feat_vec_in, filemapping_generated = self._get_file_mapping_feat_vec(dict_file_feature)
        self.logger.info('Start: Evaluating similarity for getting duplicates')
        self.result_score = CosEval(feat_vec_in, feat_vec_in).\
            get_retrievals_at_thresh(file_mapping_query=filemapping_generated,
                                     file_mapping_ret=filemapping_generated,
                                     thresh=threshold)
        self.logger.info('End: Evaluating similarity for getting duplicates')

        if scores:
            return self.result_score
        else:
            return self._get_only_filenames(self.result_score)

    def _find_duplicates_dir(self, path_dir: PosixPath, threshold: float = 0.8, scores: bool = False):
        """Takes in path of the directory on which duplicates are to be detected.
        Returns dictionary containing key as filename and value as a list of duplicate file names"""
        dict_file_feature = self.cnn_dir(path_dir)
        dict_ret = self._find_duplicates_dict(dict_file_feature=dict_file_feature, threshold=threshold, scores=scores)
        return dict_ret

    @staticmethod
    def _check_threshold_bounds(thresh: float) -> None:
        if not isinstance(thresh, float) or (thresh < -1.0 or thresh > 1.0):
            raise TypeError('Threshold must be a float between -1.0 and 1.0')

    def find_duplicates(self, path_or_dict, threshold: float = 0.8, scores: bool = False):
        self._check_threshold_bounds(thresh=threshold)
        if isinstance(path_or_dict, PosixPath):
            dict_ret = self._find_duplicates_dir(path_dir=path_or_dict, threshold=threshold, scores=scores)
        elif isinstance(path_or_dict, dict):
            dict_ret = self._find_duplicates_dict(dict_file_feature=path_or_dict, threshold=threshold, scores=scores)
        else:
            raise TypeError('Provide either a directory path variable to deduplicate or a dictionary of filenames and '
                            'vectors!')
        return dict_ret





