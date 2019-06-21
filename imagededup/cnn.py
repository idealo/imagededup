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

# TODO: check whether a valid path is given as input at every function
# TODO: Add options for making CNN forward pass quicker. (in _generator)


class CNN:
    """
    Finds duplicates using CNN and/or generates CNN features given a single image or a directory of images.
    The module can be used for 2 purposes: Feature generation and duplicate detection.

    Feature generation:
    To propagate an image through a Convolutional Neural Network architecture and generate features. The generated
    features can be used at a later time for deduplication. There are two possibilities to get features:
    1. At a single image level: Using the function 'cnn_image', the CNN feature for a single image can be obtained.
    Example usage:
    ```
    from imagededup import cnn
    mycnn = cnn.CNN()
    feature_vector = mycnn.cnn_image(Path('path/to/image.jpg'))
    ```
    2. At a directory level: In case features for several images need to be generated, the images can be placed in a
    directory and features for all of the images can be obtained using the 'cnn_dir' function.
    Example usage:
    ```
    from imagededup import cnn
    mycnn = cnn.CNN()
    dict_file_feat = mycnn.cnn_dir(Path('path/to/directory'))
    ```

    Duplicate detection:
    Find duplicates either using the feature mapping generated previously using 'cnn_dir' or using a Path to the
    directory that contains the images that need to be deduplicated. There are 2 inputs that can be provided to the
    find_duplicates function:
    1. Dictionary generated using 'cnn_dir' function above.
    Example usage:
    ```
    from imagededup import cnn
    mycnn = cnn.CNN()
    dict_ret_with_dict_inp = mycnn.find_duplicates(dict_file_feat, threshold=0.9, scores=True)
    ```
    2. Using the Path of the directory where all images are present.
    Example usage:
    ```
    from imagededup import cnn
    mycnn = cnn.CNN()
    dict_ret_path = mycnn.find_duplicates(Path('path/to/directory'), threshold=0.9, scores=True)
    ```

    If a list of file names to remove are desired, then the function find_duplicates_to_remove can be used with either the
    path to the image directory as input or the dictionary with features. A threshold for similarity should be considered.

    Example usage:
        ```
        from imagededup import cnn
        mycnn = cnn.CNN()
        list_of_files_to_remove = mycnn.find_duplicates_to_remove(Path('path/to/images/directory'), threshold=0.9)
        ```

    """

    def __init__(self) -> None:
        """
        Initializes a keras MobileNet model that is sliced at the convolutional layer closest to the output layer.
        Sets the batch size for keras generators to be 64 samples. Sets the input image size to (224, 224) for providing
        as input to MobileNet model. Initiates a results_score variable to None.
        """

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
        """
        Resizes and typecasts a pillow image to numpy array.

        :param pillow_image: A Pillow type image to be processed.
        :return: A numpy array of processed image.
        """

        im_res = pillow_image.resize(self.TARGET_SIZE)
        im_arr = np.array(im_res)
        return im_arr

    def _convert_to_array(self, path_image=None) -> np.ndarray:
        """
        Accepts either path of an image or a numpy array and processes it to feed it to CNN.

        :param path_image: PosixPath to the image file or Image typecast to numpy array.
        :return: A processed image as numpy array
        """

        if isinstance(path_image, PosixPath):
            im = Image.open(path_image)
        elif isinstance(path_image, np.ndarray):
            im = path_image.astype('uint8')  # fromarray can't take float32/64
            im = Image.fromarray(im)
        im_arr = self._image_preprocess(im)
        return im_arr

    def cnn_image(self, path_image: str) -> np.ndarray:
        """
        Generates CNN features for a single image.

        :param path_image: PosixPath to the image file or Image typecast to numpy array.
        :return: Features for the image in the form of numpy array.

        Example usage:
        ```
        from imagededup import cnn
        mycnn = cnn.CNN()
        feature_vector = mycnn.cnn_image(Path('path/to/image.jpg'))
        ```
        """

        im_arr = self._convert_to_array(path_image)
        im_arr_proc = preprocess_input(im_arr)
        im_arr_shaped = np.array(im_arr_proc)[np.newaxis, :]
        return self.model.predict(im_arr_shaped)

    @staticmethod
    def _get_sub_dir(path_dir: PosixPath) -> str:
        """
        Extracts sub directory of a PosixPath to a directory.

        :param path_dir: PosixPath to a directory.
        :return: Name of the subdirectory as a string.
        """

        return path_dir.parts[-1]

    @staticmethod
    def _get_parent_dir(path_dir: PosixPath) -> PosixPath:
        """
        Extracts parent directory of a PosixPath to a directory.

        :param path_dir: PosixPath to a directory.
        :return: Name of the parent directory as a PosixPath.
        """

        return path_dir.parent

    def _generator(self, path_dir: PosixPath) -> ImageDataGenerator:
        """
        Declares a keras ImageDataGenerator to obtain CNN features for all the images in a given directory.

        :param path_dir: PosixPath to the directory containing all the images.
        :return: An initialized keras ImageDataGenerator.
        """

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
        """
        Generates CNN features for all images in a given directory of images.

        :param path_dir: PosixPath to the directory containing all the images.
        :return: A dictionary that contains a mapping of filenames and corresponding numpy array of CNN features.
        For example:
        mapping = CNN().cnn_dir(Path('path/to/directory'))
        'mapping' contains: {'Image1.jpg': np.array([1.0, -0.2, ...]), 'Image2.jpg': np.array([0.3, 0.06, ...]), ...}

        Example usage:
        ```
        from imagededup import cnn
        mycnn = cnn.CNN()
        dict_file_feat = mycnn.cnn_dir(Path('path/to/directory'))
        ```
        """

        self.logger.info('Start: Image feature generation')
        image_generator = self._generator(path_dir)
        feat_vec = self.model.predict_generator(image_generator, len(image_generator), verbose=1)
        self.logger.info('Completed: Image feature generation')
        filenames = [i.split('/')[-1] for i in image_generator.filenames]
        dict_file_feature = {filenames[i]: feat_vec[i] for i in range(len(filenames))}
        return dict_file_feature

    @staticmethod
    def _get_file_mapping_feat_vec(dict_file_feature: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        Splits mapping dictionary into feature vector matrix and a dictionray that maintains mapping between the row of
        the feature matrix and the image filename.

        :param dict_file_feature: {'Image1.jpg': np.array([1.0, -0.2, ...]), 'Image2.jpg': np.array([0.3, 0.06, ...]), ...}
        :return: feat_vec_in: A numpy ndarray of size (number of queries, number of features).
        filemapping_generated: A dictionary mapping the row number of 'feat_vec_in' to the image filename.
        """

        # order the dictionary to ensure consistent mapping between filenames and order of rows vector in similarity
        # matrix
        keys_in_order = [i for i in dict_file_feature]
        feat_vec_in = np.array([dict_file_feature[i] for i in keys_in_order])
        filenames_generated = keys_in_order
        filemapping_generated = dict(zip(range(len(filenames_generated)), filenames_generated))
        return feat_vec_in, filemapping_generated

    @staticmethod
    def _get_only_filenames(dict_of_dict_dups: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
        """
        Derives list of file names of duplicates for each query image.

        :param dict_of_dict_dups: dictionary of dictionaries {'image1.jpg': {'image1_duplicate1.jpg':<similarity-score>,
        'image1_duplicate2.jpg':<similarity-score>, ..}, 'image2.jpg':{'image1_duplicate1.jpg':<similarity-score>,..}}
        :return: dict_ret: A dictionary consisting query file names as key and a list of duplicate file names as value.
        {'image1.jpg': ['image1_duplicate1.jpg', 'image1_duplicate2.jpg']
        'image2.jpg':['image1_duplicate1.jpg',..], ..}
        """

        dict_ret = {}
        for k, v in dict_of_dict_dups.items():
            dict_ret[k] = list(v.keys())
        return dict_ret

    def _find_duplicates_dict(self, dict_file_feature: Dict[str, np.ndarray], threshold: float = 0.8, scores: bool = False):
        """Takes in dictionary {filename: vector}, detects duplicates above the given threshold and
                returns dictionary containing key as filename and value as a list of duplicate filenames. Optionally,
                the similarity scores could be returned instead of duplicate file name for each query file.

        :param dict_file_feature: Dictionary with keys as file names and values as numpy arrays which represent the CNN
        feature for the key image file.
        :param threshold: Cosine similarity above which retrieved duplicates are valid.
        :param scores: Boolean indicating whether similarity scores are to be returned along with retrieved duplicates.
        :return: if scores is True, then a dictionary of the form {'image1.jpg': {'image1_duplicate1.jpg':<similarity-score>, 'image1_duplicate2.jpg':<similarity-score>, ..}, 'image2.jpg':{'image1_duplicate1.jpg':<similarity-score>,..}}
        if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg', 'image1_duplicate2.jpg']
        'image2.jpg':['image1_duplicate1.jpg',..], ..}
                """

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
        """Takes in path of the directory on which duplicates are to be detected above the given threshold.
        Returns dictionary containing key as filename and value as a list of duplicate file names.

        :param path_dir: PosixPath to the directory containing all the images.
        :param threshold: Cosine similarity above which retrieved duplicates are valid.
        :param scores: Boolean indicating whether similarity scores are to be returned along with retrieved duplicates.
        :return: if scores is True, then a dictionary of the form {'image1.jpg': {'image1_duplicate1.jpg':<similarity-score>, 'image1_duplicate2.jpg':<similarity-score>, ..}, 'image2.jpg':{'image1_duplicate1.jpg':<similarity-score>,..}}
        if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg', 'image1_duplicate2.jpg']
        'image2.jpg':['image1_duplicate1.jpg',..], ..}
        """

        dict_file_feature = self.cnn_dir(path_dir)
        dict_ret = self._find_duplicates_dict(dict_file_feature=dict_file_feature, threshold=threshold, scores=scores)
        return dict_ret

    @staticmethod
    def _check_threshold_bounds(thresh: float) -> None:
        """
        Checks if provided threshold is valid. Raises TypeError is wrong threshold variable type is passed or a value out
        of range is supplied.

        :param thresh: Threshold value (must be float between -1.0 and 1.0)
        """

        if not isinstance(thresh, float) or (thresh < -1.0 or thresh > 1.0):
            raise TypeError('Threshold must be a float between -1.0 and 1.0')

    def find_duplicates(self, path_or_dict, threshold: float = 0.8, scores: bool = False):
        """
        Finds duplicates. Raises TypeError if supplied directory path isn't a Path variable or a valid dictionary isn't
        supplied.

        :param path_or_dict: PosixPath to the directory containing all the images or dictionary with keys as file names
        and values as numpy arrays which represent the CNN feature for the key image file.
        :param threshold: Threshold value (must be float between -1.0 and 1.0)
        :param scores: Boolean indicating whether similarity scores are to be returned along with retrieved duplicates.
        :return: if scores is True, then a dictionary of the form {'image1.jpg': {'image1_duplicate1.jpg':<similarity-score>, 'image1_duplicate2.jpg':<similarity-score>, ..}, 'image2.jpg':{'image1_duplicate1.jpg':<similarity-score>,..}}
        if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg', 'image1_duplicate2.jpg'], 'image2.jpg':['image1_duplicate1.jpg',..], ..}

            Example usage:
        ```
        from imagededup import cnn
        mycnn = cnn.CNN()
        dict_ret_with_dict_inp = mycnn.find_duplicates(dict_file_feat, threshold=0.9, scores=True)

        OR

        from imagededup import cnn
        mycnn = cnn.CNN()
        dict_ret_path = mycnn.find_duplicates(Path('path/to/directory'), threshold=0.9, scores=True)
        ```
        """

        self._check_threshold_bounds(thresh=threshold)
        if isinstance(path_or_dict, PosixPath):
            dict_ret = self._find_duplicates_dir(path_dir=path_or_dict, threshold=threshold, scores=scores)
        elif isinstance(path_or_dict, dict):
            dict_ret = self._find_duplicates_dict(dict_file_feature=path_or_dict, threshold=threshold, scores=scores)
        else:
            raise TypeError('Provide either a directory path variable to deduplicate or a dictionary of filenames and '
                            'vectors!')
        return dict_ret

    def find_duplicates_to_remove(self, path_or_dict, threshold: float = 0.8) -> List:
        """
        Gives out a list of image file names to remove based on the similarity threshold.
        :param path_or_dict: PosixPath to the directory containing all the images or dictionary with keys as file names
        and values as numpy arrays which represent the CNN feature for the key image file.
        :param threshold: Threshold value (must be float between -1.0 and 1.0)
        :return: List of image file names that should be removed.

        Example usage:
        ```
        from imagededup import cnn
        mycnn = cnn.CNN()
        list_of_files_to_remove = mycnn.find_duplicates_to_remove(Path('path/to/images/directory'), threshold=0.9)
        ```
        """

        dict_ret = self.find_duplicates(path_or_dict=path_or_dict, threshold=threshold, scores=False)
        # iterate over dict_ret keys, get value for the key and delete the dict keys that are in the value list

        list_of_files_to_remove = []

        for k, v in dict_ret.items():
            if k not in list_of_files_to_remove:
                list_of_files_to_remove.extend(v)
        return list_of_files_to_remove



