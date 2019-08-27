from imagededup.utils.image_utils import load_image, preprocess_image
from imagededup.utils.logger import return_logger
from keras.models import Model
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.layers import Flatten, MaxPooling2D, AveragePooling2D, concatenate
from keras.utils import Sequence
from pathlib import Path, PosixPath
from typing import Tuple, Dict, List, Optional, Callable, Union
import os
import numpy as np


class DataGenerator(Sequence):
    """Class inherits from Keras Sequence base object, allows to use multiprocessing in .fit_generator.

    Attributes:
        image_dir: Path of image directory.
        batch_size: Number of images per batch.
        basenet_preprocess: Basenet specific preprocessing function.
        target_size: Dimensions that images get resized into when loaded.
    """

    def __init__(
        self,
        image_dir: PosixPath,
        batch_size: int,
        basenet_preprocess: Callable,
        target_size: Tuple[int, int],
    ) -> None:
        """Inits DataGenerator object.
        """
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.basenet_preprocess = basenet_preprocess
        self.target_size = target_size

        self._get_image_files()
        self.on_epoch_end()

    def _get_image_files(self):
        self.invalid_image_idx = []
        self.image_files = [
            i.absolute() for i in self.image_dir.glob('*') if not i.name.startswith('.')
        ]  # ignore hidden files

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        self.indexes = np.arange(len(self.image_files))
        self.valid_image_files = [
            j for i, j in enumerate(self.image_files) if i not in self.invalid_image_idx
        ]

    def __len__(self):
        """Number of batches in the Sequence."""
        return int(np.ceil(len(self.image_files) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.array, np.array]:
        """Gets batch at position `index`.
        """
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch_samples = [self.image_files[i] for i in batch_indexes]
        X, y = self._data_generator(batch_samples)
        return X, y

    def _data_generator(self, image_files: List[PosixPath]) -> Tuple[np.array, np.array]:
        """Generates data from samples in specified batch."""
        #  initialize images and labels tensors for faster processing
        X = np.empty((len(image_files), *self.target_size, 3))
        y = np.ones(len(image_files))

        invalid_image_idx = []
        for i, image_file in enumerate(image_files):
            # load and randomly augment image
            img = load_image(image_file=image_file, target_size=self.target_size, grayscale=False)

            if img is not None:
                X[i, :] = img

            else:
                invalid_image_idx.append(i)

        if invalid_image_idx:
            X = np.delete(X, invalid_image_idx, axis=0)
            self.invalid_image_idx += invalid_image_idx

        # apply basenet specific preprocessing
        # input is 4D numpy array of RGB values within [0, 255]
        X = self.basenet_preprocess(X)

        return X, y


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

    If a list of file names to remove are desired, then the function find_duplicates_to_remove can be used with either
    the path to the image directory as input or the dictionary with features. A threshold for similarity should be
    considered.

    Example usage:
        ```
        from imagededup import cnn
        mycnn = cnn.CNN()
        list_of_files_to_remove = mycnn.find_duplicates_to_remove(Path('path/to/images/directory'), threshold=0.9)
        ```

    """

    def __init__(self) -> None:
        """
        Initializes a keras MobileNet model that is sliced at the last convolutional layer.
        Sets the batch size for keras generators to be 64 samples. Sets the input image size to (224, 224) for providing
        as input to MobileNet model. Initiates a results_score variable to None.
        """

        self.target_size = (224, 224)
        self.batch_size = 64
        self.logger = return_logger(__name__, os.getcwd())
        self.result_score = None  # {query_filename: {retrieval_filename:score, ...}, ..}

        self._build_model()

    def _build_model(self):
        model = MobileNet(input_shape=(224, 224, 3), include_top=False)

        # add AdaptiveConcatPooling
        # source: https://towardsdatascience.com/how-to-build-an-image-duplicate-finder-f8714ddca9d2
        # we use a pooling window of 3 instead of 2 as we have more channels in last conv layer than resnet
        x_max = MaxPooling2D((3, 3), padding='same')(model.output)
        x_avg = AveragePooling2D((3, 3), padding='same')(model.output)
        x = concatenate([x_max, x_avg])

        x = Flatten()(x)
        self.model = Model(inputs=model.input, outputs=[x])

        self.logger.info(
            'Initialized: MobileNet pretrained on ImageNet dataset sliced at last conv layer and added '
            'AdaptiveConcatPooling'
        )

    def _get_cnn_features_single(self, image_array: np.ndarray) -> np.ndarray:
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
        image_pp = preprocess_input(image_array)
        image_pp = np.array(image_pp)[np.newaxis, :]
        return self.model.predict(image_pp)

    def _get_cnn_features_batch(self, image_dir: PosixPath) -> Dict[str, np.ndarray]:
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

        self.data_generator = DataGenerator(
            image_dir=image_dir,
            batch_size=self.batch_size,
            target_size=self.target_size,
            basenet_preprocess=preprocess_input,
        )

        feat_vec = self.model.predict_generator(
            self.data_generator, len(self.data_generator), verbose=1
        )
        self.logger.info('Completed: Image feature generation')

        filenames = [i.name for i in self.data_generator.valid_image_files]

        return {filenames[i]: feat_vec[i] for i in range(len(filenames))}

    def encode_image(
        self,
        image_file: Optional[Union[PosixPath, str]] = None,
        image_array: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if isinstance(image_file, str):
            image_file = Path(image_file)

        if isinstance(image_file, PosixPath):
            image_pp = load_image(
                image_file=image_file, target_size=self.target_size, grayscale=False
            )

        elif isinstance(image_array, np.ndarray):
            image_pp = preprocess_image(
                image=image_array, target_size=self.target_size, grayscale=False
            )
        else:
            raise ValueError('Please provide either image file path or image array!')

        return self._get_cnn_features_single(image_pp) if isinstance(image_pp, np.ndarray) else None

    def encode_images(self, image_dir: Union[PosixPath, str]) -> Dict:
        if isinstance(image_dir, str):
            image_dir = Path(image_dir)

        if not image_dir.is_dir(image_dir):
            raise ValueError('Please provide a valid directory path!')

        return self._get_cnn_features_batch(image_dir)

    @staticmethod
    def _get_file_mapping_feat_vec(
        dict_file_feature: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        Splits mapping dictionary into feature vector matrix and a dictionray that maintains mapping between the row of
        the feature matrix and the image filename.
        :param dict_file_feature: {'Image1.jpg': np.array([1.0, -0.2, ...]),
        'Image2.jpg': np.array([0.3, 0.06, ...]), ...}
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

    def _find_duplicates_dict(
        self, dict_file_feature: Dict[str, np.ndarray], threshold: float = 0.8, scores: bool = False
    ) -> Dict:
        """Takes in dictionary {filename: vector}, detects duplicates above the given threshold and
                returns dictionary containing key as filename and value as a list of duplicate filenames. Optionally,
                the similarity scores could be returned instead of duplicate file name for each query file.
        :param dict_file_feature: Dictionary with keys as file names and values as numpy arrays which represent the CNN
        feature for the key image file.
        :param threshold: Cosine similarity above which retrieved duplicates are valid.
        :param scores: Boolean indicating whether similarity scores are to be returned along with retrieved duplicates.
        :return: if scores is True, then a dictionary of the form {'image1.jpg':
        {'image1_duplicate1.jpg':<similarity-score>, 'image1_duplicate2.jpg':<similarity-score>, ..}, 'image2.jpg':
        {'image1_duplicate1.jpg':<similarity-score>,..}}
        if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg',
        'image1_duplicate2.jpg'], 'image2.jpg':['image1_duplicate1.jpg',..], ..}
                """

        feat_vec_in, filemapping_generated = self._get_file_mapping_feat_vec(dict_file_feature)
        self.logger.info('Start: Evaluating similarity for getting duplicates')
        self.result_score = CosEval(feat_vec_in, feat_vec_in).get_retrievals_at_thresh(
            file_mapping_query=filemapping_generated,
            file_mapping_ret=filemapping_generated,
            thresh=threshold,
        )
        self.logger.info('End: Evaluating similarity for getting duplicates')

        if scores:
            return self.result_score
        else:
            return self._get_only_filenames(self.result_score)

    def get_encodings(self, image_dir: PosixPath):
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
        check_directory_files(image_dir, return_file=False)
        self.logger.info('Start: Image feature generation')

        image_generator = self._generator(image_dir)
        encodings = self.model.predict_generator(image_generator, len(image_generator), verbose=1)
        self.logger.info('Completed: Image feature generation')
        filenames = [i.split('/')[-1] for i in image_generator.filenames]

        return encodings, filenames

    def get_cosine_distances(self):
        self.cosine_distances = cosine_similarity(self.encodings)

    def find_duplicates(self, threshold, image_dir=None, encodings=None, scores: bool = False):
        self._check_threshold_bounds(threshold)

        if image_dir:
            self.encodings, self.filenames = self.get_encodings(image_dir)

        elif encodings:
            self.filenames = list(self.encodings.keys())
            self.encodings = np.array(self.encodings.values())

        else:
            raise ValueError('Provide either image_dir or encoding dictionary!')

        self.get_cosine_distances()

        self.duplicates = {}
        for i, row in enumerate(self.cosine_distances):
            duplicates_idx = list(np.nonzero(row >= threshold)[0])

            duplicates_files = [self.filenames[j] for j in duplicates_idx]

            if scores and duplicates_files:
                duplicates_scores = [row[j] for j in duplicates_idx]
                self.duplicates[self.filenames[i]] = list(zip(duplicates_files, duplicates_scores))
            else:
                self.duplicates[self.filenames[i]] = duplicates_files

    def find_duplicates_to_remove(
        self, threshold, image_dir=None, encodings=None, scores: bool = False
    ):
        dict_ret = self.find_duplicates(
            path_or_dict=path_or_dict, threshold=threshold, scores=False
        )
        return get_files_to_remove(dict_ret)

    def _find_duplicates_dir(
        self, path_dir: PosixPath, threshold: float = 0.8, scores: bool = False
    ) -> Dict:
        """Takes in path of the directory on which duplicates are to be detected above the given threshold.
        Returns dictionary containing key as filename and value as a list of duplicate file names.
        :param path_dir: PosixPath to the directory containing all the images.
        :param threshold: Cosine similarity above which retrieved duplicates are valid.
        :param scores: Boolean indicating whether similarity scores are to be returned along with retrieved duplicates.
        :return: if scores is True, then a dictionary of the form {'image1.jpg': {'image1_duplicate1.jpg':
        <similarity-score>, 'image1_duplicate2.jpg':<similarity-score>, ..}, 'image2.jpg':{'image1_duplicate1.jpg':
        <similarity-score>,..}}
        if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg',
        'image1_duplicate2.jpg'], 'image2.jpg':['image1_duplicate1.jpg',..], ..}
        """

        dict_file_feature = self.cnn_dir(path_dir)
        dict_ret = self._find_duplicates_dict(
            dict_file_feature=dict_file_feature, threshold=threshold, scores=scores
        )
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
