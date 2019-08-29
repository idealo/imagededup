import os
import numpy as np
from pathlib import Path, PosixPath
from typing import Dict, Optional, Union
from keras.models import Model
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.layers import Flatten, MaxPooling2D, AveragePooling2D, concatenate
from imagededup.utils.data_generator import DataGenerator
from imagededup.utils.image_utils import load_image, preprocess_image
from imagededup.utils.logger import return_logger


class CNN:
    """
    Finds duplicates using CNN and/or generates CNN features given a single image or a directory of images.
    The module can be used for 2 purposes: Feature generation and duplicate detection.

    Feature generation:
    To propagate an image through a Convolutional Neural Network architecture and generate features. The generated
    features can be used at a later time for deduplication. There are two possibilities to get features:
    1. At a single image level: Using the method 'encode_image', the CNN feature for a single image can be obtained.
    Example usage:
    ```
    from imagededup import cnn
    mycnn = cnn.CNN()
    feature_vector = mycnn.encode_image('path/to/image.jpg')
    ```
    2. At a directory level: In case features for several images need to be generated, the images can be placed in a
    directory and features for all of the images can be obtained using the 'encode_images' method.
    Example usage:
    ```
    from imagededup import cnn
    mycnn = cnn.CNN()
    feature_vectors = mycnn.encode_imges('path/to/directory')
    ```
    """

    def __init__(self) -> None:
        """
        Initializes a keras MobileNet model that is sliced at the last convolutional layer.
        Sets the batch size for keras generators to be 64 samples. Sets the input image size to (224, 224) for providing
        as input to MobileNet model. Initiates a result_score variable to None.
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
        '''
        Example usage:
        ```
        from imagededup import cnn
        mycnn = cnn.CNN()
        feature_vector = mycnn.encode_image('path/to/image.jpg')
        ```
        '''
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
        '''
        Example usage:
        ```
        from imagededup import cnn
        mycnn = cnn.CNN()
        feature_vectors = mycnn.encode_images('path/to/directory')
        ```
        '''
        if isinstance(image_dir, str):
            image_dir = Path(image_dir)

        if not image_dir.is_dir(image_dir):
            raise ValueError('Please provide a valid directory path!')

        return self._get_cnn_features_batch(image_dir)
