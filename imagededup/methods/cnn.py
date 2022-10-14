from pathlib import Path, PurePath
from typing import Dict, List, Optional, Union
import warnings

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import transforms

from imagededup.handlers.search.retrieval import get_cosine_similarity
from imagededup.utils.data_generator import img_dataloader, MobilenetV3
from imagededup.utils.general_utils import (
    generate_relative_names,
    get_files_to_remove,
    save_json,
)
from imagededup.utils.image_utils import (
    expand_image_array_cnn,
    load_image,
    preprocess_image,
)
from imagededup.utils.logger import return_logger


class CNN:
    """
    Find duplicates using CNN and/or generate CNN encodings given a single image or a directory of images.

    The module can be used for 2 purposes: Encoding generation and duplicate detection.
    - Encodings generation:
    To propagate an image through a Convolutional Neural Network architecture and generate encodings. The generated
    encodings can be used at a later time for deduplication. Using the method 'encode_image', the CNN encodings for a
    single image can be obtained while the 'encode_images' method can be used to get encodings for all images in a
    directory.

    - Duplicate detection:
    Find duplicates either using the encoding mapping generated previously using 'encode_images' or using a Path to the
    directory that contains the images that need to be deduplicated. 'find_duplciates' and 'find_duplicates_to_remove'
    methods are provided to accomplish these tasks.
    """

    def __init__(self, verbose: bool = True) -> None:
        """
        Initialize a pytorch MobileNet model v3 that is sliced at the last convolutional layer.
        Set the batch size for pytorch dataloader to be 64 samples.

        Args:
            verbose: Display progress bar if True else disable it. Default value is True.
        """
        self.target_size = (256, 256)
        self.batch_size = 64
        self.logger = return_logger(
            __name__
        )  # The logger needs to be bound to the class, otherwise stderr also gets
        # directed to stdout (Don't know why that is the case)
        self._build_model()
        self.verbose = 1 if verbose is True else 0

    def _build_model(self):
        """
        Build MobileNet v3 model sliced at the last convolutional layer with global average pooling added. Also initialize the corresponding preprocessing transform.
        """
        self.model = MobilenetV3()
        self.logger.info(
            'Initialized: MobileNet v3 pretrained on ImageNet dataset sliced at GAP layer'
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.target_size),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def apply_mobilenet_preprocess(self, im_arr: np.array) -> torch.tensor:
        image_pil = Image.fromarray(im_arr)
        return self.transform(image_pil)

    def _get_cnn_features_single(self, image_array: np.ndarray) -> np.ndarray:
        """
        Generate CNN encodings for a single image.

        Args:
            image_array: Image typecast to numpy array.

        Returns:
            Encodings for the image in the form of numpy array.
        """
        image_pp = self.apply_mobilenet_preprocess(image_array)
        image_pp = image_pp.unsqueeze(0)
        img_features_tensor = self.model(image_pp)
        return img_features_tensor.detach().numpy()[..., 0, 0]

    def _get_cnn_features_batch(
        self, image_dir: PurePath, recursive: Optional[bool] = False
    ) -> Dict[str, np.ndarray]:
        """
        Generate CNN encodings for all images in a given directory of images.
        Args:
            image_dir: Path to the image directory.
            recursive: Optional, find images recursively in the image directory.

        Returns:
            A dictionary that contains a mapping of filenames and corresponding numpy array of CNN encodings.
        """
        self.logger.info('Start: Image encoding generation')
        self.dataloader = img_dataloader(
            image_dir=image_dir,
            batch_size=self.batch_size,
            basenet_preprocess=self.apply_mobilenet_preprocess,
            recursive=recursive,
        )

        feat_arr, all_filenames = [], []
        bad_im_count = 0

        for ims, filenames, bad_images in self.dataloader:
            arr = self.model(ims)
            feat_arr.extend(arr)
            all_filenames.extend(filenames)
            if bad_images:
                bad_im_count += 1

        if bad_im_count:
            self.logger.info(
                f'Found {bad_im_count} bad images, ignoring for encoding generation ..'
            )

        feat_vec = torch.stack(feat_arr).squeeze().detach().numpy()
        valid_image_files = [filename for filename in all_filenames if filename]
        self.logger.info('End: Image encoding generation')

        filenames = generate_relative_names(image_dir, valid_image_files)
        if len(feat_vec.shape) == 1:  # can happen when encode_images is called on a directory containing a single image
            self.encoding_map = {filenames[0]: feat_vec}
        else:
            self.encoding_map = {j: feat_vec[i, :] for i, j in enumerate(filenames)}
        return self.encoding_map

    def encode_image(
        self,
        image_file: Optional[Union[PurePath, str]] = None,
        image_array: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate CNN encoding for a single image.

        Args:
            image_file: Path to the image file.
            image_array: Optional, used instead of image_file. Image typecast to numpy array.

        Returns:
            encoding: Encodings for the image in the form of numpy array.

        Example:
        ```
        from imagededup.methods import CNN
        myencoder = CNN()
        encoding = myencoder.encode_image(image_file='path/to/image.jpg')
        OR
        encoding = myencoder.encode_image(image_array=<numpy array of image>)
        ```
        """
        if isinstance(image_file, str):
            image_file = Path(image_file)

        if isinstance(image_file, PurePath):
            if not image_file.is_file():
                raise ValueError(
                    'Please provide either image file path or image array!'
                )

            image_pp = load_image(
                image_file=image_file, target_size=None, grayscale=False
            )

        elif isinstance(image_array, np.ndarray):
            image_array = expand_image_array_cnn(
                image_array
            )  # Add 3rd dimension if array is grayscale, do sanity checks
            image_pp = preprocess_image(
                image=image_array, target_size=None, grayscale=False
            )
        else:
            raise ValueError('Please provide either image file path or image array!')

        return (
            self._get_cnn_features_single(image_pp)
            if isinstance(image_pp, np.ndarray)
            else None
        )

    def encode_images(
        self, image_dir: Union[PurePath, str], recursive: Optional[bool] = False
    ) -> Dict:
        """Generate CNN encodings for all images in a given directory of images.

        Args:
            image_dir: Path to the image directory.
            recursive: Optional, find images recursively in the image directory.
        Returns:
            dictionary: Contains a mapping of filenames and corresponding numpy array of CNN encodings.
        Example:
            ```
            from imagededup.methods import CNN
            myencoder = CNN()
            encoding_map = myencoder.encode_images(image_dir='path/to/image/directory')
            ```
        """
        if isinstance(image_dir, str):
            image_dir = Path(image_dir)

        if not image_dir.is_dir():
            raise ValueError('Please provide a valid directory path!')

        return self._get_cnn_features_batch(image_dir, recursive)

    @staticmethod
    def _check_threshold_bounds(thresh: float) -> None:
        """
        Check if provided threshold is valid. Raises TypeError if wrong threshold variable type is passed or a
        ValueError if an out of range value is supplied.

        Args:
            thresh: Threshold value (must be float between -1.0 and 1.0)

        Raises:
            TypeError: If wrong variable type is provided.
            ValueError: If wrong value is provided.
        """
        if not isinstance(thresh, float):
            raise TypeError('Threshold must be a float between -1.0 and 1.0')
        if thresh < -1.0 or thresh > 1.0:
            raise ValueError('Threshold must be a float between -1.0 and 1.0')

    def _find_duplicates_dict(
        self,
        encoding_map: Dict[str, list],
        min_similarity_threshold: float,
        scores: bool,
        outfile: Optional[str] = None,
    ) -> Dict:
        """
        Take in dictionary {filename: encoded image}, detects duplicates above the given cosine similarity threshold
        and returns a dictionary containing key as filename and value as a list of duplicate filenames. Optionally,
        the cosine distances could be returned instead of just duplicate filenames for each query file.

        Args:
            encoding_map: Dictionary with keys as file names and values as encoded images.
            min_similarity_threshold: Cosine similarity above which retrieved duplicates are valid.
            scores: Boolean indicating whether similarity scores are to be returned along with retrieved duplicates.

        Returns:
            if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
            score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}
            if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg',
            'image1_duplicate2.jpg'], 'image2.jpg':['image1_duplicate1.jpg',..], ..}
        """

        # get all image ids
        # we rely on dictionaries preserving insertion order in Python >=3.6
        image_ids = np.array([*encoding_map.keys()])

        # put image encodings into feature matrix
        features = np.array([*encoding_map.values()])

        self.logger.info('Start: Calculating cosine similarities...')

        self.cosine_scores = get_cosine_similarity(features, self.verbose)

        np.fill_diagonal(
            self.cosine_scores, 2.0
        )  # allows to filter diagonal in results, 2 is a placeholder value

        self.logger.info('End: Calculating cosine similarities.')

        self.results = {}
        for i, j in enumerate(self.cosine_scores):
            duplicates_bool = (j >= min_similarity_threshold) & (j < 2)

            if scores:
                tmp = np.array([*zip(image_ids, j)], dtype=object)
                duplicates = list(map(tuple, tmp[duplicates_bool]))

            else:
                duplicates = list(image_ids[duplicates_bool])

            self.results[image_ids[i]] = duplicates

        if outfile and scores:
            save_json(results=self.results, filename=outfile, float_scores=True)
        elif outfile:
            save_json(results=self.results, filename=outfile)
        return self.results

    def _find_duplicates_dir(
        self,
        image_dir: Union[PurePath, str],
        min_similarity_threshold: float,
        scores: bool,
        outfile: Optional[str] = None,
        recursive: Optional[bool] = False,
    ) -> Dict:
        """
        Take in path of the directory in which duplicates are to be detected above the given threshold.
        Returns dictionary containing key as filename and value as a list of duplicate file names.  Optionally,
        the cosine distances could be returned instead of just duplicate filenames for each query file.

        Args:
            image_dir: Path to the directory containing all the images.
            min_similarity_threshold: Optional, hamming distance above which retrieved duplicates are valid. Default 0.9
            scores: Optional, boolean indicating whether Hamming distances are to be returned along with retrieved
                    duplicates.
            outfile: Optional, name of the file the results should be written to.
            recursive: Optional, find images recursively in the image directory.

        Returns:
            if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
            score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}
            if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg',
            'image1_duplicate2.jpg'], 'image2.jpg':['image1_duplicate1.jpg',..], ..}
        """
        self.encode_images(image_dir=image_dir, recursive=recursive)

        return self._find_duplicates_dict(
            encoding_map=self.encoding_map,
            min_similarity_threshold=min_similarity_threshold,
            scores=scores,
            outfile=outfile,
        )

    def find_duplicates(
        self,
        image_dir: Union[PurePath, str] = None,
        encoding_map: Dict[str, list] = None,
        min_similarity_threshold: float = 0.9,
        scores: bool = False,
        outfile: Optional[str] = None,
        recursive: Optional[bool] = False,
    ) -> Dict:
        """
        Find duplicates for each file. Take in path of the directory or encoding dictionary in which duplicates are to
        be detected above the given threshold. Return dictionary containing key as filename and value as a list of
        duplicate file names. Optionally, the cosine distances could be returned instead of just duplicate filenames for
        each query file.

        Args:
            image_dir: Path to the directory containing all the images or dictionary with keys as file names
            and values as numpy arrays which represent the CNN encoding for the key image file.
            encoding_map: Optional, used instead of image_dir, a dictionary containing mapping of filenames and
                          corresponding CNN encodings.
            min_similarity_threshold: Optional, threshold value (must be float between -1.0 and 1.0). Default is 0.9
            scores: Optional, boolean indicating whether similarity scores are to be returned along with retrieved
                    duplicates.
            outfile: Optional, name of the file to save the results, must be a json. Default is None.
            recursive: Optional, find images recursively in the image directory.

        Returns:
            dictionary: if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
                        score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}. if scores is False, then a
                        dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg', 'image1_duplicate2.jpg'],
                        'image2.jpg':['image1_duplicate1.jpg',..], ..}

        Example:
        ```
        from imagededup.methods import CNN
        myencoder = CNN()
        duplicates = myencoder.find_duplicates(image_dir='path/to/directory', min_similarity_threshold=0.85, scores=True,
        outfile='results.json')

        OR

        from imagededup.methods import CNN
        myencoder = CNN()
        duplicates = myencoder.find_duplicates(encoding_map=<mapping filename to cnn encodings>,
        min_similarity_threshold=0.85, scores=True, outfile='results.json')
        ```
        """
        self._check_threshold_bounds(min_similarity_threshold)

        if image_dir:
            result = self._find_duplicates_dir(
                image_dir=image_dir,
                min_similarity_threshold=min_similarity_threshold,
                scores=scores,
                outfile=outfile,
                recursive=recursive,
            )
        elif encoding_map:
            if recursive:
                warnings.warn(
                    'recursive parameter is irrelevant when using encodings.',
                    SyntaxWarning,
                )
            result = self._find_duplicates_dict(
                encoding_map=encoding_map,
                min_similarity_threshold=min_similarity_threshold,
                scores=scores,
                outfile=outfile,
            )

        else:
            raise ValueError('Provide either an image directory or encodings!')

        return result

    def find_duplicates_to_remove(
        self,
        image_dir: PurePath = None,
        encoding_map: Dict[str, np.ndarray] = None,
        min_similarity_threshold: float = 0.9,
        outfile: Optional[str] = None,
        recursive: Optional[bool] = False,
    ) -> List:
        """
        Give out a list of image file names to remove based on the similarity threshold. Does not remove the mentioned
        files.

        Args:
            image_dir: Path to the directory containing all the images or dictionary with keys as file names
                       and values as numpy arrays which represent the CNN encoding for the key image file.
            encoding_map: Optional, used instead of image_dir, a dictionary containing mapping of filenames and
                          corresponding CNN encodings.
            min_similarity_threshold: Optional, threshold value (must be float between -1.0 and 1.0). Default is 0.9
            outfile: Optional, name of the file to save the results, must be a json. Default is None.
            recursive: Optional, find images recursively in the image directory.

        Returns:
            duplicates: List of image file names that should be removed.

        Example:
        ```
        from imagededup.methods import CNN
        myencoder = CNN()
        duplicates = myencoder.find_duplicates_to_remove(image_dir='path/to/images/directory'),
        min_similarity_threshold=0.85)

        OR

        from imagededup.methods import CNN
        myencoder = CNN()
        duplicates = myencoder.find_duplicates_to_remove(encoding_map=<mapping filename to cnn encodings>,
        min_similarity_threshold=0.85, outfile='results.json')
        ```
        """
        if image_dir or encoding_map:
            duplicates = self.find_duplicates(
                image_dir=image_dir,
                encoding_map=encoding_map,
                min_similarity_threshold=min_similarity_threshold,
                scores=False,
                recursive=recursive,
            )

        files_to_remove = get_files_to_remove(duplicates)

        if outfile:
            save_json(files_to_remove, outfile)

        return files_to_remove
