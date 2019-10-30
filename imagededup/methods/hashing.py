import os
import sys
from pathlib import PurePath, Path
from typing import Dict, List, Optional

import pywt
import numpy as np
from scipy.fftpack import dct

from imagededup.handlers.search.retrieval import HashEval
from imagededup.utils.general_utils import get_files_to_remove, save_json, parallelise
from imagededup.utils.image_utils import load_image, preprocess_image
from imagededup.utils.logger import return_logger

logger = return_logger(__name__)

"""
TODO:
Wavelet hash: Zero the LL coeff, reconstruct image, then get wavelet transform
Wavelet hash: Allow possibility of different wavelet functions

"""


class Hashing:
    """
    Find duplicates using hashing algorithms and/or generate hashes given a single image or a directory of images.

    The module can be used for 2 purposes: Encoding generation and duplicate detection.
    - Encoding generation:
    To generate hashes using specific hashing method. The generated hashes can be used at a later time for
    deduplication. Using the method 'encode_image' from the specific hashing method object, the hash for a
    single image can be obtained while the 'encode_images' method can be used to get hashes for all images in a
    directory.

    - Duplicate detection:
    Find duplicates either using the encoding mapping generated previously using 'encode_images' or using a Path to the
    directory that contains the images that need to be deduplicated. 'find_duplciates' and 'find_duplicates_to_remove'
    methods are provided to accomplish these tasks.
    """

    def __init__(self, verbose: bool = True) -> None:
        """
        Initialize hashing class.

        Args:
            verbose: Display progress bar if True else disable it. Default value is True.
        """
        self.target_size = (8, 8)  # resizing to dims
        self.verbose = verbose

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> float:
        """
        Calculate the hamming distance between two hashes. If length of hashes is not 64 bits, then pads the length
        to be 64 for each hash and then calculates the hamming distance.

        Args:
            hash1: hash string
            hash2: hash string

        Returns:
            hamming_distance: Hamming distance between the two hashes.
        """
        hash1_bin = bin(int(hash1, 16))[2:].zfill(
            64
        )  # zfill ensures that len of hash is 64 and pads MSB if it is < A
        hash2_bin = bin(int(hash2, 16))[2:].zfill(64)
        return np.sum([i != j for i, j in zip(hash1_bin, hash2_bin)])

    @staticmethod
    def _array_to_hash(hash_mat: np.ndarray) -> str:
        """
        Convert a matrix of binary numerals to 64 character hash.

        Args:
            hash_mat: A numpy array consisting of 0/1 values.

        Returns:
            An hexadecimal hash string.
        """
        return ''.join('%0.2x' % x for x in np.packbits(hash_mat))

    def encode_image(
        self, image_file=None, image_array: Optional[np.ndarray] = None
    ) -> str:
        """
        Generate hash for a single image.

        Args:
            image_file: Path to the image file.
            image_array: Optional, used instead of image_file. Image typecast to numpy array.

        Returns:
            hash: A 16 character hexadecimal string hash for the image.

        Example:
        ```
        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        myhash = myencoder.encode_image(image_file='path/to/image.jpg')
        OR
        myhash = myencoder.encode_image(image_array=<numpy array of image>)
        ```
        """
        try:
            if image_file and os.path.exists(image_file):
                image_file = Path(image_file)
                image_pp = load_image(
                    image_file=image_file, target_size=self.target_size, grayscale=True
                )

            elif isinstance(image_array, np.ndarray):
                image_pp = preprocess_image(
                    image=image_array, target_size=self.target_size, grayscale=True
                )
            else:
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError('Please provide either image file path or image array!')

        return self._hash_func(image_pp) if isinstance(image_pp, np.ndarray) else None

    def encode_images(self, image_dir=None):
        """
        Generate hashes for all images in a given directory of images.

        Args:
            image_dir: Path to the image directory.

        Returns:
            dictionary: A dictionary that contains a mapping of filenames and corresponding 64 character hash string
                        such as {'Image1.jpg': 'hash_string1', 'Image2.jpg': 'hash_string2', ...}

        Example:
        ```
        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        mapping = myencoder.encode_images('path/to/directory')
        ```
        """
        if not os.path.isdir(image_dir):
            raise ValueError('Please provide a valid directory path!')

        image_dir = Path(image_dir)

        files = [
            i.absolute() for i in image_dir.glob('*') if not i.name.startswith('.')
        ]  # ignore hidden files

        logger.info(f'Start: Calculating hashes...')

        hashes = parallelise(self.encode_image, files, self.verbose)
        hash_initial_dict = dict(zip([f.name for f in files], hashes))
        hash_dict = {
            k: v for k, v in hash_initial_dict.items() if v
        }  # To ignore None (returned if some probelm with image file)

        logger.info(f'End: Calculating hashes!')
        return hash_dict

    def _hash_algo(self, image_array: np.ndarray):
        pass

    def _hash_func(self, image_array: np.ndarray):
        hash_mat = self._hash_algo(image_array)
        return self._array_to_hash(hash_mat)

    # search part

    @staticmethod
    def _check_hamming_distance_bounds(thresh: int) -> None:
        """
        Check if provided threshold is valid. Raises TypeError if wrong threshold variable type is passed or a
        ValueError if an out of range value is supplied.

        Args:
            thresh: Threshold value (must be int between 0 and 64)

        Raises:
            TypeError: If wrong variable type is provided.
            ValueError: If invalid value is provided.
        """
        if not isinstance(thresh, int):
            raise TypeError('Threshold must be an int between 0 and 64')
        elif thresh < 0 or thresh > 64:
            raise ValueError('Threshold must be an int between 0 and 64')
        else:
            return None

    def _find_duplicates_dict(
        self,
        encoding_map: Dict[str, str],
        max_distance_threshold: int = 10,
        scores: bool = False,
        outfile: Optional[str] = None,
        search_method: str = 'brute_force_cython' if not sys.platform == 'win32' else 'bktree',
    ) -> Dict:
        """
        Take in dictionary {filename: encoded image}, detects duplicates below the given hamming distance threshold
        and returns a dictionary containing key as filename and value as a list of duplicate filenames. Optionally,
        the hamming distances could be returned instead of just duplicate filenames for each query file.

        Args:
            encoding_map: Dictionary with keys as file names and values as encoded images (hashes).
            max_distance_threshold: Hamming distance between two images below which retrieved duplicates are valid.
            scores: Boolean indicating whether hamming distance scores are to be returned along with retrieved
            duplicates.
            outfile: Optional, name of the file to save the results. Default is None.
            search_method: Algorithm used to retrieve duplicates. Default is brute_force_cython for Unix else bktree.

        Returns:
            if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
            score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}
            if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg',
            'image1_duplicate2.jpg'], 'image2.jpg':['image1_duplicate1.jpg',..], ..}
        """
        logger.info('Start: Evaluating hamming distances for getting duplicates')

        result_set = HashEval(
            test=encoding_map,
            queries=encoding_map,
            distance_function=self.hamming_distance,
            verbose=self.verbose,
            threshold=max_distance_threshold,
            search_method=search_method,
        )

        logger.info('End: Evaluating hamming distances for getting duplicates')

        self.results = result_set.retrieve_results(scores=scores)
        if outfile:
            save_json(self.results, outfile)
        return self.results

    def _find_duplicates_dir(
        self,
        image_dir: PurePath,
        max_distance_threshold: int = 10,
        scores: bool = False,
        outfile: Optional[str] = None,
        search_method: str = 'brute_force_cython' if not sys.platform == 'win32' else 'bktree',
    ) -> Dict:
        """
        Take in path of the directory in which duplicates are to be detected below the given hamming distance
        threshold. Returns dictionary containing key as filename and value as a list of duplicate file names.
        Optionally, the hamming distances could be returned instead of just duplicate filenames for each query file.

        Args:
            image_dir: Path to the directory containing all the images.
            max_distance_threshold: Hamming distance between two images below which retrieved duplicates are valid.
            scores: Boolean indicating whether Hamming distances are to be returned along with retrieved duplicates.
            outfile: Name of the file the results should be written to.
            search_method: Algorithm used to retrieve duplicates. Default is brute_force_cython for Unix else bktree.

        Returns:
            if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
            score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}
            if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg',
            'image1_duplicate2.jpg'], 'image2.jpg':['image1_duplicate1.jpg',..], ..}
        """
        encoding_map = self.encode_images(image_dir)
        results = self._find_duplicates_dict(
            encoding_map=encoding_map,
            max_distance_threshold=max_distance_threshold,
            scores=scores,
            outfile=outfile,
            search_method=search_method,
        )
        return results

    def find_duplicates(
        self,
        image_dir: PurePath = None,
        encoding_map: Dict[str, str] = None,
        max_distance_threshold: int = 10,
        scores: bool = False,
        outfile: Optional[str] = None,
        search_method: str = 'brute_force_cython' if not sys.platform == 'win32' else 'bktree',
    ) -> Dict:
        """
        Find duplicates for each file. Takes in path of the directory or encoding dictionary in which duplicates are to
        be detected. All images with hamming distance less than or equal to the max_distance_threshold are regarded as
        duplicates. Returns dictionary containing key as filename and value as a list of duplicate file names.
        Optionally, the below the given hamming distance could be returned instead of just duplicate filenames for each
        query file.

        Args:
            image_dir: Path to the directory containing all the images or dictionary with keys as file names
                       and values as hash strings for the key image file.
            encoding_map: Optional,  used instead of image_dir, a dictionary containing mapping of filenames and
                          corresponding hashes.
            max_distance_threshold: Optional, hamming distance between two images below which retrieved duplicates are
                                    valid. (must be an int between 0 and 64). Default is 10.
            scores: Optional, boolean indicating whether Hamming distances are to be returned along with retrieved duplicates.
            outfile: Optional, name of the file to save the results, must be a json. Default is None.
            search_method: Algorithm used to retrieve duplicates. Default is brute_force_cython for Unix else bktree.

        Returns:
            duplicates dictionary: if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
                        score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}. if scores is False, then a
                        dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg', 'image1_duplicate2.jpg'],
                        'image2.jpg':['image1_duplicate1.jpg',..], ..}

        Example:
        ```
        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        duplicates = myencoder.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True,
        outfile='results.json')

        OR

        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        duplicates = myencoder.find_duplicates(encoding_map=<mapping filename to hashes>,
        max_distance_threshold=15, scores=True, outfile='results.json')
        ```
        """
        self._check_hamming_distance_bounds(thresh=max_distance_threshold)
        if image_dir:
            result = self._find_duplicates_dir(
                image_dir=image_dir,
                max_distance_threshold=max_distance_threshold,
                scores=scores,
                outfile=outfile,
                search_method=search_method,
            )
        elif encoding_map:
            result = self._find_duplicates_dict(
                encoding_map=encoding_map,
                max_distance_threshold=max_distance_threshold,
                scores=scores,
                outfile=outfile,
                search_method=search_method,
            )
        else:
            raise ValueError('Provide either an image directory or encodings!')
        return result

    def find_duplicates_to_remove(
        self,
        image_dir: PurePath = None,
        encoding_map: Dict[str, str] = None,
        max_distance_threshold: int = 10,
        outfile: Optional[str] = None,
    ) -> List:
        """
        Give out a list of image file names to remove based on the hamming distance threshold threshold. Does not
        remove the mentioned files.

        Args:
            image_dir: Path to the directory containing all the images or dictionary with keys as file names
                       and values as hash strings for the key image file.
            encoding_map: Optional, used instead of image_dir, a dictionary containing mapping of filenames and
                          corresponding hashes.
            max_distance_threshold: Optional, hamming distance between two images below which retrieved duplicates are
                                    valid. (must be an int between 0 and 64). Default is 10.
            outfile: Optional, name of the file to save the results, must be a json. Default is None.

        Returns:
            duplicates: List of image file names that are found to be duplicate of me other file in the directory.

        Example:
        ```
        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        duplicates = myencoder.find_duplicates_to_remove(image_dir='path/to/images/directory'),
        max_distance_threshold=15)

        OR

        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        duplicates = myencoder.find_duplicates(encoding_map=<mapping filename to hashes>,
        max_distance_threshold=15, outfile='results.json')
        ```
        """
        result = self.find_duplicates(
            image_dir=image_dir,
            encoding_map=encoding_map,
            max_distance_threshold=max_distance_threshold,
            scores=False,
        )
        files_to_remove = get_files_to_remove(result)
        if outfile:
            save_json(files_to_remove, outfile)
        return files_to_remove


class PHash(Hashing):
    """
    Inherits from Hashing base class and implements perceptual hashing (Implementation reference:
    http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html).

    Offers all the functionality mentioned in hashing class.

    Example:
    ```
    # Perceptual hash for images
    from imagededup.methods import PHash
    phasher = PHash()
    perceptual_hash = phasher.encode_image(image_file = 'path/to/image.jpg')
    OR
    perceptual_hash = phasher.encode_image(image_array = <numpy image array>)
    OR
    perceptual_hashes = phasher.encode_images(image_dir = 'path/to/directory')  # for a directory of images

    # Finding duplicates:
    from imagededup.methods import PHash
    phasher = PHash()
    duplicates = phasher.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True)
    OR
    duplicates = phasher.find_duplicates(encoding_map=encoding_map, max_distance_threshold=15, scores=True)

    # Finding duplicates to return a single list of duplicates in the image collection
    from imagededup.methods import PHash
    phasher = PHash()
    files_to_remove = phasher.find_duplicates_to_remove(image_dir='path/to/images/directory',
                      max_distance_threshold=15)
    OR
    files_to_remove = phasher.find_duplicates_to_remove(encoding_map=encoding_map, max_distance_threshold=15)
    ```
    """

    def __init__(self, verbose: bool = True) -> None:
        """
        Initialize perceptual hashing class.

        Args:
            verbose: Display progress bar if True else disable it. Default value is True.
        """
        super().__init__(verbose)
        self.__coefficient_extract = (8, 8)
        self.target_size = (32, 32)

    def _hash_algo(self, image_array):
        """
        Get perceptual hash of the input image.

        Args:
            image_array: numpy array that corresponds to the image.

        Returns:
            A string representing the perceptual hash of the image.
        """
        dct_coef = dct(dct(image_array, axis=0), axis=1)

        # retain top left 8 by 8 dct coefficients
        dct_reduced_coef = dct_coef[
            : self.__coefficient_extract[0], : self.__coefficient_extract[1]
        ]

        # median of coefficients excluding the DC term (0th term)
        # mean_coef_val = np.mean(np.ndarray.flatten(dct_reduced_coef)[1:])
        median_coef_val = np.median(np.ndarray.flatten(dct_reduced_coef)[1:])

        # return mask of all coefficients greater than mean of coefficients
        hash_mat = dct_reduced_coef >= median_coef_val
        return hash_mat


class AHash(Hashing):
    """
    Inherits from Hashing base class and implements average hashing. (Implementation reference:
    http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html)

    Offers all the functionality mentioned in hashing class.

    Example:
    ```
    # Average hash for images
    from imagededup.methods import AHash
    ahasher = AHash()
    average_hash = ahasher.encode_image(image_file = 'path/to/image.jpg')
    OR
    average_hash = ahasher.encode_image(image_array = <numpy image array>)
    OR
    average_hashes = ahasher.encode_images(image_dir = 'path/to/directory')  # for a directory of images

    # Finding duplicates:
    from imagededup.methods import AHash
    ahasher = AHash()
    duplicates = ahasher.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True)
    OR
    duplicates = ahasher.find_duplicates(encoding_map=encoding_map, max_distance_threshold=15, scores=True)

    # Finding duplicates to return a single list of duplicates in the image collection
    from imagededup.methods import AHash
    ahasher = AHash()
    files_to_remove = ahasher.find_duplicates_to_remove(image_dir='path/to/images/directory',
                      max_distance_threshold=15)
    OR
    files_to_remove = ahasher.find_duplicates_to_remove(encoding_map=encoding_map, max_distance_threshold=15)
    ```
    """

    def __init__(self, verbose: bool = True) -> None:
        """
        Initialize average hashing class.

        Args:
            verbose: Display progress bar if True else disable it. Default value is True.
        """
        super().__init__(verbose)
        self.target_size = (8, 8)

    def _hash_algo(self, image_array: np.ndarray):
        """
        Get average hash of the input image.

        Args:
            image_array: numpy array that corresponds to the image.

        Returns:
            A string representing the average hash of the image.
        """
        avg_val = np.mean(image_array)
        hash_mat = image_array >= avg_val
        return hash_mat


class DHash(Hashing):
    """
    Inherits from Hashing base class and implements difference hashing. (Implementation reference:
    http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html)

    Offers all the functionality mentioned in hashing class.

    Example:
    ```
    # Difference hash for images
    from imagededup.methods import DHash
    dhasher = DHash()
    difference_hash = dhasher.encode_image(image_file = 'path/to/image.jpg')
    OR
    difference_hash = dhasher.encode_image(image_array = <numpy image array>)
    OR
    difference_hashes = dhasher.encode_images(image_dir = 'path/to/directory')  # for a directory of images

    # Finding duplicates:
    from imagededup.methods import DHash
    dhasher = DHash()
    duplicates = dhasher.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True)
    OR
    duplicates = dhasher.find_duplicates(encoding_map=encoding_map, max_distance_threshold=15, scores=True)

    # Finding duplicates to return a single list of duplicates in the image collection
    from imagededup.methods import DHash
    dhasher = DHash()
    files_to_remove = dhasher.find_duplicates_to_remove(image_dir='path/to/images/directory',
                      max_distance_threshold=15)
    OR
    files_to_remove = dhasher.find_duplicates_to_remove(encoding_map=encoding_map, max_distance_threshold=15)
    ```
    """

    def __init__(self, verbose: bool = True) -> None:
        """
        Initialize difference hashing class.

        Args:
            verbose: Display progress bar if True else disable it. Default value is True.
        """
        super().__init__(verbose)
        self.target_size = (9, 8)

    def _hash_algo(self, image_array):
        """
        Get difference hash of the input image.

        Args:
            image_array: numpy array that corresponds to the image.

        Returns:
            A string representing the difference hash of the image.
        """
        # Calculates difference between consecutive columns and return mask
        hash_mat = image_array[:, 1:] > image_array[:, :-1]
        return hash_mat


class WHash(Hashing):
    """
    Inherits from Hashing base class and implements wavelet hashing. (Implementation reference:
    https://fullstackml.com/wavelet-image-hash-in-python-3504fdd282b5)

    Offers all the functionality mentioned in hashing class.

    Example:
    ```
    # Wavelet hash for images
    from imagededup.methods import WHash
    whasher = WHash()
    wavelet_hash = whasher.encode_image(image_file = 'path/to/image.jpg')
    OR
    wavelet_hash = whasher.encode_image(image_array = <numpy image array>)
    OR
    wavelet_hashes = whasher.encode_images(image_dir = 'path/to/directory')  # for a directory of images

    # Finding duplicates:
    from imagededup.methods import WHash
    whasher = WHash()
    duplicates = whasher.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True)
    OR
    duplicates = whasher.find_duplicates(encoding_map=encoding_map, max_distance_threshold=15, scores=True)

    # Finding duplicates to return a single list of duplicates in the image collection
    from imagededup.methods import WHash
    whasher = WHash()
    files_to_remove = whasher.find_duplicates_to_remove(image_dir='path/to/images/directory',
                      max_distance_threshold=15)
    OR
    files_to_remove = whasher.find_duplicates_to_remove(encoding_map=encoding_map, max_distance_threshold=15)
    ```
    """

    def __init__(self, verbose: bool = True) -> None:
        """
        Initialize wavelet hashing class.

        Args:
            verbose: Display progress bar if True else disable it. Default value is True.
        """
        super().__init__(verbose)
        self.target_size = (256, 256)
        self.__wavelet_func = 'haar'

    def _hash_algo(self, image_array):
        """
        Get wavelet hash of the input image.

        Args:
            image_array: numpy array that corresponds to the image.

        Returns:
            A string representing the wavelet hash of the image.
        """
        # decomposition level set to 5 to get 8 by 8 hash matrix
        image_array = image_array / 255
        coeffs = pywt.wavedec2(data=image_array, wavelet=self.__wavelet_func, level=5)
        LL_coeff = coeffs[0]

        # median of LL coefficients
        median_coef_val = np.median(np.ndarray.flatten(LL_coeff))

        # return mask of all coefficients greater than mean of coefficients
        hash_mat = LL_coeff >= median_coef_val
        return hash_mat
