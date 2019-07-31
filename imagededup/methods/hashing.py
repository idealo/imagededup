from imagededup.handlers.search.retrieval import HashEval
from imagededup.utils.logger import return_logger
from imagededup.utils.image_utils import check_directory_files, convert_to_array
from imagededup.utils.general_utils import get_files_to_remove
import os
import pywt
import scipy.fftpack
import numpy as np
from PIL import Image
from pathlib import Path, PosixPath
from types import FunctionType
from typing import Dict, List
from copy import deepcopy

"""
TODO:
Wavelet hash: Zero the LL coeff, reconstruct image, then get wavelet transform
Wavelet hash: Allow possibility of different wavelet functions

"""


class Hashing:
    def __init__(self) -> None:
        self.logger = return_logger(__name__, os.getcwd())

    @staticmethod
    def bool_to_hex(x: np.array) -> str:
        str_bool = ''.join([str(int(i)) for i in x])
        int_base2 = int(str_bool, 2)  # int base 2
        return '{:0x}'.format(int_base2)

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> float:
        """
        Calculates the hamming distance between two hashes. If length of hashes is not 64 bits, then pads the length
        to be 64 for each hash and then calculates the hamming distance.
        :param hash1: hash string
        :param hash2: hash string
        :return: Hamming distance between the two hashes.
        """
        hash1_bin = bin(int(hash1, 16))[2:].zfill(64)  # zfill ensures that len of hash is 64 and pads MSB if it is < A
        hash2_bin = bin(int(hash2, 16))[2:].zfill(64)
        return np.sum([i != j for i, j in zip(hash1_bin, hash2_bin)])

    def _array_to_hash(self, hash_mat: np.array, n_blocks: int) -> str:
        """
        Convert a matrix of binary numerals to an n_blocks length hash.
        :param hash_mat: A numpy array consisting of 0/1 values.
        :param n_blocks: Hash length required.
        :return: An n_blocks length hash string.
        """
        calculated_hash = []
        for i in np.array_split(np.ndarray.flatten(hash_mat), n_blocks):
            calculated_hash.append(self.bool_to_hex(i))
        return ''.join(calculated_hash)

    def hash_image(self, path_image: PosixPath) -> str:
        """
        Apply a hashing function on the input image.
        :param path_image: A PosixPath to image or a numpy array that corresponds to the image.
        :return: A string representing the hash of the image.
        """
        return self.hash_func(path_image)

    def hash_images(self, image_dir):
        images = load_images(image_dir)

        for image in images:

    def hash_dir(self, path_dir: PosixPath) -> Dict:
        """
        Apply the hashing method to all images in a directory.
        :param path_dir: PosixPath to the directory containing images for which hashes are to be obtained.
        :return: Dictionary containing file names as keys and corresponding hash string as value.
        """

        filenames = check_directory_files(path_dir, return_file=True)
        self.logger.info(f'Start: Calculating hashes using {self.method}!')
        hash_dict = dict()
        for i in filenames:
            hash_dict[i.name] = self.hash_func(Path(i))
        self.logger.info(f'End: Calculating hashes using {self.method}!')
        return hash_dict  # dict_file_feature from _find_duplicates_dict input


    def encode_image(self, image):
        return hash_image(image)


    def encode_images(self, images):
        return hash_images(images)











class Hashing2:
    """
    Finds duplicates using hashing methods and/or generates hashes given a single image or a directory of images.
    The module can be used for 2 purposes: Feature generation and duplicate detection.
    Feature generation:
    To use a hashing method to generate hash for an image or a directory of images. The generated
    hashes can be used at a later time for deduplication. There are two possibilities to get hashes:
    1. At a single image level: Using the function for particular hashing method, the hash for a single image can be
    obtained. There are 4 methods to be chosen from by passing the appropriate method string during object instantiation:
        a. Perceptual hash: 'phash'
        b. Wavelet hash: 'whash'
        c. Difference hash: 'dhash'
        d. Average hash: 'ahash'

    Example usage:
        ```
        from imagededup import hashing
        myhasher = hashing.Hashing(method='phash')
        perceptual_hash_string = myhasher.hash_image(Path('path/to/image.jpg'))
        ```
    2. At a directory level: In case hashes for several images need to be generated, the images can be placed in a
    directory and hashes for all of the images can be obtained using the hash_dir function:

    Example usage:
        ```
        from imagededup import hashing
        myhasher = hashing.Hashing(method='phash')
        dict_file_hash = myhasher.hash_dir(Path('path/to/directory'))
        ```

    Duplicate detection:
    Find duplicates either using the hashes generated previously(dict_file_hash) using directory functions or using a
    Path to the directory that contains the images that need to be deduplicated. There are 2 inputs that can be provided
     to the find_duplicates function:
    1. Dictionary generated using directory function(hash_dir) above.
    Example usage:
        ```
        from imagededup import hashing
        myhasher = hashing.Hashing(method='phash')
        dict_ret_with_dict_inp = myhasher.find_duplicates(dict_file_feat, threshold=15, scores=True)
        ```
    2. Using the Path of the directory where all images are present.
    Example usage:
        ```
        from imagededup import hashing
        myhasher = hashing.Hashing(method='phash')
        dict_ret_path = myhasher.find_duplicates(Path('path/to/directory'), threshold=15, scores=True)
        ```
    If a list of file names to remove are desired, then the function find_duplicates_to_remove can be used with either
    the path to the image directory as input or the dictionary with hashes. A threshold for distance should be
    considered.
    Example usage:
        ```
        from imagededup import hashing
        myhasher = hashing.Hashing(method='phash')
        list_of_files_to_remove = myhasher.find_duplicates_to_remove(Path('path/to/images/directory'), threshold=15)
        ```
        """

    def __init__(self, method: str = 'phash') -> None:
        """
        Initializes the hashing method that needs to be applied.
        here are 4 methods to be chosen from by passing the appropriate method string during object instantiation:
        a. Perceptual hash: 'phash'
        b. Wavelet hash: 'whash'
        c. Difference hash: 'dhash'
        d. Average hash: 'ahash'

        Also initialize a result_score variable to hold deduplication results later if desired.

        Example usage for selecting average hash:
        ```
        from imagededup import hashing
        myhasher = hashing.Hashing(method='ahash')
        ```
        """

        self.method = method
        method_dict = {'phash': self._phash, 'dhash': self._dhash, 'ahash': self._ahash,
                       'whash': self._whash}

        self.hash_func = method_dict.get(self.method)

        if self.hash_func is None:
            raise ValueError('Choose a correct hashing method. The available hashing methods are: \'phash\', \'dhash\','
                             ' \'ahash\' and \'whash\'')

        self.result_score = None  # {query_filename: {retrieval_filename:hamming distance, ...}, ..}
        self.logger = return_logger(__name__, os.getcwd())

    @staticmethod
    def bool_to_hex(x: np.array) -> str:
        str_bool = ''.join([str(int(i)) for i in x])
        int_base2 = int(str_bool, 2)  # int base 2
        return '{:0x}'.format(int_base2)

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> float:
        """
        Calculates the hamming distance between two hashes. If length of hashes is not 64 bits, then pads the length
        to be 64 for each hash and then calculates the hamming distance.
        :param hash1: hash string
        :param hash2: hash string
        :return: Hamming distance between the two hashes.
        """
        hash1_bin = bin(int(hash1, 16))[2:].zfill(64)  # zfill ensures that len of hash is 64 and pads MSB if it is < A
        hash2_bin = bin(int(hash2, 16))[2:].zfill(64)
        return np.sum([i != j for i, j in zip(hash1_bin, hash2_bin)])

    # Feature generation part

    def _get_hash(self, hash_mat: np.array, n_blocks: int) -> str:
        """
        Convert a matrix of binary numerals to an n_blocks length hash.
        :param hash_mat: A numpy array consisting of 0/1 values.
        :param n_blocks: Hash length required.
        :return: An n_blocks length hash string.
        """
        calculated_hash = []
        for i in np.array_split(np.ndarray.flatten(hash_mat), n_blocks):
            calculated_hash.append(self.bool_to_hex(i))
        return ''.join(calculated_hash)

    def _phash(self, path_image: PosixPath) -> str:
        """
        Get perceptual hash of the input image.
        :param path_image: A PosixPath to image or a numpy array that corresponds to the image.
        :return: A string representing the perceptual hash of the image.
        Implementation reference: http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
        ```
        """

        res_dims = (32, 32)
        im_gray_arr = convert_to_array(path_image, resize_dims=res_dims, for_hashing=True)
        dct_coef = scipy.fftpack.dct(scipy.fftpack.dct(im_gray_arr, axis=0), axis=1)
        dct_reduced_coef = dct_coef[:8, :8]  # retain top left 8 by 8 dct coefficients
        mean_coef_val = np.mean(np.ndarray.flatten(dct_reduced_coef)[1:])  # average of coefficients excluding the DC
        # term (0th term)
        hash_mat = dct_reduced_coef >= mean_coef_val  # All coefficients greater than mean of coefficients
        return self._get_hash(hash_mat, 16)  # 16 character output

    def _ahash(self, path_image: PosixPath) -> str:
        """
        Get average hash of the input image.
        :param path_image: A PosixPath to image or a numpy array that corresponds to the image.
        :return: A string representing the average hash of the image.
        """

        res_dims = (8, 8)
        im_gray_arr = convert_to_array(path_image, resize_dims=res_dims, for_hashing=True)
        avg_val = np.mean(im_gray_arr)
        hash_mat = im_gray_arr >= avg_val
        return self._get_hash(hash_mat, 16)  # 16 character output

    def _dhash(self, path_image: PosixPath) -> str:
        """
        Get difference hash of the input image.
        :param path_image: A PosixPath to image or a numpy array that corresponds to the image.
        :return: A string representing the difference hash of the image.

        Implementation reference: http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
        """

        res_dims = (9, 8)
        im_gray_arr = convert_to_array(path_image, resize_dims=res_dims, for_hashing=True)
        # hash_mat = im_gray_arr[:, :-1] > im_gray_arr[:, 1:]  # Calculates difference between consecutive columns
        hash_mat = im_gray_arr[:, 1:] > im_gray_arr[:, :-1]
        return self._get_hash(hash_mat, 16)  # 16 character output

    def _whash(self, path_image: PosixPath) -> str:
        """
        Get wavelet hash of the input image.
        :param path_image: A PosixPath to image or a numpy array that corresponds to the image.
        :return: A string representing the wavelet hash of the image.
        """

        res_dims = (256, 256)
        im_gray_arr = convert_to_array(path_image, resize_dims=res_dims, for_hashing=True)
        coeffs = pywt.wavedec2(im_gray_arr, 'haar', level=5)  # decomposition level set to 5 to get 8 by 8 hash matrix
        LL_coeff = coeffs[0]

        mean_coef_val = np.mean(np.ndarray.flatten(LL_coeff))  # average of LL coefficients
        hash_mat = LL_coeff >= mean_coef_val  # All coefficients greater than mean of coefficients
        return self._get_hash(hash_mat, 16)  # 16 character output

    def hash_image(self, path_image: PosixPath) -> str:
        """
        Apply a hashing function on the input image.
        :param path_image: A PosixPath to image or a numpy array that corresponds to the image.
        :return: A string representing the hash of the image.
        """
        return self.hash_func(path_image)

    def hash_dir(self, path_dir: PosixPath) -> Dict:
        """
        Apply the hashing method to all images in a directory.
        :param path_dir: PosixPath to the directory containing images for which hashes are to be obtained.
        :return: Dictionary containing file names as keys and corresponding hash string as value.
        """

        filenames = check_directory_files(path_dir, return_file=True)
        self.logger.info(f'Start: Calculating hashes using {self.method}!')
        hash_dict = dict()
        for i in filenames:
            hash_dict[i.name] = self.hash_func(Path(i))
        self.logger.info(f'End: Calculating hashes using {self.method}!')
        return hash_dict  # dict_file_feature from _find_duplicates_dict input

    # Search part

    def _find_duplicates_dict(self, dict_file_feature: Dict[str, str], threshold: int = 10,
                              scores: bool = False) -> Dict:
        """Takes in dictionary {filename: hash string}, detects duplicates above the given hamming distance threshold
            and returns dictionary containing key as filename and value as a list of duplicate filenames. Optionally,
            the hamming distances could be returned instead of just duplicate file name for each query file.

        :param dict_file_feature: Dictionary with keys as file names and values as hash strings for the key image file.
        :param threshold: Cosine similarity above which retrieved duplicates are valid.
        :param scores: Boolean indicating whether similarity scores are to be returned along with retrieved duplicates.
        :return: if scores is True, then a dictionary of the form {'image1.jpg': {'image1_duplicate1.jpg':
        <distance>, 'image1_duplicate2.jpg':<distance>, ..}, 'image2.jpg':{'image1_duplicate1.jpg':
        <distance>,..}}
        if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg',
        'image1_duplicate2.jpg'], 'image2.jpg':['image1_duplicate1.jpg',..], ..}"""

        self.logger.info('Start: Evaluating hamming distances for getting duplicates')
        result_set = HashEval(test=dict_file_feature, queries=dict_file_feature, hammer=self.hamming_distance,
                      cutoff=threshold, search_method='bktree', save=False)
        self.logger.info('End: Evaluating hamming distances for getting duplicates')
        self.result_score = result_set.retrieve_results()
        if scores:
            return self.result_score
        else:
            return result_set.retrieve_result_list()

    def _find_duplicates_dir(self, path_dir: PosixPath, threshold: int = 10, scores: bool = False) -> Dict:
        """Takes in path of the directory on which duplicates are to be detected above the given threshold.
        Returns dictionary containing key as filename and value as a list of duplicate file names.

        :param path_dir: PosixPath to the directory containing all the images.
        :param threshold: Hamming distance above which retrieved duplicates are valid.
        :param scores: Boolean indicating whether Hamming distances are to be returned along with retrieved duplicates.
        :return: if scores is True, then a dictionary of the form {'image1.jpg': {'image1_duplicate1.jpg':<distance>,
        'image1_duplicate2.jpg':<distance>, ..}, 'image2.jpg':{'image1_duplicate1.jpg':<distance>,..}}
        if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg',
        'image1_duplicate2.jpg'], 'image2.jpg':['image2_duplicate1.jpg',..], ..}"""

        dict_file_feature = self.hash_dir(path_dir)
        dict_ret = self._find_duplicates_dict(dict_file_feature=dict_file_feature, threshold=threshold, scores=scores)
        return dict_ret

    @staticmethod
    def _check_hamming_distance_bounds(thresh: int) -> None:
        """
        Checks if provided threshold is valid. Raises TypeError is wrong threshold variable type is passed or a value
        out of range is supplied.

        :param thresh: Threshold value (must be int between 0 and 64 inclusive)
        """

        if not isinstance(thresh, int) or (thresh < 0 or thresh > 64):
            raise TypeError('Threshold must be an int between 0 and 64')

    def find_duplicates(self, path_or_dict, threshold: int = 10, scores: bool = False) -> Dict:
        """
        Finds duplicates. Raises TypeError if the supplied directory path isn't a Path variable or a valid dictionary
        isn't supplied.

        :param path_or_dict: PosixPath to the directory containing all the images or dictionary with keys as file names
        and values as numpy arrays which represent the CNN feature for the key image file.
        :param threshold: Threshold value (must be float between -1.0 and 1.0)
        :param scores: Boolean indicating whether similarity scores are to be returned along with retrieved duplicates.
        :return: if scores is True, then a dictionary of the form {'image1.jpg': {'image1_duplicate1.jpg':<distance>,
        'image1_duplicate2.jpg':<distance>, ..}, 'image2.jpg':{'image1_duplicate1.jpg':<distance>,..}}
        if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg',
        'image1_duplicate2.jpg'], 'image2.jpg':['image1_duplicate1.jpg',..], ..}

            Example usage:
        ```
        from imagededup import hashing
        myhasher = hashing.Hashing(method='phash')
        dict_ret_with_dict_inp = myhasher.find_duplicates(dict_file_feat, threshold=15, scores=True)

        OR

        from imagededup import hashing
        myhasher = hashing.Hashing(method='phash')
        dict_ret_path = myhasher.find_duplicates(Path('path/to/directory'), threshold=15, scores=True)
        ```
        """

        self._check_hamming_distance_bounds(thresh=threshold)
        if isinstance(path_or_dict, PosixPath):
            dict_ret = self._find_duplicates_dir(path_dir=path_or_dict, threshold=threshold, scores=scores)
        elif isinstance(path_or_dict, dict):
            dict_ret = self._find_duplicates_dict(dict_file_feature=path_or_dict, threshold=threshold, scores=scores)
        else:
            raise TypeError('Provide either a directory path variable to deduplicate or a dictionary of filenames and '
                            'vectors!')
        return dict_ret

    def find_duplicates_to_remove(self, path_or_dict, threshold: int = 10) -> List:
        """
        Gives out a list of image file names to remove based on the similarity threshold.
        :param path_or_dict: PosixPath to the directory containing all the images or dictionary with keys as file names
        and values as numpy arrays which represent the CNN feature for the key image file.
        :param threshold: Threshold value (must be float between -1.0 and 1.0)
        :return: List of image file names that should be removed.

        Example usage:
        ```
        from imagededup import hashing
        myhasher = hashing.Hashing(method='phash')
        list_of_files_to_remove = myhasher.find_duplicates_to_remove(Path('path/to/images/directory'),
        threshold=15)
        ```
        """

        dict_ret = self.find_duplicates(path_or_dict=path_or_dict, threshold=threshold, scores=False)
        return get_files_to_remove(dict_ret)


class Dataset:
    """
    Class wrapper to instantiate a Dataset object composing of a subset of test images
    and a smaller fraction of images that are used as queries to test search and retrieval.
    Object contains hashed image fingerprints as well, however, hashing method is set by user.
    """
    def __init__(self, path_to_queries: str, path_to_test: str) -> None:
        self.query_docs = self.load_image_set(path_to_queries)
        self.test_docs = self.load_image_set(path_to_test)

    @staticmethod
    def load_image_set(path: str) -> Dict:
        return {doc: os.path.join(path, doc) for doc in os.listdir(path) if doc.endswith('.jpg')}


class HashedDataset(Dataset):
    def __init__(self, hashing_function: FunctionType, *args, **kwargs) -> None:
        super(HashedDataset, self).__init__(*args, **kwargs)
        self.hasher = hashing_function
        self.fingerprint()
        self.doc2hash = deepcopy(self.test_hashes)
        self.doc2hash.update(self.query_hashes)
        self.hash2doc = {self.doc2hash[doc]: doc for doc in self.doc2hash}

    def fingerprint(self) -> None:
        self.test_hashes = {doc: str(self.hasher(np.array(Image.open(self.test_docs[doc])))) for doc in self.test_docs}
        self.query_hashes = {doc: str(self.hasher(np.array(Image.open(self.query_docs[doc])))) for doc in self.query_docs}

    def get_docmap(self) -> Dict:
        return self.hash2doc

    def _get_hashmap(self) -> Dict:
        return self.doc2hash

    def get_query_hashes(self) -> Dict:
        return self.query_hashes

    def get_test_hashes(self) -> Dict:
        return self.test_hashes
