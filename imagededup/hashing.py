from imagededup.retrieval import ResultSet
from imagededup.logger import return_logger
from imagededup.image_utils import check_directory_files, convert_to_array
import os
import pywt
import scipy.fftpack
import numpy as np
from PIL import Image
from pathlib import Path, PosixPath
from types import FunctionType
from typing import Tuple, Dict, List
from copy import deepcopy
"""
TODO:

Wavelet hash: Zero the LL coeff, reconstruct image, then get wavelet transform
Wavelet hash: Allow possibility of different wavelet functions

? Allow acceptance of os.path in addition to already existing Path and numpy image array

"""


class Hashing:
    def __init__(self) -> None:
        self.result_score = None  # {query_filename: {retrieval_filename:hamming distance, ...}, ..}
        self.logger = return_logger(__name__, os.getcwd())

    @staticmethod
    def bool_to_hex(x: np.array) -> str:
        str_bool = ''.join([str(int(i)) for i in x])
        int_base2 = int(str_bool, 2)  # int base 2
        return '{:0x}'.format(int_base2)

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> float:
        hash1_bin = bin(int(hash1, 16))[2:].zfill(64)  # zfill ensures that len of hash is 64 and pads MSB if it is < A
        hash2_bin = bin(int(hash2, 16))[2:].zfill(64)
        return np.sum([i != j for i, j in zip(hash1_bin, hash2_bin)])

    # @staticmethod
    # def _image_preprocess(pillow_image: Image, resize_dims: Tuple[int, int] = (8, 8)) -> np.ndarray:
    #     """
    #     Resizes and typecasts a pillow image to numpy array.
    #
    #     :param pillow_image: A Pillow type image to be processed.
    #     :return: A numpy array of processed image.
    #     """
    #
    #     im_res = pillow_image.resize(resize_dims, Image.ANTIALIAS)
    #     im_gray = im_res.convert('L')  # convert to grayscale (i.e., single channel)
    #     im_arr = np.array(im_gray)
    #     return im_arr
    #
    # def _convert_to_array(self, path_image=None, resize_dims: Tuple[int, int] = (8, 8)) -> np.ndarray:
    #     """
    #     Accepts either path of an image or a numpy array and processes it to feed it to CNN.
    #
    #     :param path_image: PosixPath to the image file or Image typecast to numpy array.
    #     :return: A processed image as numpy array
    #     """
    #
    #     if isinstance(path_image, PosixPath):
    #         # im = Image.open(path_image)
    #         im = load_valid_image(path_image=path_image, load=True)
    #     elif isinstance(path_image, np.ndarray):
    #         im = path_image.astype('uint8')  # fromarray can't take float32/64
    #         im = Image.fromarray(im)
    #     else:
    #         raise TypeError('Check Input Format! Input should be either a Path Variable or a numpy array!')
    #     im_arr = self._image_preprocess(im, resize_dims)
    #     return im_arr

    # Feature generation part
    def _get_hash(self, hash_mat: np.array, n_blocks: int) -> str:
        calculated_hash = []
        for i in np.array_split(np.ndarray.flatten(hash_mat), n_blocks):
            calculated_hash.append(self.bool_to_hex(i))
        return ''.join(calculated_hash)

    def phash(self, path_image: None) -> str:
        """Implementation reference: http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html"""
        res_dims = (32, 32)
        im_gray_arr = convert_to_array(path_image, resize_dims=res_dims, hashmethod=False)
        # im_gray_arr = self._convert_to_array(path_image, resize_dims=res_dims)
        dct_coef = scipy.fftpack.dct(scipy.fftpack.dct(im_gray_arr, axis=0), axis=1)
        dct_reduced_coef = dct_coef[:8, :8]  # retain top left 8 by 8 dct coefficients
        mean_coef_val = np.mean(np.ndarray.flatten(dct_reduced_coef)[1:])  # average of coefficients excluding the DC
        # term (0th term)
        hash_mat = dct_reduced_coef >= mean_coef_val  # All coefficients greater than mean of coefficients
        return self._get_hash(hash_mat, 16)  # 16 character output

    def ahash(self, path_image: PosixPath) -> str:
        res_dims = (8, 8)
        im_gray_arr = convert_to_array(path_image, resize_dims=res_dims)
        avg_val = np.mean(im_gray_arr)
        hash_mat = im_gray_arr >= avg_val
        return self._get_hash(hash_mat, 16)  # 16 character output

    def dhash(self, path_image: PosixPath) -> str:
        """Implementation reference: http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html"""
        res_dims = (9, 8)
        im_gray_arr = convert_to_array(path_image, resize_dims=res_dims)
        # hash_mat = im_gray_arr[:, :-1] > im_gray_arr[:, 1:]  # Calculates difference between consecutive columns
        hash_mat = im_gray_arr[:, 1:] > im_gray_arr[:, :-1]
        return self._get_hash(hash_mat, 16)  # 16 character output

    def whash(self, path_image: None) -> str:
        res_dims = (256, 256)
        im_gray_arr = convert_to_array(path_image, resize_dims=res_dims)
        coeffs = pywt.wavedec2(im_gray_arr, 'haar', level=5)  # decomposition level set to 5 to get 8 by 8 hash matrix
        LL_coeff = coeffs[0]

        mean_coef_val = np.mean(np.ndarray.flatten(LL_coeff))  # average of LL coefficients
        hash_mat = LL_coeff >= mean_coef_val  # All coefficients greater than mean of coefficients
        return self._get_hash(hash_mat, 16)  # 16 character output

    def _run_hash_on_dir(self, path_dir: Path, hashing_function: FunctionType) -> Dict:
        check_directory_files(path_dir)
        self.logger.info(f'Start: Calculating hashes using {hashing_function}!')
        filenames = [os.path.join(path_dir, i) for i in os.listdir(path_dir) if
                     i != '.DS_Store']  # TODO: replace with endswith
        hash_dict = dict(zip(filenames, [None] * len(filenames)))
        for i in filenames:
            hash_dict[i] = hashing_function(Path(i))
        self.logger.info(f'End: Calculating hashes using {hashing_function}!')
        return hash_dict  # dict_file_feature in cnn

    def phash_dir(self, path_dir: PosixPath) -> Dict:
        return self._run_hash_on_dir(path_dir, self.phash)

    def ahash_dir(self, path_dir: PosixPath) -> Dict:
        return self._run_hash_on_dir(path_dir, self.ahash)

    def dhash_dir(self, path_dir: PosixPath) -> Dict:
        return self._run_hash_on_dir(path_dir, self.dhash)

    def whash_dir(self, path_dir: PosixPath) -> Dict:
        return self._run_hash_on_dir(path_dir, self.whash)

    # Search part

    def _find_duplicates_dict(self, dict_file_feature: Dict[str, str], threshold: int = 10,
                              scores: bool = False):
        """Takes in dictionary {filename: hash string}, detects duplicates above the given hamming distance threshold
            and returns dictionary containing key as filename and value as a list of duplicate filenames. Optionally,
            the hamming distances could be returned instead of just duplicate file name for each query file.

        :param dict_file_feature: Dictionary with keys as file names and values as hash strings for the key image file.
        :param threshold: Cosine similarity above which retrieved duplicates are valid.
        :param scores: Boolean indicating whether similarity scores are to be returned along with retrieved duplicates.
        :return: if scores is True, then a dictionary of the form {'image1.jpg': {'image1_duplicate1.jpg':<similarity-score>, 'image1_duplicate2.jpg':<similarity-score>, ..}, 'image2.jpg':{'image1_duplicate1.jpg':<similarity-score>,..}}
        if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg', 'image1_duplicate2.jpg']
        'image2.jpg':['image1_duplicate1.jpg',..], ..}"""

        self.logger.info('Start: Evaluating hamming distances for getting duplicates')
        rs = ResultSet(test=dict_file_feature, queries=dict_file_feature, hammer=self.hamming_distance,
                       cutoff=threshold, search_method='bktree', save=False)
        self.logger.info('End: Evaluating hamming distances for getting duplicates')
        self.result_score = rs.retrieve_results()
        if scores:
            return self.result_score
        else:
            return rs.retrieve_result_list()

    def _find_duplicates_dir(self, path_dir: PosixPath, method='phash', threshold: int = 10, scores: bool = False):
        """Takes in path of the directory on which duplicates are to be detected above the given threshold.
        Returns dictionary containing key as filename and value as a list of duplicate file names.

        :param path_dir: PosixPath to the directory containing all the images.
        :param method: hashing method
        :param threshold: Hamming distance above which retrieved duplicates are valid.
        :param scores: Boolean indicating whether Hamming distances are to be returned along with retrieved duplicates.
        :return: if scores is True, then a dictionary of the form {'image1.jpg': {'image1_duplicate1.jpg':<ham-dist>, 'image1_duplicate2.jpg':<ham-dist>, ..}, 'image2.jpg':{'image1_duplicate1.jpg':<ham-dist>,..}}
        if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg', 'image1_duplicate2.jpg']
        'image2.jpg':['image1_duplicate1.jpg',..], ..}"""

        method_dict = {'phash': self.phash_dir, 'dhash': self.dhash_dir, 'ahash': self.ahash_dir,
                       'whash': self.whash_dir}
        try:
            hash_func = method_dict[method]
        except KeyError:
            raise Exception('Choose a correct hashing method. The available hashing methods are: phash, dhash, ahash '
                            'and whash')
        dict_file_feature = hash_func(path_dir)
        dict_ret = self._find_duplicates_dict(dict_file_feature=dict_file_feature, threshold=threshold, scores=scores)
        return dict_ret

    @staticmethod
    def _check_hamming_distance_bounds(thresh: int) -> None:
        """
        Checks if provided threshold is valid. Raises TypeError is wrong threshold variable type is passed or a value out
        of range is supplied.

        :param thresh: Threshold value (must be int between 0 and 64 inclusive)
        """

        if not isinstance(thresh, int) or (thresh < 0 or thresh > 64):
            raise TypeError('Threshold must be a int between 0 and 64')

    def find_duplicates(self, path_or_dict, method='phash', threshold: int = 10, scores: bool = False):
        """
        Finds duplicates. Raises TypeError if supplied directory path isn't a Path variable or a valid dictionary isn't
        supplied.

        :param path_or_dict: PosixPath to the directory containing all the images or dictionary with keys as file names
        and values as numpy arrays which represent the CNN feature for the key image file.
        :param method: hashing method
        :param threshold: Threshold value (must be float between -1.0 and 1.0)
        :param scores: Boolean indicating whether similarity scores are to be returned along with retrieved duplicates.
        :return: if scores is True, then a dictionary of the form {'image1.jpg': {'image1_duplicate1.jpg'::<ham-dist>,
        'image1_duplicate2.jpg'::<ham-dist>, ..}, 'image2.jpg':{'image1_duplicate1.jpg'::<ham-dist>,..}}
        if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg',
        'image1_duplicate2.jpg'], 'image2.jpg':['image1_duplicate1.jpg',..], ..}

            Example usage:
        ```
        from imagededup import hashing
        myhasher = hashing.Hashing()
        dict_ret_with_dict_inp = myhasher.find_duplicates(dict_file_feat, method='phash', threshold=15, scores=True)

        OR

        from imagededup import hashing
        myhasher = hashing.Hashing()
        dict_ret_path = myhasher.find_duplicates(Path('path/to/directory'), method='phash', threshold=15, scores=True)
        ```
        """
        self._check_hamming_distance_bounds(thresh=threshold)
        if isinstance(path_or_dict, PosixPath):
            dict_ret = self._find_duplicates_dir(path_dir=path_or_dict, method=method, threshold=threshold,
                                                 scores=scores)
        elif isinstance(path_or_dict, dict):
            dict_ret = self._find_duplicates_dict(dict_file_feature=path_or_dict, threshold=threshold, scores=scores)
        else:
            raise TypeError('Provide either a directory path variable to deduplicate or a dictionary of filenames and '
                            'vectors!')
        return dict_ret

    def find_duplicates_to_remove(self, path_or_dict, method='phash', threshold: int = 10) -> List:
        """
        Gives out a list of image file names to remove based on the similarity threshold.
        :param path_or_dict: PosixPath to the directory containing all the images or dictionary with keys as file names
        and values as numpy arrays which represent the CNN feature for the key image file.
        :param method: hashing method
        :param threshold: Threshold value (must be float between -1.0 and 1.0)
        :return: List of image file names that should be removed.

        Example usage:
        ```
        from imagededup import hashing
        myhasher = hashing.Hashing()
        list_of_files_to_remove = myhasher.find_duplicates_to_remove(Path('path/to/images/directory'), method='phash',
        threshold=15)
        ```
        """

        dict_ret = self.find_duplicates(path_or_dict=path_or_dict,  method=method, threshold=threshold, scores=False)
        # iterate over dict_ret keys, get value for the key and delete the dict keys that are in the value list

        list_of_files_to_remove = []

        for k, v in dict_ret.items():
            if k not in list_of_files_to_remove:
                list_of_files_to_remove.extend(v)
        return list(set(list_of_files_to_remove)) # set to remove duplicates


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
