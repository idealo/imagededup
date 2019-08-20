from imagededup.handlers.search.retrieval import HashEval
from imagededup.utils.logger import return_logger
from imagededup.utils.image_utils import load_image, preprocess_image
from imagededup.utils.general_utils import get_files_to_remove, save_json
import os
import pywt
import numpy as np
from scipy.fftpack import dct
from pathlib import PosixPath
from typing import Dict, List, Optional

"""
TODO:
Wavelet hash: Zero the LL coeff, reconstruct image, then get wavelet transform
Wavelet hash: Allow possibility of different wavelet functions

"""


class Hashing:
    def __init__(self) -> None:
        self.target_size = (8, 8)  # resizing to dims
        self.logger = return_logger(__name__, os.getcwd())

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> float:
        """
        Calculates the hamming distance between two hashes. If length of hashes is not 64 bits, then pads the length
        to be 64 for each hash and then calculates the hamming distance.
        :param hash1: hash string
        :param hash2: hash string
        :return: Hamming distance between the two hashes.
        """
        hash1_bin = bin(int(hash1, 16))[2:].zfill(
            64
        )  # zfill ensures that len of hash is 64 and pads MSB if it is < A
        hash2_bin = bin(int(hash2, 16))[2:].zfill(64)
        return np.sum([i != j for i, j in zip(hash1_bin, hash2_bin)])

    @staticmethod
    def _array_to_hash(hash_mat: np.ndarray) -> str:
        """
        Convert a matrix of binary numerals to an n_blocks length hash.
        :param hash_mat: A numpy array consisting of 0/1 values.
        :return: An hexadecimal hash string.
        """
        return ''.join('%0.2x' % x for x in np.packbits(hash_mat))

    def encode_image(
        self, image_file: Optional[PosixPath] = None, image_array: Optional[np.ndarray] = None
    ) -> str:
        """
        Apply a hashing function on the input image.
        :param path_image: A PosixPath to image or a numpy array that corresponds to the image.
        :return: A string representing the hash of the image.
        """
        if isinstance(image_file, PosixPath):
            image_pp = load_image(
                image_file=image_file, target_size=self.target_size, grayscale=True
            )

        elif isinstance(image_array, np.ndarray):
            image_pp = preprocess_image(
                image=image_array, target_size=self.target_size, grayscale=True
            )
        else:
            raise ValueError('Please provide either image file or image array!')

        return self._hash_func(image_pp) if isinstance(image_pp, np.ndarray) else None

    def encode_images(self, image_dir: PosixPath):

        if not os.path.isdir(image_dir):
            raise ValueError('Please provide a valid directory path!')

        if not isinstance(image_dir, PosixPath):
            raise ValueError('Please provide a Path variable to the image directory!')

        files = [
            i.absolute() for i in image_dir.glob('*') if not i.name.startswith('.')
        ]  # ignore hidden files

        hash_dict = dict()
        self.logger.info(f'Start: Calculating hashes...')
        for _file in files:
            encoding = self.encode_image(_file)

            if encoding:
                hash_dict[_file.name] = encoding

        self.logger.info(f'End: Calculating hashes!')
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
        Checks if provided threshold is valid. Raises TypeError is wrong threshold variable type is passed or a value
        out of range is supplied.

        :param thresh: Threshold value (must be int between 0 and 64 inclusive)
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
        threshold: int = 10,
        scores: bool = False,
        outfile: Optional[str] = None,
    ) -> Dict:
        """Takes in dictionary {filename: hash string}, detects duplicates above the given hamming distance threshold
            and returns dictionary containing key as filename and value as a list of duplicate filenames. Optionally,
            the hamming distances could be returned instead of just duplicate file name for each query file.
        :param encoding_map: Dictionary with keys as file names and values as hash strings for the key image file.
        :param threshold: Cosine similarity above which retrieved duplicates are valid.
        :param scores: Boolean indicating whether similarity scores are to be returned along with retrieved duplicates.
        :return: if scores is True, then a dictionary of the form {'image1.jpg': {'image1_duplicate1.jpg':
        <distance>, 'image1_duplicate2.jpg':<distance>, ..}, 'image2.jpg':{'image1_duplicate1.jpg':
        <distance>,..}}
        if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg',
        'image1_duplicate2.jpg'], 'image2.jpg':['image1_duplicate1.jpg',..], ..}"""

        self.logger.info('Start: Evaluating hamming distances for getting duplicates')

        result_set = HashEval(
            test=encoding_map,
            queries=encoding_map,
            hammer=self.hamming_distance,
            cutoff=threshold,
            search_method='bktree',
        )
        self.logger.info('End: Evaluating hamming distances for getting duplicates')
        self.results = result_set.retrieve_results(scores=scores)
        if outfile:
            save_json(self.results, outfile)
        return self.results

    def _find_duplicates_dir(
        self,
        image_dir: PosixPath,
        threshold: int = 10,
        scores: bool = False,
        outfile: Optional[str] = None,
    ) -> Dict:
        """Takes in path of the directory on which duplicates are to be detected above the given threshold.
        Returns dictionary containing key as filename and value as a list of duplicate file names.
        :param path_dir: PosixPath to the directory containing all the images.
        :param threshold: Hamming distance above which retrieved duplicates are valid.
        :param scores: Boolean indicating whether Hamming distances are to be returned along with retrieved duplicates.
        :return: if scores is True, then a dictionary of the form {'image1.jpg': {'image1_duplicate1.jpg':<distance>,
        'image1_duplicate2.jpg':<distance>, ..}, 'image2.jpg':{'image1_duplicate1.jpg':<distance>,..}}
        if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg',
        'image1_duplicate2.jpg'], 'image2.jpg':['image2_duplicate1.jpg',..], ..}"""

        encoding_map = self.encode_images(image_dir)
        results = self._find_duplicates_dict(
            encoding_map=encoding_map, threshold=threshold, scores=scores, outfile=outfile
        )
        return results

    def find_duplicates(
        self,
        image_dir: PosixPath = None,
        encoding_map: Dict[str, str] = None,
        threshold: int = 10,
        scores: bool = False,
        outfile: Optional[str] = None,
    ) -> Dict:
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
        if image_dir:
            result = self._find_duplicates_dir(
                image_dir=image_dir, threshold=threshold, scores=scores, outfile=outfile
            )
        elif encoding_map:
            result = self._find_duplicates_dict(
                encoding_map=encoding_map, threshold=threshold, scores=scores, outfile=outfile
            )
        else:
            raise ValueError('Provide either an image directory or encodings!')
        return result

    def find_duplicates_to_remove(
        self,
        image_dir: PosixPath = None,
        encoding_map: Dict[str, str] = None,
        threshold: int = 10,
        outfile: Optional[str] = None,
    ) -> List:
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

        result = self.find_duplicates(
            image_dir=image_dir, encoding_map=encoding_map, threshold=threshold, scores=False
        )
        files_to_remove = get_files_to_remove(result)
        if outfile:
            save_json(files_to_remove, outfile)
        return files_to_remove


class PHash(Hashing):
    def __init__(self) -> None:
        super().__init__()
        self.__coefficient_extract = (8, 8)
        self.target_size = (32, 32)

    def _hash_algo(self, image_array):
        """
        Get perceptual hash of the input image.
        :param path_image: A PosixPath to image or a numpy array that corresponds to the image.
        :return: A string representing the perceptual hash of the image.
        Implementation reference: http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
        """
        dct_coef = dct(dct(image_array, axis=0), axis=1)

        # retain top left 8 by 8 dct coefficients
        dct_reduced_coef = dct_coef[
            : self.__coefficient_extract[0], : self.__coefficient_extract[1]
        ]

        # average of coefficients excluding the DC term (0th term)
        # mean_coef_val = np.mean(np.ndarray.flatten(dct_reduced_coef)[1:])
        median_coef_val = np.median(np.ndarray.flatten(dct_reduced_coef)[1:])

        # return mask of all coefficients greater than mean of coefficients
        hash_mat = dct_reduced_coef >= median_coef_val
        return hash_mat


class AHash(Hashing):
    def __init__(self) -> None:
        super().__init__()
        self.target_size = (8, 8)

    def _hash_algo(self, image_array: np.ndarray):
        """
        Get average hash of the input image.
        :param path_image: A PosixPath to image or a numpy array that corresponds to the image.
        :return: A string representing the average hash of the image.
        """
        avg_val = np.mean(image_array)
        hash_mat = image_array >= avg_val
        return hash_mat


class DHash(Hashing):
    def __init__(self) -> None:
        super().__init__()
        self.target_size = (9, 8)

    def _hash_algo(self, image_array):
        """
        Get difference hash of the input image.
        :param path_image: A PosixPath to image or a numpy array that corresponds to the image.
        :return: A string representing the difference hash of the image.
        Implementation reference: http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
        """
        # Calculates difference between consecutive columns and return mask
        hash_mat = image_array[:, 1:] > image_array[:, :-1]
        return hash_mat


class WHash(Hashing):
    def __init__(self) -> None:
        super().__init__()
        self.target_size = (256, 256)
        self.__wavelet_func = 'haar'

    def _hash_algo(self, image_array):
        """
        Get average hash of the input image.
        :param path_image: A PosixPath to image or a numpy array that corresponds to the image.
        :return: A string representing the average hash of the image.
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
