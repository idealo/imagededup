import os
import pywt
import scipy.fftpack
import numpy as np
from PIL import Image
from pathlib import Path
from types import FunctionType
from typing import Tuple
import random
from copy import deepcopy
"""
TODO:

Wavelet hash: Zero the LL coeff, reconstruct image, the get wavelet transform
Wavelet hash: Allow possibility of different wavelet functions

For all: Restrict image sizes to be greater than a certain size
Allow acceptance of os.path in addition to already existing Path and numpy image array.
"""

class Hashing:
    def __init__(self):
        pass

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

    @staticmethod
    def run_hash_on_dir(path_dir: Path, hashing_function: FunctionType) -> dict:
        filenames = [os.path.join(path_dir, i) for i in os.listdir(path_dir) if i != '.DS_Store']
        hash_dict = dict(zip(filenames, [None] * len(filenames)))
        for i in filenames:
            hash_dict[i] = hashing_function(Path(i))
        return hash_dict

    @staticmethod
    def image_preprocess(path_image: Path, resize_dims: Tuple[int, int]) -> np.array:
        im = Image.open(path_image)
        im_res = im.resize(resize_dims, Image.ANTIALIAS)
        im_gray = im_res.convert('L')  # convert to grayscale (i.e., single channel)
        im_gray_arr = np.array(im_gray)
        return im_gray_arr

    def convert_to_array(self, path_image: None, resize_dims: Tuple[int, int] = (8, 8)) -> np.ndarray:
        try:
            if isinstance(path_image, Path):
                im_gray_arr = self.image_preprocess(path_image, resize_dims)
            elif isinstance(path_image, np.ndarray):
                im = Image.fromarray(path_image)
                im_res = im.resize(resize_dims, Image.ANTIALIAS)
                im_gray = im_res.convert('L')
                im_gray_arr = np.array(im_gray)
            else:
                raise Exception
            return im_gray_arr
        except Exception:
            print('Check Input Format! Input should be either a Path Variable or a numpy array!')
            raise

    def get_hash(self, hash_mat: np.array, n_blocks: int) -> str:
        calculated_hash = []
        for i in np.array_split(np.ndarray.flatten(hash_mat), n_blocks):
            calculated_hash.append(self.bool_to_hex(i))
        return ''.join(calculated_hash)

    def phash(self, path_image: None) -> str:
        """Implementation reference: http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html"""
        res_dims = (32, 32)
        im_gray_arr = self.convert_to_array(path_image, resize_dims=res_dims)
        dct_coef = scipy.fftpack.dct(scipy.fftpack.dct(im_gray_arr, axis=0), axis=1)
        dct_reduced_coef = dct_coef[:8, :8]  # retain top left 8 by 8 dct coefficients
        mean_coef_val = np.mean(np.ndarray.flatten(dct_reduced_coef)[1:])  # average of coefficients excluding the DC
        # term (0th term)
        hash_mat = dct_reduced_coef >= mean_coef_val  # All coefficients greater than mean of coefficients
        return self.get_hash(hash_mat, 16)  # 16 character output

    def ahash(self, path_image: Path) -> str:
        res_dims = (8, 8)
        im_gray_arr = self.convert_to_array(path_image, resize_dims=res_dims)
        avg_val = np.mean(im_gray_arr)
        hash_mat = im_gray_arr >= avg_val
        return self.get_hash(hash_mat, 16)  # 16 character output

    def dhash(self, path_image: Path) -> str:
        """Implementation reference: http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html"""
        res_dims = (9, 8)
        im_gray_arr = self.convert_to_array(path_image, resize_dims=res_dims)
        # hash_mat = im_gray_arr[:, :-1] > im_gray_arr[:, 1:]  # Calculates difference between consecutive columns
        hash_mat = im_gray_arr[:, 1:] > im_gray_arr[:, :-1]
        return self.get_hash(hash_mat, 16)  # 16 character output

    def whash(self, path_image: None) -> str:
        res_dims = (256, 256)
        im_gray_arr = self.convert_to_array(path_image, resize_dims=res_dims)
        coeffs = pywt.wavedec2(im_gray_arr, 'haar', level=5)  # decomposition level set to 5 to get 8 by 8 hash matrix
        LL_coeff = coeffs[0]

        mean_coef_val = np.mean(np.ndarray.flatten(LL_coeff))  # average of LL coefficients
        hash_mat = LL_coeff >= mean_coef_val  # All coefficients greater than mean of coefficients
        return self.get_hash(hash_mat, 16)  # 16 character output

    def phash_dir(self, path_dir: Path) -> dict:
        return self.run_hash_on_dir(path_dir, self.phash)

    def ahash_dir(self, path_dir: Path) -> dict:
        return self.run_hash_on_dir(path_dir, self.ahash)

    def dhash_dir(self, path_dir: Path) -> dict:
        return self.run_hash_on_dir(path_dir, self.dhash)


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
    def load_image_set(path: str) -> dict: 
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
        
        
    def get_docmap(self) -> dict:
        return self.hash2doc
    
    def get_hashmap(self) -> dict:
        return self.doc2hash


    def get_query_hashes(self) -> dict:
        return self.query_hashes


    def get_test_hashes(self) -> dict:
        return self.test_hashes