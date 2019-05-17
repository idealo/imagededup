import os
import scipy.fftpack
import numpy as np
from PIL import Image
from pathlib import Path
from types import FunctionType
from typing import Tuple

"""
TODO:
refactor: Make another function for hash generation given 8 by 8 hash matrix
Add wavelet hash?
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
        return np.sum([i != j for i, j in zip(hash1, hash2)])

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

    def phash_dir(self, path_dir: Path) -> dict:
        return self.run_hash_on_dir(path_dir, self.phash)

    def ahash_dir(self, path_dir: Path) -> dict:
        return self.run_hash_on_dir(path_dir, self.ahash)

    def dhash_dir(self, path_dir: Path) -> dict:
        return self.run_hash_on_dir(path_dir, self.dhash)
