import os
import scipy.fftpack
import numpy as np
from PIL import Image
from pathlib import Path

"""
TODO:
refactor: Make another function for hash generation given 8 by 8 hash matrix
Add wavelet hash?
"""


class Hashing:
    def __init__(self):
        pass

    @staticmethod
    def _bool_to_hex(x: np.array) -> str:
        str_bool = ''.join([str(int(i)) for i in x])
        int_base2 = int(str_bool, 2) # int base 2
        return '{:0x}'.format(int_base2)

    @staticmethod
    def image_preprocess(path_image: Path, resize_dims: (int, int)) -> np.array:
        im = Image.open(path_image)
        im_res = im.resize(resize_dims)
        im_gray = im_res.convert('L') # convert to grayscale (i.e., single channel)
        im_gray_arr = np.array(im_gray)
        return im_gray_arr

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> str:
        return np.sum([i!=j for i,j in zip(hash1, hash2)])

    def get_hash(self, hash_mat: np.array, n_blocks: int) -> str:
        calculated_hash = []
        for i in np.array_split(np.ndarray.flatten(hash_mat), n_blocks):
            calculated_hash.append(self._bool_to_hex(i))
        return ''.join(calculated_hash)

    def run_hash_on_dir(self, path_dir, hashing_function):
        filenames = [os.path.join(path_dir, i) for i in os.listdir(path_dir) if i != '.DS_Store']
        hash_dict = dict(zip(filenames, [None] * len(filenames)))

        for i in filenames:
            hash_dict[i] = hashing_function(i)
        return hash_dict

    def phash(self, path_image: Path) -> str:
        """Implementation reference: http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html"""
        im_gray_arr = self.image_preprocess(path_image, (32, 32))
        dct_coef = scipy.fftpack.dct(scipy.fftpack.dct(im_gray_arr, axis=0), axis=1)
        dct_reduced_coef = dct_coef[:8, :8] # retain top left 8 by 8 dct coefficients
        mean_coef_val = np.mean(np.ndarray.flatten(dct_reduced_coef)[1:]) # average of coefficients exclusing the DC
                                                                          # term (0th term)
        hash_mat = dct_reduced_coef >= mean_coef_val # All coefficients greater than mean of coefficients
        return self.get_hash(hash_mat, 16) # 16 character output

    def phash_dir(self, path_dir: Path) -> dict:
        return self.run_hash_on_dir(path_dir, self.phash)

    def ahash(self, path_image: Path) -> str:
        im_gray_arr = self.image_preprocess(path_image, (8, 8))
        avg_val = np.mean(im_gray_arr)
        hash_mat = im_gray_arr >= avg_val
        return self.get_hash(hash_mat, 16) # 16 character output

    def ahash_dir(self, path_dir: Path) -> dict:
        return self.run_hash_on_dir(path_dir, self.ahash)

    def dhash(self, path_image: Path) -> str:
        """Implementation reference: http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html"""
        im_gray_arr = self.image_preprocess(path_image, (9, 8))
        hash_mat = im_gray_arr[:, :-1] > im_gray_arr[:, 1:] # Calculates difference between consecutive columns
        return self.get_hash(hash_mat, 16) # 16 character output

    def dhash_dir(self, path_dir: Path) -> dict:
        return self.run_hash_on_dir(path_dir, self.dhash)




