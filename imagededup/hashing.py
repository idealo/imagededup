import scipy.fftpack
import numpy as np
from PIL import Image

"""
TODO: Put Image loading, resizing, grayscale conversion and arrayfication in one function
Make another function for hash generation given 64 by 64 hash matrix
Convert functions to type annotation format
Add function to calculate hamming distance between hashes of 2 images
Add functions to calculate hashes given a directory
Add wavelet hash?
"""


class Hashing:
    def __init__(self):
        pass

    @staticmethod
    def _bool_to_hex(x):
        str_bool = ''.join([str(int(i)) for i in x])
        int_bool = int(str_bool, 2)
        return '{:0x}'.format(int_bool)

    def phash(self, path_image):
        """Implementation reference: http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html"""
        im = Image.open(path_image)
        im_res = im.resize((32, 32))
        im_gray = im_res.convert('L')
        im_gray_arr = np.array(im_gray)
        dct_coef = scipy.fftpack.dct(scipy.fftpack.dct(im_gray_arr, axis=0), axis=1)
        dct_reduced_coef = dct_coef[:8, :8]
        mean_coef_val = np.mean(np.ndarray.flatten(dct_reduced_coef)[1:])
        hash_mat = dct_reduced_coef >= mean_coef_val

        calculated_hash = []
        for i in np.array_split(np.ndarray.flatten(hash_mat), 16):
            calculated_hash.append(self._bool_to_hex(i))

        return ''.join(calculated_hash)

    def ahash(self, path_image):
        im = Image.open(path_image)
        im_res = im.resize((8, 8))
        im_gray = im_res.convert('L')
        im_gray_arr = np.array(im_gray)
        avg_val = np.mean(im_gray_arr)
        hash_mat = im_gray_arr >= avg_val

        calculated_hash = []
        for i in np.array_split(np.ndarray.flatten(hash_mat), 16):
            calculated_hash.append(self._bool_to_hex(i))

        return ''.join(calculated_hash)

    def dhash(self, path_image):
        """Implementation reference: http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html"""
        im = Image.open(path_image)
        im_res = im.resize((9, 8))
        im_gray = im_res.convert('L')
        im_gray_arr = np.array(im_gray).astype('int')
        hash_mat = im_gray_arr[:, :-1] > im_gray_arr[:, 1:]

        calculated_hash = []
        for i in np.array_split(np.ndarray.flatten(hash_mat), 16):
            calculated_hash.append(self._bool_to_hex(i))

        return ''.join(calculated_hash)




