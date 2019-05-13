from imagededup.hashing import Hashing
import pytest
import numpy as np
from pathlib import Path
from PIL import Image

""" Run from project root with: python -m pytest -vs tests/test_hash.py"""


def test_hamming_distance():
    """Put two numbers and check if hamming distance is correct"""
    number_1 = "1101"
    number_2 = "1100"
    hamdist = Hashing.hamming_distance(number_1, number_2)
    assert hamdist == 1


def test_if_using_path_works(path_image=Path('tests/data/images/ukbench00120.jpg')):
    hash_obj = Hashing()
    hash_im = hash_obj.convert_to_array(path_image)
    assert len(hash_im)


def test_if_using_numpy_input_works(path_image=Image.open(Path('tests/data/images/ukbench00120.jpg'))):
    hash_obj = Hashing()
    hash_im = hash_obj.convert_to_array(np.array(path_image))
    assert len(hash_im)


@pytest.mark.parametrize('hash_function', [Hashing().phash, Hashing().ahash, Hashing().dhash])
class TestCommon:
    def test_len_hash(self, hash_function):
        hash_im = hash_function(Path('tests/data/images/ukbench00120.jpg'))
        assert len(hash_im) == 16

    def test_hash_resize(self, hash_function):
        """Resize one image to (300, 300) and check that hamming distance between hashes is not too different"""
        hash_im_1 = hash_function(Path('tests/data/images/ukbench00120.jpg'))
        hash_im_2 = hash_function(Path('tests/data/images/ukbench00120_resize.jpg'))
        hamdist = Hashing.hamming_distance(hash_im_1, hash_im_2)
        assert hamdist < 3

    def test_hash_small_rotation(self, hash_function):
        """Rotate image slightly (1 degree) and check that hamming distance between hashes is not too different"""
        orig_image = Image.open(Path('tests/data/images/ukbench00120.jpg'))
        rotated_image = np.array(orig_image.rotate(1))
        hash_im_1 = hash_function(np.array(orig_image))
        hash_im_2 = hash_function(rotated_image)
        hamdist = Hashing.hamming_distance(hash_im_1, hash_im_2)
        assert hamdist < 3

    def test_hash_distinct_images(self, hash_function):
        """Put in distinct images and check that hamming distance between hashes is large"""
        image_1 = Image.open(Path('tests/data/images/ukbench00120.jpg'))
        image_2 = Image.open(Path('tests/data/images/ukbench09268.jpg'))
        hash_im_1 = hash_function(np.array(image_1))
        hash_im_2 = hash_function(np.array(image_2))
        hamdist = Hashing.hamming_distance(hash_im_1, hash_im_2)
        assert hamdist > 10





