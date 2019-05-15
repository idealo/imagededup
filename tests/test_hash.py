from imagededup.hashing import Hashing
import pytest
import numpy as np
from pathlib import Path
from PIL import Image

""" Run from project root with: python -m pytest -vs tests/test_hash.py"""


def test_bool_to_hex():
    bool_num = '11011101'
    assert Hashing.bool_to_hex(bool_num) == 'dd'


def test_hamming_distance():
    """Put two numbers and check if hamming distance is correct"""
    number_1 = "1101"
    number_2 = "1100"
    hamdist = Hashing.hamming_distance(number_1, number_2)
    assert hamdist == 1


def test_image_preprocess_return_type():
    path_image = Path('tests/data/images/ukbench00120.jpg')
    resize_dims = (8, 8)
    im_gray_arr = Hashing.image_preprocess(path_image, resize_dims)
    assert isinstance(im_gray_arr, np.ndarray)


def test_image_preprocess_return_shape():
    path_image = Path('tests/data/images/ukbench00120.jpg')
    resize_dims = (8, 8)
    im_gray_arr = Hashing.image_preprocess(path_image, resize_dims)
    assert im_gray_arr.shape == resize_dims


def test_image_preprocess_return_nonempty():
    path_image = Path('tests/data/images/ukbench00120.jpg')
    resize_dims = (8, 8)
    im_gray_arr = Hashing.image_preprocess(path_image, resize_dims)
    assert im_gray_arr.size != 0


def test_if_using_path_works(path_image=Path('tests/data/images/ukbench00120.jpg')):
    hash_obj = Hashing()
    hash_im = hash_obj.convert_to_array(path_image)
    assert len(hash_im)


def test_if_using_numpy_input_works(path_image=Image.open(Path('tests/data/images/ukbench00120.jpg'))):
    hash_obj = Hashing()
    hash_im = hash_obj.convert_to_array(np.array(path_image))
    assert len(hash_im)


hash_obj = Hashing()


@pytest.mark.parametrize('hash_function', [hash_obj.phash, hash_obj.ahash, hash_obj.dhash])
class TestCommon:
    def test_len_hash(self, hash_function):
        hash_im = hash_function(Path('tests/data/images/ukbench00120.jpg'))
        assert len(hash_im) == 16

    def test_hash_resize(self, hash_function):
        """Resize one image to (300, 300) and check that hamming distance between hashes is not too large"""
        hash_im_1 = hash_function(Path('tests/data/images/ukbench00120.jpg'))
        hash_im_2 = hash_function(Path('tests/data/images/ukbench00120_resize.jpg'))
        hamdist = Hashing.hamming_distance(hash_im_1, hash_im_2)
        assert hamdist < 3

    def test_hash_small_rotation(self, hash_function):
        """Rotate image slightly (1 degree) and check that hamming distance between hashes is not too large"""
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

    def test_hash_on_dir_returns_dict(self, hash_function):
        path_dir = Path('tests/data/base_images')
        hash_obj = Hashing()
        hash_dict = hash_obj.run_hash_on_dir(path_dir, hash_function)
        assert isinstance(hash_dict, dict)

    def test_hash_on_dir_return_non_none_hashes(self, hash_function):
        path_dir = Path('tests/data/base_images')
        hash_obj = Hashing()
        hash_dict = hash_obj.run_hash_on_dir(path_dir, hash_function)
        for v in hash_dict.values():
            assert v is not None

    def test_hash_on_dir_runs_for_all_files_in_dir(self, hash_function):
        """There are 10 images in the directory below"""
        path_dir = Path('tests/data/base_images')
        hash_obj = Hashing()
        hash_dict = hash_obj.run_hash_on_dir(path_dir, hash_function)
        assert len(hash_dict.keys()) == 10


def test_phash_dir(mocker):
    path_dir = Path('tests/data/base_images')
    hash_obj = Hashing()
    hash_func = hash_obj.phash
    mocker.patch.object(hash_obj, 'run_hash_on_dir')
    hash_obj.phash_dir(path_dir)
    hash_obj.run_hash_on_dir.assert_called_with(path_dir, hash_func)


def test_ahash_dir(mocker):
    path_dir = Path('tests/data/base_images')
    hash_obj = Hashing()
    hash_func = hash_obj.ahash
    mocker.patch.object(hash_obj, 'run_hash_on_dir')
    hash_obj.ahash_dir(path_dir)
    hash_obj.run_hash_on_dir.assert_called_with(path_dir, hash_func)


def test_dhash_dir(mocker):
    path_dir = Path('tests/data/base_images')
    hash_obj = Hashing()
    hash_func = hash_obj.dhash
    mocker.patch.object(hash_obj, 'run_hash_on_dir')
    hash_obj.dhash_dir(path_dir)
    hash_obj.run_hash_on_dir.assert_called_with(path_dir, hash_func)
