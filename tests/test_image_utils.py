from imagededup.utils import image_utils
from pathlib import Path
from PIL import Image
import numpy as np
import pytest


def test_check_valid_file_file_not_exists():
    path_file = Path('tests/data/bla.jpg')
    with pytest.raises(FileNotFoundError):
        image_utils._validate_single_image(path_file)


def test_check_valid_file_unsupported_format():
    path_file = Path('tests/data/formats_images/ukbench09380.svg')
    with pytest.raises(TypeError):
        image_utils._validate_single_image(path_file)


def test_check_valid_file_correct_formats_passes():
    path_file = Path('tests/data/base_images/ukbench00120.jpg')
    assert image_utils._validate_single_image(path_file) == 1
    path_file = Path('tests/data/formats_images/ukbench09380.bmp')
    assert image_utils._validate_single_image(path_file) == 1
    path_file = Path('tests/data/formats_images/ukbench09380.png')
    assert image_utils._validate_single_image(path_file) == 1
    path_file = Path('tests/data/formats_images/ukbench09380.jpeg')
    assert image_utils._validate_single_image(path_file) == 1


def test__load_image_valid_image_loads():
    path_file = Path('tests/data/base_images/ukbench00120.jpg')
    assert image_utils._load_image(path_file).format == 'JPEG'


def test_check_directory_files():
    path_dir = Path('tests/data/base_images')
    assert len(image_utils.check_directory_files(path_dir=path_dir, return_file=True)) == 10


def test_check_directory_files_no_return():
    path_dir = Path('tests/data/base_images')
    assert image_utils.check_directory_files(path_dir=path_dir, return_file=False) is None


def test_check_directory_files_invalid():
    path_dir = Path('tests/data/formats_images')
    with pytest.raises(Exception):
        image_utils.check_directory_files(path_dir=path_dir, return_file=False)
    with pytest.raises(Exception):
        image_utils.check_directory_files(path_dir=path_dir, return_file=True)


def test__image_preprocess_return_type():
    """Give Pillow image and check that return type array"""
    pillow_image = Image.open(Path('tests/data/mixed_images/ukbench00120.jpg'))
    assert isinstance(image_utils._image_preprocess(pillow_image, resize_dims=(8, 8)), np.ndarray)


def test__image_preprocess_forhashing_false():
    """Give Pillow image and check that return type array"""
    pillow_image = Image.open(Path('tests/data/mixed_images/ukbench00120.jpg'))
    assert isinstance(image_utils._image_preprocess(pillow_image, resize_dims=(8, 8), for_hashing=False), np.ndarray)


def test__image_preprocess_size():
    """Give Pillow image and check that returned array has size=self.TARGET_SIZE"""
    pillow_image = Image.open(Path('tests/data/mixed_images/ukbench00120.jpg'))
    res_dim = (8, 8)
    assert image_utils._image_preprocess(pillow_image, resize_dims=res_dim).shape == res_dim


def test_convert_to_array_path():
    path_dir = Path('tests/data/mixed_images/ukbench00120.jpg')
    assert len(image_utils.convert_to_array(path_dir, resize_dims=(8, 8))) != 0


def test_convert_to_array_array():
    path_image = Path('tests/data/mixed_images/ukbench00120.jpg')
    im_arr = np.array(Image.open(path_image))
    assert len(image_utils.convert_to_array(im_arr, resize_dims=(8, 8))) != 0


def test_convert_to_array_float_array():
    path_image = Path('tests/data/mixed_images/ukbench00120.jpg')
    im_arr = np.array(Image.open(path_image)).astype('float32')
    assert isinstance(image_utils.convert_to_array(im_arr, resize_dims=(8, 8)), np.ndarray)


def test__convert_to_array_unacceptable_input():
    with pytest.raises(TypeError):
        image_utils.convert_to_array('tests/data/mixed_images', resize_dims=(8, 8))