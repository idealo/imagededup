from imagededup.utils.image_utils import preprocess_image, load_image
from pathlib import Path
from PIL import Image
import numpy as np
import pytest

PATH_SINGLE_IMAGE = Path('tests/data/mixed_images/ukbench00120.jpg')


def test_preprocess_image_accepts_array_input():
    inp_x = Image.open(PATH_SINGLE_IMAGE)
    inp_x = np.array(inp_x)
    target_size = (2, 2)
    ret_array = preprocess_image(inp_x, target_size=target_size, grayscale=True)
    assert isinstance(ret_array, np.ndarray)
    assert ret_array.shape == target_size


def test_preprocess_image_accepts_pil_input():
    inp_x = Image.open(PATH_SINGLE_IMAGE)
    target_size = (2, 2)
    ret_array = preprocess_image(inp_x, target_size=target_size, grayscale=True)
    assert isinstance(ret_array, np.ndarray)
    assert ret_array.shape == target_size


def test_preprocess_image_wrong_input():
    inp = 'test_string'
    with pytest.raises(ValueError):
        preprocess_image(inp, target_size=(2, 2))


def test_preprocess_image_grayscale_false():
    inp_x = Image.open(PATH_SINGLE_IMAGE)
    target_size = (2, 2)
    ret_array = preprocess_image(inp_x, target_size=target_size, grayscale=False)
    assert isinstance(ret_array, np.ndarray)
    assert ret_array.shape == target_size + (3,)  # 3 for RGB


# load_image


def test_load_image_accepts_pil(mocker):
    preprocess_mocker = mocker.patch('imagededup.utils.image_utils.preprocess_image')
    load_image(PATH_SINGLE_IMAGE)
    preprocess_mocker.assert_called_once_with(
        Image.open(PATH_SINGLE_IMAGE), target_size=None, grayscale=False
    )


def test_load_image_returns_none_wrong_input():
    inp = 'test_string'
    assert load_image(inp) is None


def test_load_image_wrong_image_format():
    assert load_image(Path('tests/data/formats_images/Iggy.1024.ppm')) is None


@pytest.fixture
def preprocess_mocker(mocker):
    return mocker.patch('imagededup.utils.image_utils.preprocess_image')


def test_load_image_alpha_channel_image_converts(preprocess_mocker):
    PATH_ALPHA_IMAGE = Path('tests/data/alpha_channel_image.png')
    alpha_converted = Image.open(PATH_ALPHA_IMAGE).convert('RGBA').convert('RGB')
    load_image(PATH_ALPHA_IMAGE)
    preprocess_mocker.assert_called_once_with(alpha_converted, target_size=None, grayscale=False)


def test_load_image_target_size_grayscale_true(preprocess_mocker):
    load_image(image_file=PATH_SINGLE_IMAGE, target_size=(8, 8), grayscale=True)
    preprocess_mocker.assert_called_once_with(
        Image.open(PATH_SINGLE_IMAGE), target_size=(8, 8), grayscale=True
    )


# Integration test


def test_load_image_all_inputs_correct():
    target_size = (8, 8)
    loaded_image = load_image(image_file=PATH_SINGLE_IMAGE, target_size=target_size, grayscale=True)
    assert isinstance(loaded_image, np.ndarray)
    assert loaded_image.shape == target_size
    assert np.issubdtype(np.uint8, loaded_image.dtype)  # return numpy array dtype is uint8


# def test_check_valid_file_file_not_exists():
#     path_file = Path('tests/data/bla.jpg')
#     with pytest.raises(FileNotFoundError):
#         image_utils._validate_single_image(path_file)
#
#
# def test_check_valid_file_unsupported_format():
#     path_file = Path('tests/data/formats_images/ukbench09380.svg')
#     with pytest.raises(TypeError):
#         image_utils._validate_single_image(path_file)
#
#
# def test_check_valid_file_correct_formats_passes():
#     path_file = Path('tests/data/base_images/ukbench00120.jpg')
#     assert image_utils._validate_single_image(path_file) == 1
#     path_file = Path('tests/data/formats_images/ukbench09380.bmp')
#     assert image_utils._validate_single_image(path_file) == 1
#     path_file = Path('tests/data/formats_images/ukbench09380.png')
#     assert image_utils._validate_single_image(path_file) == 1
#     path_file = Path('tests/data/formats_images/ukbench09380.jpeg')
#     assert image_utils._validate_single_image(path_file) == 1
#
#
# def test__load_image_valid_image_loads():
#     path_file = Path('tests/data/base_images/ukbench00120.jpg')
#     assert image_utils._load_image(path_file).format == 'JPEG'
#
#
# def test_check_directory_files():
#     path_dir = Path('tests/data/base_images')
#     assert len(image_utils.check_directory_files(path_dir=path_dir, return_file=True)) == 10
#
#
# def test_check_directory_files_no_return():
#     path_dir = Path('tests/data/base_images')
#     assert image_utils.check_directory_files(path_dir=path_dir, return_file=False) is None
#
#
# def test_check_directory_files_invalid():
#     path_dir = Path('tests/data/formats_images')
#     with pytest.raises(Exception):
#         image_utils.check_directory_files(path_dir=path_dir, return_file=False)
#     with pytest.raises(Exception):
#         image_utils.check_directory_files(path_dir=path_dir, return_file=True)
#
#
# def test__image_preprocess_return_type():
#     """Give Pillow image and check that return type array"""
#     pillow_image = Image.open(Path('tests/data/mixed_images/ukbench00120.jpg'))
#     assert isinstance(image_utils._image_preprocess(pillow_image, resize_dims=(8, 8)), np.ndarray)
#
#
# def test__image_preprocess_forhashing_false():
#     """Give Pillow image and check that return type array"""
#     pillow_image = Image.open(Path('tests/data/mixed_images/ukbench00120.jpg'))
#     assert isinstance(image_utils._image_preprocess(pillow_image, resize_dims=(8, 8), for_hashing=False), np.ndarray)
#
#
# def test__image_preprocess_size():
#     """Give Pillow image and check that returned array has size=resize_dims"""
#     pillow_image = Image.open(Path('tests/data/mixed_images/ukbench00120.jpg'))
#     res_dim = (8, 8)
#     assert image_utils._image_preprocess(pillow_image, resize_dims=res_dim).shape == res_dim
#
#
# def test_convert_to_array_path():
#     path_dir = Path('tests/data/mixed_images/ukbench00120.jpg')
#     assert len(image_utils.convert_to_array(path_dir, resize_dims=(8, 8))) != 0
#
#
# def test_convert_to_array_array():
#     path_image = Path('tests/data/mixed_images/ukbench00120.jpg')
#     im_arr = np.array(Image.open(path_image))
#     assert len(image_utils.convert_to_array(im_arr, resize_dims=(8, 8))) != 0
#
#
# def test_convert_to_array_float_array():
#     path_image = Path('tests/data/mixed_images/ukbench00120.jpg')
#     im_arr = np.array(Image.open(path_image)).astype('float32')
#     assert isinstance(image_utils.convert_to_array(im_arr, resize_dims=(8, 8)), np.ndarray)
#
#
# def test__convert_to_array_unacceptable_input():
#     with pytest.raises(TypeError):
#         image_utils.convert_to_array('tests/data/mixed_images', resize_dims=(8, 8))
#
#
# def test__convert_to_array_pillow_input_raises_typeerror():
#     im_pil = Image.open(Path('tests/data/base_images/ukbench00120.jpg'))
#     with pytest.raises(TypeError):
#         image_utils.convert_to_array(im_pil, resize_dims=(8, 8))
