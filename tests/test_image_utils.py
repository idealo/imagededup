import pytest
from pathlib import Path

import numpy as np
from PIL import Image

from imagededup.utils.image_utils import preprocess_image, load_image

p = Path(__file__)
PATH_SINGLE_IMAGE = p.parent / 'data/mixed_images/ukbench00120.jpg'


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


@pytest.fixture
def preprocess_mocker(mocker):
    return mocker.patch('imagededup.utils.image_utils.preprocess_image')


def test_load_image_alpha_channel_image_converts(preprocess_mocker):
    PATH_ALPHA_IMAGE = p.parent / 'data/alpha_channel_image.png'
    alpha_converted = Image.open(PATH_ALPHA_IMAGE).convert('RGBA').convert('RGB')
    load_image(PATH_ALPHA_IMAGE)
    preprocess_mocker.assert_called_once_with(
        alpha_converted, target_size=None, grayscale=False
    )


def test_load_image_target_size_grayscale_true(preprocess_mocker):
    load_image(image_file=PATH_SINGLE_IMAGE, target_size=(8, 8), grayscale=True)
    preprocess_mocker.assert_called_once_with(
        Image.open(PATH_SINGLE_IMAGE), target_size=(8, 8), grayscale=True
    )


# Integration test


def test_load_image_all_inputs_correct():
    target_size = (8, 8)
    loaded_image = load_image(
        image_file=PATH_SINGLE_IMAGE, target_size=target_size, grayscale=True
    )
    assert isinstance(loaded_image, np.ndarray)
    assert loaded_image.shape == target_size
    assert np.issubdtype(
        np.uint8, loaded_image.dtype
    )  # return numpy array dtype is uint8
