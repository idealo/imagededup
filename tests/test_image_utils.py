import pytest
from pathlib import Path

import numpy as np
from PIL import Image

from imagededup.utils.image_utils import (
    preprocess_image,
    load_image,
    _check_3_dim,
    _add_third_dim,
    _raise_wrong_dim_value_error,
    check_image_array_hash,
    expand_image_array_cnn,
)

p = Path(__file__)
PATH_SINGLE_IMAGE = p.parent / 'data/mixed_images/ukbench00120.jpg'


# Array sanity


def test___check_3_dim_raises_assertionerror_wrong_input_shape():
    arr_shape = (3, 224, 224)

    with pytest.raises(AssertionError):
        _check_3_dim(arr_shape)


def test___check_3_dim_not_raises_assertionerror_right_input_shape():
    arr_shape = (224, 224, 3)
    _check_3_dim(arr_shape)


def test__add_third_dim_converts2_to_3_dims():
    two_d_arr = np.array([[1, 2], [3, 4]])
    three_d_arr = _add_third_dim(two_d_arr)
    assert len(three_d_arr.shape) == 3
    np.testing.assert_array_equal(two_d_arr, three_d_arr[..., 0])
    np.testing.assert_array_equal(three_d_arr[..., 0], three_d_arr[..., 1])
    np.testing.assert_array_equal(three_d_arr[..., 1], three_d_arr[..., 2])


def test__raise_wrong_dim_value_error_raises_error():
    import re

    arr_shape = (3, 3)
    with pytest.raises(
        ValueError,
        match=re.escape(
            f'Received image array with shape: {arr_shape}, expected number of image array dimensions are 3 for rgb '
            f'image and 2 for grayscale image!'
        ),
    ):
        _raise_wrong_dim_value_error(arr_shape)


@pytest.fixture()
def chk_3_dim_mocker(mocker):
    return mocker.patch('imagededup.utils.image_utils._check_3_dim')


@pytest.fixture()
def raise_wrong_dim_value_error_mocker(mocker):
    return mocker.patch('imagededup.utils.image_utils._raise_wrong_dim_value_error')


def test_check_image_array_hash_checks_3_dims(chk_3_dim_mocker):
    image_arr_3d = np.random.random((3, 3, 3))
    check_image_array_hash(image_arr_3d)
    chk_3_dim_mocker.assert_called_once_with(image_arr_3d.shape)


def test_check_image_array_wrong_dims_raises_error(
    chk_3_dim_mocker, raise_wrong_dim_value_error_mocker
):
    image_arr_4d = np.random.random((3, 3, 2, 5))
    check_image_array_hash(image_arr_4d)
    chk_3_dim_mocker.assert_not_called()
    raise_wrong_dim_value_error_mocker.assert_called_once_with(image_arr_4d.shape)


def test_check_image_array_2_dims_nothing_happens(
    chk_3_dim_mocker, raise_wrong_dim_value_error_mocker
):
    image_arr_2d = np.random.random((3, 3))
    check_image_array_hash(image_arr_2d)
    chk_3_dim_mocker.assert_not_called()
    raise_wrong_dim_value_error_mocker.assert_not_called()


def test_expand_image_array_cnn_checks_3_dims_and_returns_input_array(chk_3_dim_mocker):
    image_arr_3d = np.random.random((3, 3, 3))
    ret_arr = expand_image_array_cnn(image_arr_3d)
    chk_3_dim_mocker.assert_called_once_with(image_arr_3d.shape)
    np.testing.assert_array_equal(ret_arr, image_arr_3d)


def test_expand_image_array_cnn_2d_adds_dim_unit(
    mocker, chk_3_dim_mocker, raise_wrong_dim_value_error_mocker
):
    image_arr_2d = np.random.random((3, 3))
    reshape_2_dim_mocker = mocker.patch('imagededup.utils.image_utils._add_third_dim')
    expand_image_array_cnn(image_arr_2d)
    chk_3_dim_mocker.assert_not_called()
    raise_wrong_dim_value_error_mocker.assert_not_called()
    reshape_2_dim_mocker.assert_called_once_with(image_arr_2d)


def test_expand_image_array_cnn_2d_adds_dim_int():
    image_arr_2d = np.random.random((3, 3))
    ret_arr = expand_image_array_cnn(image_arr_2d)
    np.testing.assert_array_equal(image_arr_2d, ret_arr[..., 0])
    np.testing.assert_array_equal(ret_arr[..., 0], ret_arr[..., 1])
    np.testing.assert_array_equal(ret_arr[..., 1], ret_arr[..., 2])


def test_expand_image_array_cnn_wrong_dims_raises_error(
    chk_3_dim_mocker, raise_wrong_dim_value_error_mocker
):
    image_arr_4d = np.random.random((3, 3, 2, 5))
    expand_image_array_cnn(image_arr_4d)
    chk_3_dim_mocker.assert_not_called()
    raise_wrong_dim_value_error_mocker.assert_called_once_with(image_arr_4d.shape)


# Preprocess image


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
