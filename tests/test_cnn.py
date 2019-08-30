import pytest
from pathlib import Path
import numpy as np
from keras.models import Model
from imagededup.methods.cnn import CNN
from imagededup.utils.image_utils import load_image


p = Path(__file__)
TEST_IMAGE = p.parent / 'data' / 'base_images' / 'ukbench00120.jpg'
TEST_IMAGE_DIR = p.parent / 'data' / 'base_images'
TEST_IMAGE_FORMATS_DIR = p.parent / 'data' / 'formats_images'

TEST_BATCH_SIZE = 64
TEST_TARGET_SIZE = (224, 224)


@pytest.fixture(scope='module')
def cnn():
    return CNN()


def test__init(cnn):
    assert cnn.result_score is None
    assert cnn.batch_size == TEST_BATCH_SIZE
    assert cnn.target_size == TEST_TARGET_SIZE
    assert isinstance(cnn.model, Model)


def test__get_cnn_features_single(cnn):
    img = load_image(TEST_IMAGE, target_size=(224, 224))

    result = cnn._get_cnn_features_single(img)

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 18432)  # 18432 = 3*3*1024*2


def test__get_cnn_features_batch(cnn):

    result = cnn._get_cnn_features_batch(TEST_IMAGE_DIR)

    expected_predicted_files = [
        'ukbench00120.jpg',
        'ukbench01380.jpg',
        'ukbench08976.jpg',
        'ukbench08996.jpg',
        'ukbench09012.jpg',
        'ukbench09040.jpg',
        'ukbench09060.jpg',
        'ukbench09268.jpg',
        'ukbench09348.jpg',
        'ukbench09380.jpg',
    ]

    assert list(result.keys()) == expected_predicted_files

    for i in result.values():
        assert isinstance(i, np.ndarray)
        assert i.shape == (18432,)

    result = cnn._get_cnn_features_batch(TEST_IMAGE_FORMATS_DIR)

    expected_predicted_files = [
        'ukbench09380.bmp',
        'ukbench09380.jpeg',
        'ukbench09380.png',
        'ukbench09380.svg',
    ]

    assert list(result.keys()) == expected_predicted_files

    for i in result.values():
        assert isinstance(i, np.ndarray)
        assert i.shape == (18432,)


def test_encode_image(cnn):
    result = cnn.encode_image(TEST_IMAGE)

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 18432)  # 18432 = 3*3*1024*2

    result = cnn.encode_image(str(TEST_IMAGE))

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 18432)  # 18432 = 3*3*1024*2

    result = cnn.encode_image('')
    assert result is None


def test_encode_images(cnn):
    result = cnn.encode_images(TEST_IMAGE_DIR)

    expected_predicted_files = [
        'ukbench00120.jpg',
        'ukbench01380.jpg',
        'ukbench08976.jpg',
        'ukbench08996.jpg',
        'ukbench09012.jpg',
        'ukbench09040.jpg',
        'ukbench09060.jpg',
        'ukbench09268.jpg',
        'ukbench09348.jpg',
        'ukbench09380.jpg',
    ]

    assert list(result.keys()) == expected_predicted_files

    for i in result.values():
        assert isinstance(i, np.ndarray)
        assert i.shape == (18432,)

    result = cnn.encode_images(TEST_IMAGE_FORMATS_DIR)

    expected_predicted_files = [
        'ukbench09380.bmp',
        'ukbench09380.jpeg',
        'ukbench09380.png',
        'ukbench09380.svg',
    ]

    assert list(result.keys()) == expected_predicted_files

    for i in result.values():
        assert isinstance(i, np.ndarray)
        assert i.shape == (18432,)

    result = cnn.encode_images(str(TEST_IMAGE_FORMATS_DIR))

    expected_predicted_files = [
        'ukbench09380.bmp',
        'ukbench09380.jpeg',
        'ukbench09380.png',
        'ukbench09380.svg',
    ]

    assert list(result.keys()) == expected_predicted_files

    for i in result.values():
        assert isinstance(i, np.ndarray)
        assert i.shape == (18432,)

    with pytest.raises(ValueError):
        cnn.encode_images('abc')
