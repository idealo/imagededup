from imagededup import cnn
from pathlib import Path
from PIL import Image
from pathlib import PosixPath
import pytest
import numpy as np


@pytest.fixture(scope='module')
def initialized_cnn_obj():
    cnn_obj = cnn.CNN()
    return cnn_obj


def test__image_preprocess_return_type(initialized_cnn_obj):
    """Give Pillow image and check that return type array"""
    pillow_image = Image.open(Path('tests/data/mixed_images/ukbench00120.jpg'))
    assert isinstance(initialized_cnn_obj._image_preprocess(pillow_image), np.ndarray)


def test__image_preprocess_size(initialized_cnn_obj):
    """Give Pillow image and check that returned array has size=self.TARGET_SIZE"""
    pillow_image = Image.open(Path('tests/data/mixed_images/ukbench00120.jpg'))
    assert initialized_cnn_obj._image_preprocess(pillow_image).shape == (initialized_cnn_obj.TARGET_SIZE[0],
                                                                         initialized_cnn_obj.TARGET_SIZE[1], 3)


def test__convert_to_array_path(initialized_cnn_obj, mocker):
    path_dir = Path('tests/data/mixed_images/ukbench00120.jpg')
    mocker.patch.object(initialized_cnn_obj, '_image_preprocess')
    initialized_cnn_obj._convert_to_array(path_dir)
    initialized_cnn_obj._image_preprocess.assert_called_with(Image.open(path_dir))


def test__convert_to_array_array(initialized_cnn_obj, mocker):
    path_image = Path('tests/data/mixed_images/ukbench00120.jpg')
    im_arr = np.array(Image.open(path_image))
    expected_called = Image.fromarray(im_arr)

    mocker.patch.object(initialized_cnn_obj, '_image_preprocess')
    initialized_cnn_obj._convert_to_array(im_arr)
    initialized_cnn_obj._image_preprocess.assert_called_with(expected_called)


def test__convert_to_array_float_array(initialized_cnn_obj, mocker):
    path_image = Path('tests/data/mixed_images/ukbench00120.jpg')
    im_arr = np.array(Image.open(path_image)).astype('float32')
    expected_called = Image.fromarray(im_arr.astype('uint8'))

    mocker.patch.object(initialized_cnn_obj, '_image_preprocess')
    initialized_cnn_obj._convert_to_array(im_arr)
    initialized_cnn_obj._image_preprocess.assert_called_with(expected_called)


def test__get_parent_dir(initialized_cnn_obj):
    path_dir = Path('tests/data/mixed_images')
    assert initialized_cnn_obj._get_parent_dir(path_dir) == PosixPath('tests/data')


def test__get_sub_dir(initialized_cnn_obj):
    path_dir = Path('tests/data/mixed_images')
    assert initialized_cnn_obj._get_sub_dir(path_dir) == 'mixed_images'


def test_cnn_image_path(initialized_cnn_obj):
    path_image = Path('tests/data/mixed_images/ukbench00120.jpg')
    assert initialized_cnn_obj.cnn_image(path_image).shape == (1, 1000)


def test_cnn_image_path_nonempty(initialized_cnn_obj):
    path_image = Path('tests/data/mixed_images/ukbench00120.jpg')
    assert initialized_cnn_obj.cnn_image(path_image) is not None


def test_cnn_image_arr(initialized_cnn_obj):
    path_image = Path('tests/data/mixed_images/ukbench00120.jpg')
    im_arr = np.array(Image.open(path_image)).astype('float32')
    assert initialized_cnn_obj.cnn_image(im_arr).shape == (1, 1000)


def test_cnn_image_arr_nonempty(initialized_cnn_obj):
    path_image = Path('tests/data/mixed_images/ukbench00120.jpg')
    im_arr = np.array(Image.open(path_image)).astype('float32')
    assert initialized_cnn_obj.cnn_image(im_arr) is not None


def test__generator(initialized_cnn_obj):
    path_dir = Path('tests/data/mixed_images')
    gen = initialized_cnn_obj._generator(path_dir)
    assert gen.batch_size == initialized_cnn_obj.BATCH_SIZE
    assert gen.directory == Path('tests/data/')
    assert gen.target_size == initialized_cnn_obj.TARGET_SIZE
    assert not gen.shuffle
    assert len(np.unique(gen.classes)) == 1


def test_cnn_dir(initialized_cnn_obj):
    path_dir = Path('tests/data/mixed_images')
    dict_ret = initialized_cnn_obj.cnn_dir(path_dir)
    assert type(dict_ret) == dict
    expected_set = set(
        ['ukbench00120.jpg', 'ukbench00120_hflip.jpg', 'ukbench00120_vflip.jpg', 'ukbench00120_resize.jpg',
         'ukbench00120_rotation.jpg', 'ukbench09268.jpg'])
    assert len(set(dict_ret.keys()).intersection(expected_set)) == 5
    assert dict_ret['ukbench00120.jpg'].shape == (1000,)
    assert dict_ret['ukbench00120_hflip.jpg'] is not None


def test__get_file_mapping_feat_vec():
    pass


def test__get_only_filenames():
    pass


def test__find_duplicates_dict():
    pass


def test__find_duplicates_dir():
    pass


def test__check_threshold_bounds():
    pass


def test_find_duplicates():
    pass