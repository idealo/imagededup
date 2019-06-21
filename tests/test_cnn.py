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


def test__get_file_mapping_feat_vec(initialized_cnn_obj):
    expected_featvec = np.array([1, 0, 0, 1])
    expected_filename = {0: 'ukbench00002.jpg'}
    in_dict = {expected_filename[i]: expected_featvec for i in range(len(expected_filename))}
    out_feat_vec, out_filename = initialized_cnn_obj._get_file_mapping_feat_vec(in_dict)
    np.testing.assert_array_equal(out_feat_vec[0], expected_featvec)
    assert out_filename == expected_filename


def test__get_file_mapping_feat_vec_order_correctness(initialized_cnn_obj):
    expected_featvec = np.array([[1, 0, 0, 1], [1, 1, 0, 1], [1, 0, 1, 1], [1, 1, 1, 1]])
    expected_filename = {0: 'ukbench00002.jpg', 1: 'ukbench00003.jpg', 2: 'ukbench00004.jpg', 3: 'ukbench00005.jpg'}
    in_dict = {expected_filename[i]: expected_featvec[i] for i in range(len(expected_filename))}
    out_feat_vec, out_filename = initialized_cnn_obj._get_file_mapping_feat_vec(in_dict)
    for i in out_filename.keys():
        np.testing.assert_array_equal(in_dict[out_filename[i]], out_feat_vec[i])


def test__get_only_filenames(initialized_cnn_obj):
    dict_with_scores = {'ukbench00002.jpg': {'ukbench00002_hflip.jpg': 0.9999999, 'ukbench00002_resize.jpg': 0.96390784,
                          'ukbench00002_vflip.jpg': 0.9600804},'ukbench00002_cropped.jpg': {'ukbench00002_hflip.jpg': 1.0}}
    dict_ret = initialized_cnn_obj._get_only_filenames(dict_with_scores)
    assert set(dict_ret['ukbench00002.jpg']) == set(dict_with_scores['ukbench00002.jpg'].keys())


def test__find_duplicates_dict_scores_true(initialized_cnn_obj):
    # check correctness, check that result_score has dictionary, check return dict
    expected_featvec = [np.array([1, 0, 0, 1]), np.array([1, 1, 0, 1]), np.array([1, 0, 0, 1])]
    expected_filename = {0: 'ukbench00002.jpg', 1: 'ukbench00003.jpg', 2: 'ukbench00002_dup.jpg'}
    in_dict = {expected_filename[i]: expected_featvec[i] for i in range(len(expected_filename))}
    assert initialized_cnn_obj.result_score is None
    dict_ret = initialized_cnn_obj._find_duplicates_dict(in_dict, threshold=0.9, scores=True)
    assert type(dict_ret['ukbench00002.jpg']) == dict
    assert set(dict_ret['ukbench00002.jpg'].keys()) == set(['ukbench00002_dup.jpg'])
    assert initialized_cnn_obj.result_score == dict_ret


def test__find_duplicates_dict_scores_false():
    initialized_cnn_obj = cnn.CNN()
    # check correctness, check that result_score has dictionary, check return dict
    expected_featvec = [np.array([1, 0, 0, 1]), np.array([1, 1, 0, 1]), np.array([1, 0, 0, 1])]
    expected_filename = {0: 'ukbench00002.jpg', 1: 'ukbench00003.jpg', 2: 'ukbench00002_dup.jpg'}
    in_dict = {expected_filename[i]: expected_featvec[i] for i in range(len(expected_filename))}
    assert initialized_cnn_obj.result_score is None
    dict_ret = initialized_cnn_obj._find_duplicates_dict(in_dict, threshold=0.9, scores=False)
    assert dict_ret['ukbench00002.jpg'] == ['ukbench00002_dup.jpg']
    assert initialized_cnn_obj.result_score is not None


def test__find_duplicates_dir(initialized_cnn_obj, monkeypatch):
    path_dir = Path('tests/data/mixed_images')

    def mock_cnn_dir(path_dir):
        expected_featvec = [np.array([1, 0, 0, 1]), np.array([1, 1, 0, 1]), np.array([1, 0, 0, 1])]
        expected_filename = {0: 'ukbench00002.jpg', 1: 'ukbench00003.jpg', 2: 'ukbench00002_dup.jpg'}
        in_dict = {expected_filename[i]: expected_featvec[i] for i in range(len(expected_filename))}
        return in_dict

    monkeypatch.setattr(initialized_cnn_obj, 'cnn_dir', mock_cnn_dir)
    dict_ret = initialized_cnn_obj._find_duplicates_dir(path_dir, threshold=0.9)
    assert set(dict_ret['ukbench00002.jpg']) == set(['ukbench00002_dup.jpg'])


def test__check_threshold_bounds_input_not_float(initialized_cnn_obj):
    with pytest.raises(TypeError):
        initialized_cnn_obj._check_threshold_bounds(thresh=1)


def test__check_threshold_bounds_input_out_of_range(initialized_cnn_obj):
    with pytest.raises(TypeError):
        initialized_cnn_obj._check_threshold_bounds(thresh=1.1)


def test_find_duplicates_path(initialized_cnn_obj, mocker):
    path_dir = Path('tests/data/mixed_images')
    threshold = 0.8
    mocker.patch.object(initialized_cnn_obj, '_check_threshold_bounds')
    mocker.patch.object(initialized_cnn_obj, '_find_duplicates_dir')
    initialized_cnn_obj.find_duplicates(path_dir, threshold=threshold)
    initialized_cnn_obj._check_threshold_bounds.assert_called_with(thresh=threshold)
    initialized_cnn_obj._find_duplicates_dir.assert_called_with(path_dir=path_dir, scores=False, threshold=threshold)


def test_find_duplicates_dict(initialized_cnn_obj, mocker):
    expected_featvec = [np.array([1, 0, 0, 1]), np.array([1, 1, 0, 1]), np.array([1, 0, 0, 1])]
    expected_filename = {0: 'ukbench00002.jpg', 1: 'ukbench00003.jpg', 2: 'ukbench00002_dup.jpg'}
    in_dict = {expected_filename[i]: expected_featvec[i] for i in range(len(expected_filename))}

    threshold = 0.8
    mocker.patch.object(initialized_cnn_obj, '_check_threshold_bounds')
    mocker.patch.object(initialized_cnn_obj, '_find_duplicates_dict')
    initialized_cnn_obj.find_duplicates(in_dict, threshold=threshold)
    initialized_cnn_obj._check_threshold_bounds.assert_called_with(thresh=threshold)
    initialized_cnn_obj._find_duplicates_dict.assert_called_with(dict_file_feature=in_dict, scores=False, threshold=threshold)


def test_find_duplicates_unacceptable_input(initialized_cnn_obj):
    with pytest.raises(TypeError):
        initialized_cnn_obj.find_duplicates('tests/data/mixed_images')


def test_find_duplicates_to_remove(initialized_cnn_obj, monkeypatch):
    path_dir = Path('tests/data/mixed_images')

    def mock_find_duplicates(path_or_dict=path_dir, threshold=0.8, scores=False):
        from collections import OrderedDict
        dict_a = OrderedDict({'1': ['2'], '2': ['1', '3'], '3': ['4'], '4': ['3'], '5': []})
        return dict_a
    monkeypatch.setattr(initialized_cnn_obj, 'find_duplicates', mock_find_duplicates)
    dups_to_remove = initialized_cnn_obj.find_duplicates_to_remove(path_or_dict=path_dir)
    assert dups_to_remove == ['2', '4']

