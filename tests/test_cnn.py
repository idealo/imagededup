from pathlib import Path

import os
import json
import numpy as np
import pytest
from tensorflow.keras.models import Model

from imagededup.methods.cnn import CNN
from imagededup.utils.image_utils import load_image


p = Path(__file__)
TEST_IMAGE = p.parent / 'data' / 'base_images' / 'ukbench00120.jpg'
TEST_IMAGE_DIR = p.parent / 'data' / 'base_images'
TEST_IMAGE_FORMATS_DIR = p.parent / 'data' / 'formats_images'
TEST_IMAGE_DIR_MIXED = p.parent / 'data' / 'mixed_images'

TEST_BATCH_SIZE = 64
TEST_TARGET_SIZE = (224, 224)


def data_encoding_map():
    return {
        'ukbench00002.jpg': np.array([1, 0, 0, 1]),
        'ukbench00003.jpg': np.array([1, 1, 0, 1]),
        'ukbench00002_dup.jpg': np.array([1, 0, 0, 1]),
    }


@pytest.fixture(scope='module')
def cnn():
    return CNN()


@pytest.fixture
def mocker_save_json(mocker):
    return mocker.patch('imagededup.methods.cnn.save_json')


def test__init(cnn):
    assert cnn.batch_size == TEST_BATCH_SIZE
    assert cnn.target_size == TEST_TARGET_SIZE
    assert isinstance(cnn.model, Model)


def test__get_cnn_features_single(cnn):
    img = load_image(TEST_IMAGE, target_size=(224, 224))

    result = cnn._get_cnn_features_single(img)

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 1024)


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

    assert list(sorted(result.keys(), key=str.lower)) == expected_predicted_files

    for i in result.values():
        assert isinstance(i, np.ndarray)
        assert i.shape == (1024,)

    result = cnn._get_cnn_features_batch(TEST_IMAGE_FORMATS_DIR)

    expected_predicted_files = [
        'baboon.pgm',
        'copyleft.tiff',
        'giphy.gif',
        'Iggy.1024.ppm',
        'marbles.pbm',
        'mpo_image.MPO',
        'ukbench09380.bmp',
        'ukbench09380.jpeg',
        'ukbench09380.png',
        'ukbench09380.svg',
    ]

    assert list(sorted(result.keys(), key=str.lower)) == expected_predicted_files

    for i in result.values():
        assert isinstance(i, np.ndarray)
        assert i.shape == (1024,)


def test_encode_image(cnn):
    result = cnn.encode_image(TEST_IMAGE)

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 1024)  # 1024 = 3*3*1024*2

    result = cnn.encode_image(str(TEST_IMAGE))

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 1024)  # 1024 = 3*3*1024*2

    with pytest.raises(ValueError):
        cnn.encode_image("")

    image_array = load_image(TEST_IMAGE)
    result = cnn.encode_image(image_array=image_array)
    assert result.shape == (1, 1024)  # 1024 = 3*3*1024*2


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

    assert list(sorted(result.keys(), key=str.lower)) == expected_predicted_files

    for i in result.values():
        assert isinstance(i, np.ndarray)
        assert i.shape == (1024,)

    result = cnn.encode_images(TEST_IMAGE_FORMATS_DIR)

    expected_predicted_files = [
        'baboon.pgm',
        'copyleft.tiff',
        'giphy.gif',
        'Iggy.1024.ppm',
        'marbles.pbm',
        'mpo_image.MPO',
        'ukbench09380.bmp',
        'ukbench09380.jpeg',
        'ukbench09380.png',
        'ukbench09380.svg',
    ]

    assert list(sorted(result.keys(), key=str.lower)) == expected_predicted_files

    for i in result.values():
        assert isinstance(i, np.ndarray)
        assert i.shape == (1024,)

    result = cnn.encode_images(str(TEST_IMAGE_FORMATS_DIR))

    expected_predicted_files = [
        'baboon.pgm',
        'copyleft.tiff',
        'giphy.gif',
        'Iggy.1024.ppm',
        'marbles.pbm',
        'mpo_image.MPO',
        'ukbench09380.bmp',
        'ukbench09380.jpeg',
        'ukbench09380.png',
        'ukbench09380.svg',
    ]

    assert list(sorted(result.keys(), key=str.lower)) == expected_predicted_files

    for i in result.values():
        assert isinstance(i, np.ndarray)
        assert i.shape == (1024,)

    with pytest.raises(ValueError):
        cnn.encode_images('abc')


def test__check_threshold_bounds_input_not_float(cnn):
    with pytest.raises(TypeError):
        cnn._check_threshold_bounds(thresh=1)


def test__check_threshold_bounds_input_out_of_range(cnn):
    with pytest.raises(ValueError):
        cnn._check_threshold_bounds(thresh=1.1)


# _find_duplicates_dict


def test__find_duplicates_dict_scores_false(cnn):
    # check correctness
    encoding_map = data_encoding_map()
    dict_ret = cnn._find_duplicates_dict(
        encoding_map, min_similarity_threshold=0.9, scores=False
    )
    assert isinstance(dict_ret['ukbench00002.jpg'], list)
    assert len(dict_ret['ukbench00002.jpg']) == 1
    assert not isinstance(dict_ret['ukbench00002.jpg'][0], tuple)
    assert dict_ret['ukbench00002.jpg'][0] == 'ukbench00002_dup.jpg'


def test__find_duplicates_dict_scores_true(cnn, mocker_save_json):
    # check correctness, also check that saving file is not triggered as outfile default value is False
    encoding_map = data_encoding_map()
    dict_ret = cnn._find_duplicates_dict(
        encoding_map, min_similarity_threshold=0.9, scores=True
    )

    assert isinstance(dict_ret['ukbench00002.jpg'], list)
    assert len(dict_ret['ukbench00002.jpg']) == 1
    assert isinstance(dict_ret['ukbench00002.jpg'][0], tuple)
    assert dict_ret['ukbench00002.jpg'][0][0] == 'ukbench00002_dup.jpg'
    assert isinstance(dict_ret['ukbench00002.jpg'][0][1], float)
    np.testing.assert_almost_equal(dict_ret['ukbench00002.jpg'][0][1], 1.0)
    mocker_save_json.assert_not_called()


def test__find_duplicates_dict_outfile_true(cnn, mocker_save_json):
    encoding_map = data_encoding_map()
    threshold = 0.8
    scores = True
    outfile = True
    cnn._find_duplicates_dict(
        encoding_map=encoding_map,
        min_similarity_threshold=threshold,
        scores=scores,
        outfile=outfile,
    )
    mocker_save_json.assert_called_once_with(results=cnn.results, filename=outfile, float_scores=True)


# _find_duplicates_dir


def test__find_duplicates_dir(cnn, mocker):
    encoding_map = data_encoding_map()
    threshold = 0.8
    scores = True
    outfile = True
    ret_val_find_dup_dict = {
        'filename1.jpg': [('dup1.jpg', 0.82)],
        'filename2.jpg': [('dup2.jpg', 0.90)],
    }
    encode_images_mocker = mocker.patch('imagededup.methods.cnn.CNN.encode_images')
    cnn.encoding_map = encoding_map
    find_dup_dict_mocker = mocker.patch(
        'imagededup.methods.cnn.CNN._find_duplicates_dict',
        return_value=ret_val_find_dup_dict,
    )
    cnn._find_duplicates_dir(
        image_dir=TEST_IMAGE_DIR,
        min_similarity_threshold=threshold,
        scores=scores,
        outfile=outfile,
    )
    encode_images_mocker.assert_called_once_with(image_dir=TEST_IMAGE_DIR)
    find_dup_dict_mocker.assert_called_once_with(
        encoding_map=cnn.encoding_map,
        min_similarity_threshold=threshold,
        scores=scores,
        outfile=outfile,
    )


# find_duplicates


def test_find_duplicates_dir(cnn, mocker):
    threshold = 0.9
    scores = True
    outfile = True
    find_dup_dir_mocker = mocker.patch(
        'imagededup.methods.cnn.CNN._find_duplicates_dir'
    )
    cnn.find_duplicates(
        image_dir=TEST_IMAGE_DIR,
        min_similarity_threshold=threshold,
        outfile=outfile,
        scores=scores,
    )
    find_dup_dir_mocker.assert_called_once_with(
        image_dir=TEST_IMAGE_DIR,
        min_similarity_threshold=threshold,
        scores=scores,
        outfile=outfile,
    )


def test_find_duplicates_dict(cnn, mocker):
    encoding_map = data_encoding_map()
    threshold = 0.9
    scores = True
    outfile = True
    find_dup_dict_mocker = mocker.patch(
        'imagededup.methods.cnn.CNN._find_duplicates_dict'
    )
    cnn.find_duplicates(
        encoding_map=encoding_map,
        min_similarity_threshold=threshold,
        outfile=outfile,
        scores=scores,
    )
    find_dup_dict_mocker.assert_called_once_with(
        encoding_map=encoding_map,
        min_similarity_threshold=threshold,
        scores=scores,
        outfile=outfile,
    )


def test_find_duplicates_wrong_threhsold_input(cnn):
    with pytest.raises(ValueError):
        cnn.find_duplicates(min_similarity_threshold=1.3)


def test_find_duplicates_wrong_input(cnn):
    with pytest.raises(ValueError):
        cnn.find_duplicates()


# find_duplicates_to_remove


def test_find_duplicates_to_remove_outfile_false(cnn, mocker, mocker_save_json):
    threshold = 0.9
    outfile = False
    ret_val_find_dup_dict = {
        'filename.jpg': [('dup1.jpg', 3)],
        'filename2.jpg': [('dup2.jpg', 10)],
    }
    find_duplicates_mocker = mocker.patch(
        'imagededup.methods.cnn.CNN.find_duplicates', return_value=ret_val_find_dup_dict
    )
    get_files_to_remove_mocker = mocker.patch(
        'imagededup.methods.cnn.get_files_to_remove'
    )
    cnn.find_duplicates_to_remove(
        image_dir=TEST_IMAGE_DIR, min_similarity_threshold=threshold, outfile=outfile
    )
    find_duplicates_mocker.assert_called_once_with(
        image_dir=TEST_IMAGE_DIR,
        encoding_map=None,
        min_similarity_threshold=threshold,
        scores=False,
    )
    get_files_to_remove_mocker.assert_called_once_with(ret_val_find_dup_dict)
    mocker_save_json.assert_not_called()


def test_find_duplicates_to_remove_outfile_true(cnn, mocker, mocker_save_json):
    threshold = 0.9
    outfile = True
    ret_val_find_dup_dict = {
        'filename.jpg': ['dup1.jpg'],
        'filename2.jpg': ['dup2.jpg'],
    }
    ret_val_get_files_to_remove = ['1.jpg', '2.jpg']

    find_duplicates_mocker = mocker.patch(
        'imagededup.methods.cnn.CNN.find_duplicates', return_value=ret_val_find_dup_dict
    )
    get_files_to_remove_mocker = mocker.patch(
        'imagededup.methods.cnn.get_files_to_remove',
        return_value=ret_val_get_files_to_remove,
    )
    cnn.find_duplicates_to_remove(
        image_dir=TEST_IMAGE_DIR, min_similarity_threshold=threshold, outfile=outfile
    )
    find_duplicates_mocker.assert_called_once_with(
        image_dir=TEST_IMAGE_DIR,
        encoding_map=None,
        min_similarity_threshold=threshold,
        scores=False,
    )
    get_files_to_remove_mocker.assert_called_once_with(ret_val_find_dup_dict)
    mocker_save_json.assert_called_once_with(ret_val_get_files_to_remove, outfile)


def test_find_duplicates_to_remove_encoding_map(cnn, mocker, mocker_save_json):
    threshold = 0.9
    outfile = True
    ret_val_find_dup_dict = {
        'filename.jpg': ['dup1.jpg'],
        'filename2.jpg': ['dup2.jpg'],
    }
    ret_val_get_files_to_remove = ['1.jpg', '2.jpg']
    encoding_map = data_encoding_map()
    find_duplicates_mocker = mocker.patch(
        'imagededup.methods.cnn.CNN.find_duplicates', return_value=ret_val_find_dup_dict
    )
    get_files_to_remove_mocker = mocker.patch(
        'imagededup.methods.cnn.get_files_to_remove',
        return_value=ret_val_get_files_to_remove,
    )
    cnn.find_duplicates_to_remove(
        encoding_map=encoding_map, min_similarity_threshold=threshold, outfile=outfile
    )
    find_duplicates_mocker.assert_called_once_with(
        encoding_map=encoding_map,
        image_dir=None,
        min_similarity_threshold=threshold,
        scores=False,
    )
    get_files_to_remove_mocker.assert_called_once_with(ret_val_find_dup_dict)
    mocker_save_json.assert_called_once_with(ret_val_get_files_to_remove, outfile)


# Integration tests


# test find_duplicates with directory path
def test_find_duplicates_dir_integration(cnn):
    expected_duplicates = {
        'ukbench00120.jpg': [
            ('ukbench00120_hflip.jpg', 0.9672552),
            ('ukbench00120_resize.jpg', 0.98120844),
        ],
        'ukbench00120_hflip.jpg': [
            ('ukbench00120.jpg', 0.9672552),
            ('ukbench00120_resize.jpg', 0.95676106),
        ],
        'ukbench00120_resize.jpg': [
            ('ukbench00120.jpg', 0.98120844),
            ('ukbench00120_hflip.jpg', 0.95676106),
        ],
        'ukbench00120_rotation.jpg': [],
        'ukbench09268.jpg': [],
    }
    duplicates = cnn.find_duplicates(
        image_dir=TEST_IMAGE_DIR_MIXED,
        min_similarity_threshold=0.9,
        scores=True,
        outfile=False,
    )
    # verify variable type
    assert isinstance(duplicates['ukbench00120.jpg'][0][1], np.float32)

    # verify that all files have been considered for deduplication
    assert len(duplicates) == len(expected_duplicates)

    # verify for each file that expected files have been received as duplicates
    for k in duplicates.keys():
        dup_val = duplicates[k]
        expected_val = expected_duplicates[k]
        dup_ret = set(map(lambda x: x[0], dup_val))
        expected_ret = set(map(lambda x: x[0], expected_val))
        assert dup_ret == expected_ret


# test find_duplicates with encoding map
def test_find_duplicates_encoding_integration(cnn):
    expected_duplicates = {
        'ukbench00120.jpg': [
            ('ukbench00120_hflip.jpg', 0.9672552),
            ('ukbench00120_resize.jpg', 0.98120844),
        ],
        'ukbench00120_hflip.jpg': [
            ('ukbench00120.jpg', 0.9672552),
            ('ukbench00120_resize.jpg', 0.95676106),
        ],
        'ukbench00120_resize.jpg': [
            ('ukbench00120.jpg', 0.98120844),
            ('ukbench00120_hflip.jpg', 0.95676106),
        ],
        'ukbench00120_rotation.jpg': [],
        'ukbench09268.jpg': [],
    }

    encodings = cnn.encode_images(TEST_IMAGE_DIR_MIXED)
    duplicates = cnn.find_duplicates(
        encoding_map=encodings, min_similarity_threshold=0.9, scores=True, outfile=False
    )
    # verify variable type
    assert isinstance(duplicates['ukbench00120.jpg'][0][1], np.float32)

    # verify that all files have been considered for deduplication
    assert len(duplicates) == len(expected_duplicates)

    # verify for each file that expected files have been received as duplicates
    for k in duplicates.keys():
        dup_val = duplicates[k]
        expected_val = expected_duplicates[k]
        dup_ret = set(map(lambda x: x[0], dup_val))
        expected_ret = set(map(lambda x: x[0], expected_val))
        assert dup_ret == expected_ret


# test find_duplicates_to_remove with directory path
def test_find_duplicates_to_remove_dir_integration(cnn):
    duplicates_list = cnn.find_duplicates_to_remove(
        image_dir=TEST_IMAGE_DIR_MIXED, min_similarity_threshold=0.9, outfile=False
    )
    assert set(duplicates_list) == set(
        ['ukbench00120_resize.jpg', 'ukbench00120_hflip.jpg']
    )


# test find_duplicates_to_remove with encoding map
def test_find_duplicates_to_remove_encoding_integration(cnn):
    encodings = cnn.encode_images(TEST_IMAGE_DIR_MIXED)
    duplicates_list = cnn.find_duplicates_to_remove(
        encoding_map=encodings, min_similarity_threshold=0.9, outfile=False
    )
    assert set(duplicates_list) == set(
        ['ukbench00120_resize.jpg', 'ukbench00120_hflip.jpg']
    )


# test verbose
def test_encode_images_verbose_true(capsys):
    cnn = CNN(verbose=True)
    cnn.encode_images(image_dir=TEST_IMAGE_DIR)
    out, err = capsys.readouterr()

    assert '[==============================]' in out
    assert '' == err


def test_encode_images_verbose_false(capsys):
    cnn = CNN(verbose=False)
    cnn.encode_images(image_dir=TEST_IMAGE_DIR)
    out, err = capsys.readouterr()

    assert '' == out
    assert '' == err


def test_find_duplicates_verbose_true(capsys):
    cnn = CNN(verbose=True)
    cnn.find_duplicates(
        image_dir=TEST_IMAGE_DIR,
        min_similarity_threshold=0.8,
        scores=False,
        outfile=False,
    )
    out, err = capsys.readouterr()

    assert '[==============================]' in out
    assert '' == err


def test_find_duplicates_verbose_false(capsys):
    cnn = CNN(verbose=False)
    cnn.find_duplicates(
        image_dir=TEST_IMAGE_DIR,
        min_similarity_threshold=0.8,
        scores=False,
        outfile=False,
    )
    out, err = capsys.readouterr()

    assert '' == out
    assert '' == err


def test_scores_saving(cnn):
    save_file = 'myduplicates.json'
    cnn.find_duplicates(
        image_dir=TEST_IMAGE_DIR_MIXED,
        min_similarity_threshold=0.6,
        scores=True,
        outfile=save_file,
    )
    with open(save_file, 'r') as f:
        saved_json = json.load(f)

    assert len(saved_json) == 5  # all valid files present as keys
    assert len(saved_json['ukbench00120.jpg']) == 3  # file with duplicates have all entries
    assert len(saved_json['ukbench09268.jpg']) == 0  # file with no duplicates have no entries
    assert isinstance(saved_json['ukbench00120.jpg'], list)  # a list of files is returned
    assert isinstance(saved_json['ukbench00120.jpg'][0], list) # each entry in the duplicate list is a list (not a tuple, since json can't save tuples)
    assert isinstance(saved_json['ukbench00120.jpg'][0][1], float) # saved score is of type 'float'

    os.remove(save_file)  # clean up

