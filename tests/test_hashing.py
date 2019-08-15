from imagededup.methods.hashing import Hashing, PHash, DHash, AHash, WHash  #, HashedDataset, Dataset
import os
import pytest
import numpy as np
from pathlib import Path
from PIL import Image

""" Run from project root with: python -m pytest -vs tests/test_hashing.py"""

PATH_IMAGE_DIR = Path('tests/data/mixed_images')
PATH_SINGLE_IMAGE = Path('tests/data/mixed_images/ukbench00120.jpg')
PATH_SINGLE_IMAGE_RESIZED = Path('tests/data/mixed_images/ukbench00120_resize.jpg')


# Test parent class (static methods/class attributes initialization)


@pytest.fixture(scope='module')
def hasher():
    hashobj = Hashing()
    return hashobj


def test_correct_init_hashing(hasher):
    assert hasher.target_size == (8, 8)


def test_hamming_distance(hasher):
    # Put two numbers and check if hamming distance is correct
    number_1 = "1a"
    number_2 = "1f"
    hamdist = hasher.hamming_distance(number_1, number_2)
    assert hamdist == 2


def test__array_to_hash(hasher):
    hash_mat = np.array([1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0])
    assert hasher._array_to_hash(hash_mat) == '9191fa'


def test__check_hamming_distance_bounds_input_not_int(hasher):
    with pytest.raises(TypeError):
        hasher._check_hamming_distance_bounds(thresh=1.0)


def test__check_hamming_distance_bounds_out_of_bound(hasher):
    with pytest.raises(ValueError):
        hasher._check_hamming_distance_bounds(thresh=68)


def test__check_hamming_distance_bounds_correct(hasher):
    assert  hasher._check_hamming_distance_bounds(thresh=20) is None

# encode_image


def test_encode_image_accepts_image_path(mocker, hasher):
    ret_val = np.zeros((2, 2))
    load_image_mocker = mocker.patch('imagededup.methods.hashing.load_image', return_value=ret_val,
                                     autospec=True)
    hash_func_mocker = mocker.patch('imagededup.methods.hashing.Hashing._hash_func', return_value=ret_val)
    hasher.encode_image(image_file=PATH_SINGLE_IMAGE)
    load_image_mocker.assert_called_with(image_file=PATH_SINGLE_IMAGE, grayscale=True,  target_size=(8, 8))
    np.testing.assert_array_equal(ret_val, hash_func_mocker.call_args[0][0])


def test_encode_image_accepts_numpy_array(mocker, hasher):
    ret_val = np.zeros((2, 2))
    preprocess_image_mocker = mocker.patch('imagededup.methods.hashing.preprocess_image', return_value=ret_val)
    hash_func_mocker = mocker.patch('imagededup.methods.hashing.Hashing._hash_func', return_value=ret_val)
    hasher.encode_image(image_array=ret_val)
    preprocess_image_mocker.assert_called_with(image=ret_val, target_size=(8, 8), grayscale=True)
    np.testing.assert_array_equal(ret_val, hash_func_mocker.call_args[0][0])


def test_encode_image_valerror_wrong_input(hasher):
    pil_im = Image.open(PATH_SINGLE_IMAGE)
    with pytest.raises(ValueError):
        hasher.encode_image(image_file=pil_im)


def test_encode_image_valerror_wrong_input_array(hasher):
    pil_im = Image.open(PATH_SINGLE_IMAGE)
    with pytest.raises(ValueError):
        hasher.encode_image(image_array=pil_im)


def test_encode_image_returns_none_image_pp_not_array(hasher, mocker):
    mocker.patch('imagededup.methods.hashing.load_image', return_value=None)
    assert hasher.encode_image(PATH_SINGLE_IMAGE) is None


def test_encode_image_returns_none_image_pp_not_array_array_input(hasher, mocker):
    mocker.patch('imagededup.methods.hashing.preprocess_image', return_value=None)
    assert hasher.encode_image(image_array=np.zeros((2, 2))) is None


# encode_images

def test_encode_images_accepts_valid_posixpath(hasher, mocker):
    mocker.patch('imagededup.methods.hashing.Hashing.encode_image', return_value='123456789ABCDEFA')
    assert len(hasher.encode_images(PATH_IMAGE_DIR)) == 5  # 5 files in the directory


def test_encode_images_rejects_non_posixpath(hasher):
    with pytest.raises(ValueError):
        hasher.encode_images('tests/data/base_images')


def test_encode_images_rejects_non_directory_paths(hasher):
    with pytest.raises(ValueError):
        hasher.encode_images(PATH_SINGLE_IMAGE)


def test_encode_images_return_vals(hasher, mocker):
    encoded_val = '123456789ABCDEFA'
    mocker.patch('imagededup.methods.hashing.Hashing.encode_image', return_value=encoded_val)
    hashes = hasher.encode_images(PATH_IMAGE_DIR)
    assert isinstance(hashes, dict)
    assert list(hashes.values())[0] == encoded_val
    assert PATH_SINGLE_IMAGE.name in hashes.keys()


def test_hash_func(hasher, mocker):
    inp_array = np.array((3, 3))
    ret_arr = np.array((2, 2))
    hash_algo_mocker = mocker.patch('imagededup.methods.hashing.Hashing._hash_algo', return_value=ret_arr)
    array_mocker = mocker.patch('imagededup.methods.hashing.Hashing._array_to_hash')
    hasher._hash_func(inp_array)
    np.testing.assert_array_equal(inp_array, hash_algo_mocker.call_args[0][0])
    array_mocker.assert_called_with(ret_arr)

# _find_duplicates_dict


def test__find_duplicates_dict_outfile_none(hasher, mocker):
    encoding_map = {'1.jpg': '123456'}
    threshold = 10
    scores = True
    outfile = None
    hasheval_mocker = mocker.patch('imagededup.methods.hashing.HashEval')
    save_json_mocker = mocker.patch('imagededup.methods.hashing.save_json')
    hasher._find_duplicates_dict(encoding_map=encoding_map, threshold=threshold, scores=scores, outfile=outfile)
    hasheval_mocker.assert_called_with(test=encoding_map, queries=encoding_map, hammer=Hashing.hamming_distance,
                                 cutoff=threshold, search_method='bktree')
    hasheval_mocker.return_value.retrieve_results.assert_called_once_with(scores=scores)
    save_json_mocker.assert_not_called()


def test__find_duplicates_dict_outfile_true(hasher, mocker):
    encoding_map = {'1.jpg': '123456'}
    threshold = 10
    scores = True
    outfile = True
    hasheval_mocker = mocker.patch('imagededup.methods.hashing.HashEval')
    hasheval_mocker.return_value.retrieve_results.return_value = {'filename.jpg': [('dup1.jpg', 3)], 'filename2.jpg':
                                                                                  [('dup2.jpg', 10)]}
    save_json_mocker = mocker.patch('imagededup.methods.hashing.save_json')
    hasher._find_duplicates_dict(encoding_map=encoding_map, threshold=threshold, scores=scores, outfile=outfile)
    hasheval_mocker.assert_called_with(test=encoding_map, queries=encoding_map, hammer=Hashing.hamming_distance,
                                 cutoff=threshold, search_method='bktree')
    hasheval_mocker.return_value.retrieve_results.assert_called_once_with(scores=scores)
    save_json_mocker.assert_called_once_with(hasheval_mocker.return_value.retrieve_results.return_value, outfile)


# _find_duplicates_dir

def test__find_duplicates_dir(hasher, mocker):
    encoding_map = {'1.jpg': '123456'}
    threshold = 10
    scores = True
    outfile = True
    ret_val_find_dup_dict = {'filename.jpg': [('dup1.jpg', 3)], 'filename2.jpg':[('dup2.jpg', 10)]}
    encode_images_mocker = mocker.patch('imagededup.methods.hashing.Hashing.encode_images', return_value=encoding_map)
    find_dup_dict_mocker = mocker.patch('imagededup.methods.hashing.Hashing._find_duplicates_dict',
                                        return_value=ret_val_find_dup_dict)
    hasher._find_duplicates_dir(image_dir=PATH_IMAGE_DIR, threshold= threshold, scores=scores, outfile=outfile)
    encode_images_mocker.assert_called_once_with(PATH_IMAGE_DIR)
    find_dup_dict_mocker.assert_called_once_with(encoding_map=encoding_map, threshold=threshold, scores=scores,
                                                 outfile=outfile)


# find_duplicates

def test_find_duplicates_dir(hasher, mocker):
    threshold = 10
    scores = True
    outfile = True
    check_hamming_mocker = mocker.patch('imagededup.methods.hashing.Hashing._check_hamming_distance_bounds')
    find_dup_dir_mocker = mocker.patch('imagededup.methods.hashing.Hashing._find_duplicates_dir')
    hasher.find_duplicates(image_dir=PATH_IMAGE_DIR, threshold=threshold, outfile=outfile, scores=scores)
    check_hamming_mocker.assert_called_once_with(thresh=threshold)
    find_dup_dir_mocker.assert_called_once_with(image_dir=PATH_IMAGE_DIR, threshold=threshold, scores=scores,
                                                outfile=outfile)


def test_find_duplicates_dict(hasher, mocker):
    encoding_map = {'1.jpg': '123456'}
    threshold = 10
    scores = True
    outfile = True
    check_hamming_mocker = mocker.patch('imagededup.methods.hashing.Hashing._check_hamming_distance_bounds')
    find_dup_dict_mocker = mocker.patch('imagededup.methods.hashing.Hashing._find_duplicates_dict')
    hasher.find_duplicates(encoding_map=encoding_map, threshold=threshold, outfile=outfile, scores=scores)
    check_hamming_mocker.assert_called_once_with(thresh=threshold)
    find_dup_dict_mocker.assert_called_once_with(encoding_map=encoding_map, threshold=threshold, scores=scores,
                                                outfile=outfile)


def test_find_duplicates_wrong_input(hasher):
    with pytest.raises(ValueError):
        hasher.find_duplicates(threshold=10)

# find_duplicates_to_remove


def test_find_duplicates_to_remove_outfile_false(hasher, mocker):
    threshold = 10
    outfile = False
    ret_val_find_dup_dict = {'filename.jpg': [('dup1.jpg', 3)], 'filename2.jpg': [('dup2.jpg', 10)]}
    find_duplicates_mocker = mocker.patch('imagededup.methods.hashing.Hashing.find_duplicates',
                                          return_value=ret_val_find_dup_dict)
    get_files_to_remove_mocker = mocker.patch('imagededup.methods.hashing.get_files_to_remove')
    save_json_mocker = mocker.patch('imagededup.methods.hashing.save_json')
    hasher.find_duplicates_to_remove(image_dir=PATH_IMAGE_DIR, threshold=threshold, outfile=outfile)
    find_duplicates_mocker.assert_called_once_with(image_dir=PATH_IMAGE_DIR, encoding_map=None, threshold=threshold,
                                                   scores=False)
    get_files_to_remove_mocker.assert_called_once_with(ret_val_find_dup_dict)
    save_json_mocker.assert_not_called()


def test_find_duplicates_to_remove_outfile_true(hasher, mocker):
    threshold = 10
    outfile = True
    ret_val_find_dup_dict = {'filename.jpg': [('dup1.jpg', 3)], 'filename2.jpg': [('dup2.jpg', 10)]}
    ret_val_get_files_to_remove = ['1.jpg','2.jpg']
    find_duplicates_mocker = mocker.patch('imagededup.methods.hashing.Hashing.find_duplicates',
                                          return_value=ret_val_find_dup_dict)
    get_files_to_remove_mocker = mocker.patch('imagededup.methods.hashing.get_files_to_remove',
                                              return_value=ret_val_get_files_to_remove)
    save_json_mocker = mocker.patch('imagededup.methods.hashing.save_json')
    hasher.find_duplicates_to_remove(image_dir=PATH_IMAGE_DIR, threshold=threshold, outfile=outfile)
    find_duplicates_mocker.assert_called_once_with(image_dir=PATH_IMAGE_DIR, encoding_map=None, threshold=threshold,
                                                   scores=False)
    get_files_to_remove_mocker.assert_called_once_with(ret_val_find_dup_dict)
    save_json_mocker.assert_called_once_with(ret_val_get_files_to_remove, outfile)


def test_find_duplicates_to_remove_encoding_map(hasher, mocker):
    encoding_map = {'1.jpg': '123456'}
    threshold = 10
    outfile = False
    ret_val_find_dup_dict = {'filename.jpg': [('dup1.jpg', 3)], 'filename2.jpg': [('dup2.jpg', 10)]}
    find_duplicates_mocker = mocker.patch('imagededup.methods.hashing.Hashing.find_duplicates',
                                          return_value=ret_val_find_dup_dict)
    get_files_to_remove_mocker = mocker.patch('imagededup.methods.hashing.get_files_to_remove')
    save_json_mocker = mocker.patch('imagededup.methods.hashing.save_json')
    hasher.find_duplicates_to_remove(encoding_map=encoding_map, threshold=threshold, outfile=outfile)
    find_duplicates_mocker.assert_called_once_with(encoding_map=encoding_map, image_dir=None, threshold=threshold,
                                                   scores=False)
    get_files_to_remove_mocker.assert_called_once_with(ret_val_find_dup_dict)
    save_json_mocker.assert_not_called()


# For all methods, test encode_image and encode_images

phasher = PHash()
dhasher = DHash()
ahasher = AHash()
whasher = WHash()

common_test_parameters = [phasher.encode_image, dhasher.encode_image, ahasher.encode_image, whasher.encode_image]


@pytest.mark.parametrize('hash_function', common_test_parameters)
class TestCommon:
    def test_len_hash(self, hash_function):
        hash_im = hash_function(PATH_SINGLE_IMAGE)
        assert len(hash_im) == 16

    def test_hash_resize(self, hash_function):
        # Resize one image to (300, 300) and check that hamming distance between hashes is not too large
        hash_im_1 = hash_function(PATH_SINGLE_IMAGE)
        hash_im_2 = hash_function(PATH_SINGLE_IMAGE_RESIZED)
        hamdist = Hashing.hamming_distance(hash_im_1, hash_im_2)
        assert hamdist < 3

    def test_hash_small_rotation(self, hash_function):
        # Rotate image slightly (1 degree) and check that hamming distance between hashes is not too large
        orig_image = Image.open(PATH_SINGLE_IMAGE)
        rotated_image = np.array(orig_image.rotate(1))
        hash_im_1 = hash_function(image_array=np.array(orig_image))
        hash_im_2 = hash_function(image_array=rotated_image)
        hamdist = Hashing.hamming_distance(hash_im_1, hash_im_2)
        assert hamdist < 3

    def test_hash_distinct_images(self, hash_function):
        # Put in distinct images and check that hamming distance between hashes is large
        hash_im_1 = hash_function(PATH_SINGLE_IMAGE)
        hash_im_2 = hash_function(Path('tests/data/mixed_images/ukbench09268.jpg'))
        hamdist = Hashing.hamming_distance(hash_im_1, hash_im_2)
        assert hamdist > 20


def test_encode_images_returns_dict():
    hash_dict = phasher.encode_images(PATH_IMAGE_DIR)
    assert isinstance(hash_dict, dict)


def test_encode_images_return_non_none_hashes():
    hash_dict = dhasher.encode_images(PATH_IMAGE_DIR)
    for v in hash_dict.values():
        assert v is not None



"""
@pytest.fixture(scope='module')
def initialized_hash_obj():
    hashobj = Hashing()
    return hashobj


def test__find_duplicates_dict_scores_true(initialized_hash_obj):
    # check correctness, check that result_score has dictionary, check return dict

    dummy_hashes = {
        'ukbench09060.jpg': 'e064ece078d7c96a',
        'ukbench09060_dup.jpg': 'd064ece078d7c96a',
        'ukbench09061.jpg': 'e051ece099d7faea',
        'ukbench09062.jpg': 'd465fd2078d8936c',
    }

    assert initialized_hash_obj.result_score is None

    dict_ret = initialized_hash_obj._find_duplicates_dict(dummy_hashes, threshold=10, scores=True)
    assert type(dict_ret['ukbench09060.jpg']) == dict
    assert set(dict_ret['ukbench09060.jpg'].keys()) == set(['ukbench09060_dup.jpg'])
    assert initialized_hash_obj.result_score == dict_ret


def test__find_duplicates_dict_scores_false():
    initialized_hash_obj = Hashing()
    # check correctness, check that result_score has dictionary, check return dict

    dummy_hashes = {
        'ukbench09060.jpg': 'e064ece078d7c96a',
        'ukbench09060_dup.jpg': 'd064ece078d7c96a',
        'ukbench09061.jpg': 'e051ece099d7faea',
        'ukbench09062.jpg': 'd465fd2078d8936c',
    }

    assert initialized_hash_obj.result_score is None
    dict_ret = initialized_hash_obj._find_duplicates_dict(dummy_hashes, threshold=10, scores=False)
    assert dict_ret['ukbench09060.jpg'] == ['ukbench09060_dup.jpg']
    assert initialized_hash_obj.result_score is not None


def test__find_duplicates_dir(initialized_hash_obj):
    path_dir = Path('tests/data/mixed_images')
    dict_ret = initialized_hash_obj._find_duplicates_dir(path_dir=path_dir)
    assert dict_ret is not None


def test__check_hamming_distance_bounds_input_not_int(initialized_hash_obj):
    with pytest.raises(TypeError):
        initialized_hash_obj._check_hamming_distance_bounds(thresh=1.0)


def test_find_duplicates_path(initialized_hash_obj, mocker):
    path_dir = Path('tests/data/mixed_images')
    threshold = 10
    mocker.patch.object(initialized_hash_obj, '_check_hamming_distance_bounds')
    mocker.patch.object(initialized_hash_obj, '_find_duplicates_dir')
    initialized_hash_obj.find_duplicates(path_dir, threshold=threshold)
    initialized_hash_obj._check_hamming_distance_bounds.assert_called_with(thresh=threshold)
    initialized_hash_obj._find_duplicates_dir.assert_called_with(path_dir=path_dir, scores=False, threshold=threshold)


def test_find_duplicates_dict(initialized_hash_obj, mocker):
    dummy_hashes = {
        'ukbench09060.jpg': 'e064ece078d7c96a',
        'ukbench09060_dup.jpg': 'd064ece078d7c96a',
        'ukbench09061.jpg': 'e051ece099d7faea',
        'ukbench09062.jpg': 'd465fd2078d8936c',
    }

    threshold = 10
    mocker.patch.object(initialized_hash_obj, '_check_hamming_distance_bounds')
    mocker.patch.object(initialized_hash_obj, '_find_duplicates_dict')
    initialized_hash_obj.find_duplicates(dummy_hashes, threshold=threshold)
    initialized_hash_obj._check_hamming_distance_bounds.assert_called_with(thresh=threshold)
    initialized_hash_obj._find_duplicates_dict.assert_called_with(dict_file_feature=dummy_hashes, scores=False,
                                                                  threshold=threshold)


def test__check_hamming_distance_bounds_input_out_of_range(initialized_hash_obj):
    with pytest.raises(TypeError):
        initialized_hash_obj._check_hamming_distance_bounds(thresh=65)


def test_find_duplicates_unacceptable_input(initialized_hash_obj):
    with pytest.raises(TypeError):
        initialized_hash_obj.find_duplicates('tests/data/mixed_images')


def test_retrieve_dups(initialized_hash_obj, monkeypatch):
    def mock_find_duplicates(path_or_dict, threshold, scores):
        dict_ret = {'1': ['3', '4'], '2': ['3'], '3': ['1'], '4': ['1']}
        return dict_ret

    monkeypatch.setattr(initialized_hash_obj, 'find_duplicates', mock_find_duplicates)
    list_to_rem = initialized_hash_obj.find_duplicates_to_remove(path_or_dict=dict())
    assert set(list_to_rem) == set(['3', '4'])


def test_dict_dir_same_results(initialized_hash_obj):
    path_dir = Path('tests/data/base_images')
    dup_dir = initialized_hash_obj._find_duplicates_dir(path_dir=path_dir)
    dict_hash = initialized_hash_obj.hash_dir(path_dir)
    dup_dict = initialized_hash_obj._find_duplicates_dict(dict_hash)
    assert dup_dir == dup_dict
"""