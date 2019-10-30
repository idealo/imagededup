import os
import sys
from pathlib import Path
from PIL import Image

import pytest
import numpy as np

from imagededup.methods.hashing import Hashing, PHash, DHash, AHash, WHash

p = Path(__file__)

PATH_IMAGE_DIR = p.parent / 'data/mixed_images'
PATH_IMAGE_DIR_STRING = os.path.join(os.getcwd(), 'tests/data/mixed_images')
PATH_SINGLE_IMAGE = p.parent / 'data/mixed_images/ukbench00120.jpg'
PATH_SINGLE_IMAGE_STRING = p.parent / 'data/mixed_images/ukbench00120.jpg'
PATH_SINGLE_IMAGE_CORRUPT = p.parent / 'data/mixed_images/ukbench09268_corrupt.jpg'
PATH_SINGLE_IMAGE_RESIZED = p.parent / 'data/mixed_images/ukbench00120_resize.jpg'


# Test parent class (static methods/class attributes initialization)


@pytest.fixture
def hasher():
    hashobj = Hashing()
    return hashobj


def test_correct_init_hashing(hasher):
    assert hasher.target_size == (8, 8)


def test_hamming_distance(hasher):
    # Put two numbers and check if hamming distance is correct
    number_1 = '1a'
    number_2 = '1f'
    hamdist = hasher.hamming_distance(number_1, number_2)
    assert hamdist == 2


def test__array_to_hash(hasher):
    hash_mat = np.array(
        [1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0]
    )
    assert hasher._array_to_hash(hash_mat) == '9191fa'


def test__check_hamming_distance_bounds_input_not_int(hasher):
    with pytest.raises(TypeError):
        hasher._check_hamming_distance_bounds(thresh=1.0)


def test__check_hamming_distance_bounds_out_of_bound(hasher):
    with pytest.raises(ValueError):
        hasher._check_hamming_distance_bounds(thresh=68)


def test__check_hamming_distance_bounds_correct(hasher):
    assert hasher._check_hamming_distance_bounds(thresh=20) is None


# encode_image


@pytest.fixture
def mocker_preprocess_image(mocker):
    ret_val = np.zeros((2, 2))
    preprocess_image_mocker = mocker.patch(
        'imagededup.methods.hashing.preprocess_image', return_value=ret_val
    )
    return preprocess_image_mocker


@pytest.fixture
def mocker_hash_func(mocker):
    ret_val = np.zeros((2, 2))
    hash_func_mocker = mocker.patch(
        'imagededup.methods.hashing.Hashing._hash_func', return_value=ret_val
    )
    return hash_func_mocker


@pytest.fixture
def mocker_load_image(mocker):
    ret_val = np.zeros((2, 2))
    load_image_mocker = mocker.patch(
        'imagededup.methods.hashing.load_image', return_value=ret_val, autospec=True
    )
    return load_image_mocker


def test_encode_image_accepts_image_posixpath(
    hasher, mocker_load_image, mocker_hash_func
):
    ret_val = np.zeros((2, 2))
    hasher.encode_image(image_file=PATH_SINGLE_IMAGE)
    mocker_load_image.assert_called_with(
        image_file=PATH_SINGLE_IMAGE, grayscale=True, target_size=(8, 8)
    )
    np.testing.assert_array_equal(ret_val, mocker_hash_func.call_args[0][0])


def test_encode_image_accepts_numpy_array(
    hasher, mocker_preprocess_image, mocker_hash_func
):
    ret_val = np.zeros((2, 2))
    hasher.encode_image(image_array=ret_val)
    mocker_preprocess_image.assert_called_with(
        image=ret_val, target_size=(8, 8), grayscale=True
    )
    np.testing.assert_array_equal(ret_val, mocker_hash_func.call_args[0][0])


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


def test_encode_image_accepts_non_posixpath(
    hasher, mocker_load_image, mocker_hash_func
):
    ret_val = np.zeros((2, 2))
    hasher.encode_image(image_file=PATH_SINGLE_IMAGE_STRING)
    mocker_load_image.assert_called_with(
        image_file=PATH_SINGLE_IMAGE, grayscale=True, target_size=(8, 8)
    )
    np.testing.assert_array_equal(ret_val, mocker_hash_func.call_args[0][0])


# _encoder


@pytest.fixture
def mocker_encode_image(mocker):
    mocker.patch(
        'imagededup.methods.hashing.parallelise', return_value='123456789ABCDEFA'
    )


# encode_images


def test_encode_images_accepts_valid_posixpath(hasher, mocker_encode_image):
    assert len(hasher.encode_images(PATH_IMAGE_DIR)) == 6  # 6 files in the directory


def test_encode_images_accepts_non_posixpath(hasher, mocker_encode_image):
    assert len(hasher.encode_images(PATH_IMAGE_DIR_STRING)) == 6


def test_encode_images_rejects_non_directory_paths(hasher):
    with pytest.raises(ValueError):
        hasher.encode_images(PATH_SINGLE_IMAGE)


def test_encode_images_return_vals(hasher, mocker_encode_image):
    encoded_val = '123456789ABCDEFA'
    hashes = hasher.encode_images(PATH_IMAGE_DIR)
    assert isinstance(hashes, dict)
    assert list(hashes.values())[0] == encoded_val[0]
    assert PATH_SINGLE_IMAGE.name in hashes.keys()


def test_hash_func(hasher, mocker):
    inp_array = np.array((3, 3))
    ret_arr = np.array((2, 2))
    hash_algo_mocker = mocker.patch(
        'imagededup.methods.hashing.Hashing._hash_algo', return_value=ret_arr
    )
    array_mocker = mocker.patch('imagededup.methods.hashing.Hashing._array_to_hash')
    hasher._hash_func(inp_array)
    np.testing.assert_array_equal(inp_array, hash_algo_mocker.call_args[0][0])
    array_mocker.assert_called_with(ret_arr)


# _find_duplicates_dict


@pytest.mark.skipif(sys.platform == 'win32', reason='Does not run on Windows.')
def test__find_duplicates_dict_outfile_none(mocker):
    encoding_map = {'1.jpg': '123456'}
    threshold = 10
    scores = True
    outfile = None
    verbose = False
    myhasher = PHash(verbose=verbose)
    hasheval_mocker = mocker.patch('imagededup.methods.hashing.HashEval')
    save_json_mocker = mocker.patch('imagededup.methods.hashing.save_json')
    myhasher._find_duplicates_dict(
        encoding_map=encoding_map,
        max_distance_threshold=threshold,
        scores=scores,
        outfile=outfile,
    )
    hasheval_mocker.assert_called_with(
        test=encoding_map,
        queries=encoding_map,
        distance_function=Hashing.hamming_distance,
        verbose=verbose,
        threshold=threshold,
        search_method='brute_force_cython',
    )
    hasheval_mocker.return_value.retrieve_results.assert_called_once_with(scores=scores)
    save_json_mocker.assert_not_called()


@pytest.mark.skipif(sys.platform == 'win32', reason='Does not run on Windows.')
def test__find_duplicates_dict_outfile_none_verbose(hasher, mocker):
    encoding_map = {'1.jpg': '123456'}
    threshold = 10
    scores = True
    outfile = None
    hasheval_mocker = mocker.patch('imagededup.methods.hashing.HashEval')
    save_json_mocker = mocker.patch('imagededup.methods.hashing.save_json')
    hasher._find_duplicates_dict(
        encoding_map=encoding_map,
        max_distance_threshold=threshold,
        scores=scores,
        outfile=outfile,
    )
    hasheval_mocker.assert_called_with(
        test=encoding_map,
        queries=encoding_map,
        distance_function=Hashing.hamming_distance,
        verbose=True,
        threshold=threshold,
        search_method='brute_force_cython',
    )
    hasheval_mocker.return_value.retrieve_results.assert_called_once_with(scores=scores)
    save_json_mocker.assert_not_called()


@pytest.mark.skipif(sys.platform == 'win32', reason='Does not run on Windows.')
def test__find_duplicates_dict_outfile_true(hasher, mocker):
    encoding_map = {'1.jpg': '123456'}
    threshold = 10
    scores = True
    outfile = True
    verbose = True
    hasheval_mocker = mocker.patch('imagededup.methods.hashing.HashEval')
    hasheval_mocker.return_value.retrieve_results.return_value = {
        'filename.jpg': [('dup1.jpg', 3)],
        'filename2.jpg': [('dup2.jpg', 10)],
    }
    save_json_mocker = mocker.patch('imagededup.methods.hashing.save_json')
    hasher._find_duplicates_dict(
        encoding_map=encoding_map,
        max_distance_threshold=threshold,
        scores=scores,
        outfile=outfile,
    )
    hasheval_mocker.assert_called_with(
        test=encoding_map,
        queries=encoding_map,
        distance_function=Hashing.hamming_distance,
        verbose=verbose,
        threshold=threshold,
        search_method='brute_force_cython',
    )
    hasheval_mocker.return_value.retrieve_results.assert_called_once_with(scores=scores)
    save_json_mocker.assert_called_once_with(
        hasheval_mocker.return_value.retrieve_results.return_value, outfile
    )


# _find_duplicates_dir


@pytest.mark.skipif(sys.platform == 'win32', reason='Does not run on Windows.')
def test__find_duplicates_dir(hasher, mocker):
    encoding_map = {'1.jpg': '123456'}
    threshold = 10
    scores = True
    outfile = True
    ret_val_find_dup_dict = {
        'filename.jpg': [('dup1.jpg', 3)],
        'filename2.jpg': [('dup2.jpg', 10)],
    }
    encode_images_mocker = mocker.patch(
        'imagededup.methods.hashing.Hashing.encode_images', return_value=encoding_map
    )
    find_dup_dict_mocker = mocker.patch(
        'imagededup.methods.hashing.Hashing._find_duplicates_dict',
        return_value=ret_val_find_dup_dict,
    )
    hasher._find_duplicates_dir(
        image_dir=PATH_IMAGE_DIR,
        max_distance_threshold=threshold,
        scores=scores,
        outfile=outfile,
        search_method='brute_force_cython',
    )
    encode_images_mocker.assert_called_once_with(PATH_IMAGE_DIR)
    find_dup_dict_mocker.assert_called_once_with(
        encoding_map=encoding_map,
        max_distance_threshold=threshold,
        scores=scores,
        outfile=outfile,
        search_method='brute_force_cython',
    )


# find_duplicates


@pytest.fixture
def mocker_hamming_distance(mocker):
    return mocker.patch(
        'imagededup.methods.hashing.Hashing._check_hamming_distance_bounds'
    )


@pytest.mark.skipif(sys.platform == 'win32', reason='Does not run on Windows.')
def test_find_duplicates_dir(hasher, mocker, mocker_hamming_distance):
    threshold = 10
    scores = True
    outfile = True
    find_dup_dir_mocker = mocker.patch(
        'imagededup.methods.hashing.Hashing._find_duplicates_dir'
    )
    hasher.find_duplicates(
        image_dir=PATH_IMAGE_DIR,
        max_distance_threshold=threshold,
        outfile=outfile,
        scores=scores,
        search_method='brute_force_cython',
    )
    mocker_hamming_distance.assert_called_once_with(thresh=threshold)
    find_dup_dir_mocker.assert_called_once_with(
        image_dir=PATH_IMAGE_DIR,
        max_distance_threshold=threshold,
        scores=scores,
        outfile=outfile,
        search_method='brute_force_cython',
    )


@pytest.mark.skipif(sys.platform == 'win32', reason='Does not run on Windows.')
def test_find_duplicates_dict(hasher, mocker, mocker_hamming_distance):
    encoding_map = {'1.jpg': '123456'}
    threshold = 10
    scores = True
    outfile = True
    find_dup_dict_mocker = mocker.patch(
        'imagededup.methods.hashing.Hashing._find_duplicates_dict'
    )
    hasher.find_duplicates(
        encoding_map=encoding_map,
        max_distance_threshold=threshold,
        outfile=outfile,
        scores=scores,
        search_method='brute_force_cython',
    )
    mocker_hamming_distance.assert_called_once_with(thresh=threshold)
    find_dup_dict_mocker.assert_called_once_with(
        encoding_map=encoding_map,
        max_distance_threshold=threshold,
        scores=scores,
        outfile=outfile,
        search_method='brute_force_cython',
    )


def test_find_duplicates_wrong_input(hasher):
    with pytest.raises(ValueError):
        hasher.find_duplicates(max_distance_threshold=10)


# find_duplicates_to_remove


def test_find_duplicates_to_remove_outfile_false(hasher, mocker):
    threshold = 10
    outfile = False
    ret_val_find_dup_dict = {
        'filename.jpg': [('dup1.jpg', 3)],
        'filename2.jpg': [('dup2.jpg', 10)],
    }
    find_duplicates_mocker = mocker.patch(
        'imagededup.methods.hashing.Hashing.find_duplicates',
        return_value=ret_val_find_dup_dict,
    )
    get_files_to_remove_mocker = mocker.patch(
        'imagededup.methods.hashing.get_files_to_remove'
    )
    save_json_mocker = mocker.patch('imagededup.methods.hashing.save_json')
    hasher.find_duplicates_to_remove(
        image_dir=PATH_IMAGE_DIR, max_distance_threshold=threshold, outfile=outfile
    )
    find_duplicates_mocker.assert_called_once_with(
        image_dir=PATH_IMAGE_DIR,
        encoding_map=None,
        max_distance_threshold=threshold,
        scores=False,
    )
    get_files_to_remove_mocker.assert_called_once_with(ret_val_find_dup_dict)
    save_json_mocker.assert_not_called()


def test_find_duplicates_to_remove_outfile_true(hasher, mocker):
    threshold = 10
    outfile = True
    ret_val_find_dup_dict = {
        'filename.jpg': [('dup1.jpg', 3)],
        'filename2.jpg': [('dup2.jpg', 10)],
    }
    ret_val_get_files_to_remove = ['1.jpg', '2.jpg']
    find_duplicates_mocker = mocker.patch(
        'imagededup.methods.hashing.Hashing.find_duplicates',
        return_value=ret_val_find_dup_dict,
    )
    get_files_to_remove_mocker = mocker.patch(
        'imagededup.methods.hashing.get_files_to_remove',
        return_value=ret_val_get_files_to_remove,
    )
    save_json_mocker = mocker.patch('imagededup.methods.hashing.save_json')
    hasher.find_duplicates_to_remove(
        image_dir=PATH_IMAGE_DIR, max_distance_threshold=threshold, outfile=outfile
    )
    find_duplicates_mocker.assert_called_once_with(
        image_dir=PATH_IMAGE_DIR,
        encoding_map=None,
        max_distance_threshold=threshold,
        scores=False,
    )
    get_files_to_remove_mocker.assert_called_once_with(ret_val_find_dup_dict)
    save_json_mocker.assert_called_once_with(ret_val_get_files_to_remove, outfile)


def test_find_duplicates_to_remove_encoding_map(hasher, mocker):
    encoding_map = {'1.jpg': '123456'}
    threshold = 10
    outfile = False
    ret_val_find_dup_dict = {
        'filename.jpg': [('dup1.jpg', 3)],
        'filename2.jpg': [('dup2.jpg', 10)],
    }
    find_duplicates_mocker = mocker.patch(
        'imagededup.methods.hashing.Hashing.find_duplicates',
        return_value=ret_val_find_dup_dict,
    )
    get_files_to_remove_mocker = mocker.patch(
        'imagededup.methods.hashing.get_files_to_remove'
    )
    save_json_mocker = mocker.patch('imagededup.methods.hashing.save_json')
    hasher.find_duplicates_to_remove(
        encoding_map=encoding_map, max_distance_threshold=threshold, outfile=outfile
    )
    find_duplicates_mocker.assert_called_once_with(
        encoding_map=encoding_map,
        image_dir=None,
        max_distance_threshold=threshold,
        scores=False,
    )
    get_files_to_remove_mocker.assert_called_once_with(ret_val_find_dup_dict)
    save_json_mocker.assert_not_called()


# Integration tests

phasher = PHash()
dhasher = DHash()
ahasher = AHash()
whasher = WHash()

common_test_parameters = [
    phasher.encode_image,
    dhasher.encode_image,
    ahasher.encode_image,
    whasher.encode_image,
]


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
        hash_im_2 = hash_function(p.parent / 'data/mixed_images/ukbench09268.jpg')
        hamdist = Hashing.hamming_distance(hash_im_1, hash_im_2)
        assert hamdist > 20

    def test_same_hashes_with_different_inputs(self, hash_function):
        arr_inp = np.array(Image.open(PATH_SINGLE_IMAGE))
        assert hash_function(image_array=arr_inp) == hash_function(PATH_SINGLE_IMAGE)


def test_encode_images_returns_dict():
    hash_dict = phasher.encode_images(PATH_IMAGE_DIR)
    assert isinstance(hash_dict, dict)


def test_encode_images_return_non_none_hashes():
    hash_dict = dhasher.encode_images(PATH_IMAGE_DIR)
    for v in hash_dict.values():
        assert v is not None


# For each of the hash types, check correctness of hashes for known images
# Check encode_image(s)


@pytest.mark.parametrize(
    'hash_object, expected_hash',
    [
        (phasher, '9fee256239984d71'),
        (dhasher, '2b69707551f1b87a'),
        (ahasher, '81b8bc3c3c3c1e0a'),
        (whasher, '89b8bc3c3c3c5e0e'),
    ],
)
def test_encode_image_hash(hash_object, expected_hash):
    assert hash_object.encode_image(PATH_SINGLE_IMAGE) == expected_hash


def test_encode_image_corrupt_file():
    whasher = WHash()
    assert whasher.encode_image(PATH_SINGLE_IMAGE_CORRUPT) is None


def test_encode_images_corrupt_and_good_images():
    ahasher = AHash()
    hashes = ahasher.encode_images(PATH_IMAGE_DIR)
    assert len(hashes) == 5  # 5 non-corrupt files in the directory, 1 corrupt
    assert isinstance(hashes, dict)


def test_find_duplicates_correctness():
    phasher = PHash()
    duplicate_dict = phasher.find_duplicates(
        image_dir=PATH_IMAGE_DIR, max_distance_threshold=10
    )
    assert isinstance(duplicate_dict, dict)
    assert isinstance(list(duplicate_dict.values())[0], list)
    assert len(duplicate_dict['ukbench09268.jpg']) == 0
    assert duplicate_dict['ukbench00120.jpg'] == ['ukbench00120_resize.jpg']


def test_find_duplicates_correctness_score():
    phasher = PHash()
    duplicate_dict = phasher.find_duplicates(
        image_dir=PATH_IMAGE_DIR, max_distance_threshold=10, scores=True
    )
    assert isinstance(duplicate_dict, dict)
    duplicates = list(duplicate_dict.values())
    assert isinstance(duplicates[0], list)
    assert isinstance(duplicates[0][0], tuple)
    assert duplicate_dict['ukbench09268.jpg'] == []
    assert duplicate_dict['ukbench00120.jpg'] == [('ukbench00120_resize.jpg', 0)]


@pytest.mark.skipif(sys.platform == 'win32', reason='Does not run on Windows.')
def test_find_duplicates_clearing():
    phasher = PHash()
    duplicate_dict = phasher.find_duplicates(
        image_dir=PATH_IMAGE_DIR,
        max_distance_threshold=10,
        scores=True,
        search_method='brute_force_cython',
    )

    duplicate_dict = phasher.find_duplicates(
        image_dir=PATH_IMAGE_DIR,
        max_distance_threshold=10,
        scores=True,
        search_method='brute_force_cython',
    )

    assert isinstance(duplicate_dict, dict)
    duplicates = list(duplicate_dict.values())
    assert isinstance(duplicates[0], list)
    assert isinstance(duplicates[0][0], tuple)
    assert duplicate_dict['ukbench09268.jpg'] == []
    assert duplicate_dict['ukbench00120.jpg'] == [('ukbench00120_resize.jpg', 0)]


def test_find_duplicates_outfile():
    dhasher = DHash()
    outfile_name = 'score_output.json'
    if os.path.exists(outfile_name):
        os.remove(outfile_name)
    _ = dhasher.find_duplicates(
        image_dir=PATH_IMAGE_DIR,
        max_distance_threshold=10,
        scores=True,
        outfile=outfile_name,
    )
    assert os.path.exists(outfile_name)
    # clean up
    if os.path.exists(outfile_name):
        os.remove(outfile_name)


def test_find_duplicates_encoding_map_input():
    encoding = {
        'ukbench00120_resize.jpg': '9fee256239984d71',
        'ukbench00120_rotation.jpg': '850d513c4fdcbb72',
        'ukbench00120.jpg': '9fee256239984d71',
        'ukbench00120_hflip.jpg': 'cabb7237e8cd3824',
        'ukbench09268.jpg': 'c73c36c2da2f29c9',
    }
    phasher = PHash()
    duplicate_dict = phasher.find_duplicates(
        encoding_map=encoding, max_distance_threshold=10
    )
    assert isinstance(duplicate_dict, dict)
    assert isinstance(list(duplicate_dict.values())[0], list)
    assert len(duplicate_dict['ukbench09268.jpg']) == 0
    assert duplicate_dict['ukbench00120.jpg'] == ['ukbench00120_resize.jpg']


def test_find_duplicates_to_remove_dir():
    phasher = PHash()
    removal_list = phasher.find_duplicates_to_remove(
        image_dir=PATH_IMAGE_DIR, max_distance_threshold=10
    )
    assert isinstance(removal_list, list)
    assert removal_list == ['ukbench00120.jpg'] or removal_list == [
        'ukbench00120_resize.jpg'
    ]


def test_find_duplicates_to_remove_encoding():
    encoding = {
        'ukbench00120_resize.jpg': '9fee256239984d71',
        'ukbench00120_rotation.jpg': '850d513c4fdcbb72',
        'ukbench00120.jpg': '9fee256239984d71',
        'ukbench00120_hflip.jpg': 'cabb7237e8cd3824',
        'ukbench09268.jpg': 'c73c36c2da2f29c9',
    }
    phasher = PHash()
    removal_list = phasher.find_duplicates_to_remove(
        encoding_map=encoding, max_distance_threshold=10
    )
    assert isinstance(removal_list, list)
    assert removal_list == ['ukbench00120.jpg'] or removal_list == [
        'ukbench00120_resize.jpg'
    ]


def test_find_duplicates_to_remove_outfile():
    dhasher = DHash()
    outfile_name = 'removal_list.json'
    if os.path.exists(outfile_name):
        os.remove(outfile_name)
    _ = dhasher.find_duplicates(
        image_dir=PATH_IMAGE_DIR, max_distance_threshold=10, outfile=outfile_name
    )
    assert os.path.exists(outfile_name)
    # clean up
    if os.path.exists(outfile_name):
        os.remove(outfile_name)


# test verbose
def test_encode_images_verbose_true(capsys):
    phasher = PHash(verbose=True)
    phasher.encode_images(image_dir=PATH_IMAGE_DIR)
    out, err = capsys.readouterr()

    assert '%' in err
    assert '' == out


def test_encode_images_verbose_false(capsys):
    phasher = PHash(verbose=False)
    phasher.encode_images(image_dir=PATH_IMAGE_DIR)
    out, err = capsys.readouterr()

    assert '' == err
    assert '' == out


def test_find_duplicates_verbose_true(capsys):
    phasher = PHash(verbose=True)
    phasher.find_duplicates(
        image_dir=PATH_IMAGE_DIR, max_distance_threshold=10, scores=False, outfile=False
    )
    out, err = capsys.readouterr()

    assert '%' in err
    assert '' == out


def test_find_duplicates_verbose_false(capsys):
    phasher = PHash(verbose=False)
    phasher.find_duplicates(
        image_dir=PATH_IMAGE_DIR, max_distance_threshold=10, scores=False, outfile=False
    )
    out, err = capsys.readouterr()

    assert '' == out
    assert '' == err
