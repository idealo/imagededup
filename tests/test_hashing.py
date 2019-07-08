from imagededup.hashing import Hashing, HashedDataset, Dataset
import pytest
import numpy as np
from pathlib import Path
from PIL import Image

""" Run from project root with: python -m pytest -vs tests/test_hashing.py"""


def test_bool_to_hex():
    bool_num = '11011100'
    assert Hashing.bool_to_hex(bool_num) == 'dc'


def test_bool_to_hex_msb_0():
    bool_num = '00011100'
    assert Hashing.bool_to_hex(bool_num) == '1c'


def test_hamming_distance():
    """Put two numbers and check if hamming distance is correct"""
    number_1 = "1a"
    number_2 = "1f"
    hamdist = Hashing.hamming_distance(number_1, number_2)
    assert hamdist == 2


hash_obj = Hashing()


@pytest.mark.parametrize('hash_function', [hash_obj.phash, hash_obj.ahash, hash_obj.dhash, hash_obj.whash])
class TestCommon:
    def test_len_hash(self, hash_function):
        hash_im = hash_function(Path('tests/data/mixed_images/ukbench00120.jpg'))
        assert len(hash_im) == 16

    def test_hash_resize(self, hash_function):
        """Resize one image to (300, 300) and check that hamming distance between hashes is not too large"""
        hash_im_1 = hash_function(Path('tests/data/mixed_images/ukbench00120.jpg'))
        hash_im_2 = hash_function(Path('tests/data/mixed_images/ukbench00120_resize.jpg'))
        hamdist = Hashing.hamming_distance(hash_im_1, hash_im_2)
        assert hamdist < 3

    def test_hash_small_rotation(self, hash_function):
        """Rotate image slightly (1 degree) and check that hamming distance between hashes is not too large"""
        if hash_function == hash_obj.whash:
            pytest.skip()
        orig_image = Image.open(Path('tests/data/mixed_images/ukbench00120.jpg'))
        rotated_image = np.array(orig_image.rotate(1))
        hash_im_1 = hash_function(np.array(orig_image))
        hash_im_2 = hash_function(rotated_image)
        hamdist = Hashing.hamming_distance(hash_im_1, hash_im_2)
        assert hamdist < 3

    def test_hash_distinct_images(self, hash_function):
        """Put in distinct images and check that hamming distance between hashes is large"""
        image_1 = Image.open(Path('tests/data/mixed_images/ukbench00120.jpg'))
        image_2 = Image.open(Path('tests/data/mixed_images/ukbench09268.jpg'))
        hash_im_1 = hash_function(np.array(image_1))
        hash_im_2 = hash_function(np.array(image_2))
        hamdist = Hashing.hamming_distance(hash_im_1, hash_im_2)
        assert hamdist > 10

    def test_hash_on_dir_returns_dict(self, hash_function):
        path_dir = Path('tests/data/base_images')
        hash_obj = Hashing()
        hash_dict = hash_obj._run_hash_on_dir(path_dir, hash_function)
        assert isinstance(hash_dict, dict)

    def test_hash_on_dir_return_non_none_hashes(self, hash_function):
        path_dir = Path('tests/data/base_images')
        hash_obj = Hashing()
        hash_dict = hash_obj._run_hash_on_dir(path_dir, hash_function)
        for v in hash_dict.values():
            assert v is not None

    def test_hash_on_dir_runs_for_all_files_in_dir(self, hash_function):
        """There are 10 images in the directory below"""
        path_dir = Path('tests/data/base_images')
        hash_obj = Hashing()
        hash_dict = hash_obj._run_hash_on_dir(path_dir, hash_function)
        assert len(hash_dict.keys()) == 10


def test_phash_dir(mocker):
    path_dir = Path('tests/data/base_images')
    hash_obj = Hashing()
    hash_func = hash_obj.phash
    mocker.patch.object(hash_obj, '_run_hash_on_dir')
    hash_obj.phash_dir(path_dir)
    hash_obj._run_hash_on_dir.assert_called_with(path_dir, hash_func)


def test_ahash_dir(mocker):
    path_dir = Path('tests/data/base_images')
    hash_obj = Hashing()
    hash_func = hash_obj.ahash
    mocker.patch.object(hash_obj, '_run_hash_on_dir')
    hash_obj.ahash_dir(path_dir)
    hash_obj._run_hash_on_dir.assert_called_with(path_dir, hash_func)


def test_dhash_dir(mocker):
    path_dir = Path('tests/data/base_images')
    hash_obj = Hashing()
    hash_func = hash_obj.dhash
    mocker.patch.object(hash_obj, '_run_hash_on_dir')
    hash_obj.dhash_dir(path_dir)
    hash_obj._run_hash_on_dir.assert_called_with(path_dir, hash_func)


def test_whash_dir(mocker):
    path_dir = Path('tests/data/base_images')
    hash_obj = Hashing()
    hash_func = hash_obj.whash
    mocker.patch.object(hash_obj, '_run_hash_on_dir')
    hash_obj.whash_dir(path_dir)
    hash_obj._run_hash_on_dir.assert_called_with(path_dir, hash_func)


def test_load_image_initializes_docs(path_dir=Path('tests/data/base_images')):
    dummy = Dataset(path_dir, path_dir)
    assert dummy.query_docs and dummy.test_docs


def test_fingerprint_hashes_images_succesfully(path_dir=Path('tests/data/base_images')):
    dummy_hashes = {
        'ukbench09060.jpg': 'e064ece078d7c96a',
        'ukbench08976.jpg': 'aa49c9c3eae6e4f0',
        'ukbench00120.jpg': '2b69707551f1b87a',
        'ukbench09348.jpg': 'b269697072b2b2f0',
        'ukbench09012.jpg': 'e1a6c3a2a2b2c2c1',
        'ukbench09380.jpg': 'c888888c869292cc',
        'ukbench09040.jpg': 'ccf1b0f2f2f2e0c1',
        'ukbench08996.jpg': 'fe27656362723d7f',
        'ukbench09268.jpg': 'ac9c72f8e1c2c448',
        'ukbench01380.jpg': 'c129654d4d4d0d25'
    }
    dummy_hasher = Hashing().dhash
    dummy_set = HashedDataset(dummy_hasher, path_dir, path_dir)
    assert dummy_set.test_hashes == dummy_hashes


def test_dataset_hashing_lacks_collisions(path_dir=Path('tests/data/base_images'), test_path=Path('tests/data/transformed_images')):
    dummy_hasher = Hashing().dhash
    dummy_set = HashedDataset(dummy_hasher, path_dir, test_path)
    assert len(dummy_set.doc2hash) == len(dummy_set.test_hashes) + len(dummy_set.query_hashes)


def test_dataset_hashing_completeness(path_dir=Path('tests/data/base_images'), test_path=Path('tests/data/transformed_images')):
    dummy_hasher = Hashing().dhash
    dummy_set = HashedDataset(dummy_hasher, path_dir, test_path)
    all_dummy_hashes = {**dummy_set.test_hashes, **dummy_set.query_hashes}
    assert dummy_set.doc2hash.keys() == all_dummy_hashes.keys()


def test_dataset_hashmaps_veracity(path_dir=Path('tests/data/base_images'), test_path=Path('tests/data/transformed_images')):
    dummy_hasher = Hashing().dhash
    dummy_set = HashedDataset(dummy_hasher, path_dir, test_path)
    assert set(dummy_set.doc2hash.keys()) == set(dummy_set.hash2doc.values())


@pytest.fixture(scope='module')
def initialized_hash_obj():
    hashobj = Hashing()
    return hashobj


def test__get_hash(initialized_hash_obj):
    hash_mat = np.array([1, 0, 0, 1, 0, 0, 0, 1])
    assert initialized_hash_obj._get_hash(hash_mat, 2) == '91'

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


def test__find_duplicates_dir_wrong_hashing_method_raises_error(initialized_hash_obj):
    path_dir = Path('tests/data/mixed_images')
    with pytest.raises(Exception):
        initialized_hash_obj._find_duplicates_dir(path_dir=path_dir, method='hash')


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
    initialized_hash_obj._find_duplicates_dir.assert_called_with(path_dir=path_dir, method='phash', scores=False, threshold=threshold)


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
    initialized_hash_obj._find_duplicates_dict.assert_called_with(dict_file_feature=dummy_hashes, scores=False, threshold=threshold)


def test__check_hamming_distance_bounds_input_out_of_range(initialized_hash_obj):
    with pytest.raises(TypeError):
        initialized_hash_obj._check_hamming_distance_bounds(thresh=65)


def test_find_duplicates_unacceptable_input(initialized_hash_obj):
    with pytest.raises(TypeError):
        initialized_hash_obj.find_duplicates('tests/data/mixed_images')


def test_retrieve_dups(initialized_hash_obj, monkeypatch):
    def mock_find_duplicates(path_or_dict, method, threshold, scores):
        dict_ret = {'1': ['3', '4'], '2': ['3'], '3': ['1'], '4': ['1']}
        return dict_ret

    monkeypatch.setattr(initialized_hash_obj, 'find_duplicates', mock_find_duplicates)
    list_to_rem = initialized_hash_obj.find_duplicates_to_remove(path_or_dict=dict())
    assert set(list_to_rem) == set(['3', '4'])


def test_dict_dir_same_results(initialized_hash_obj):
    path_dir = Path('tests/data/base_images')

    dup_dir = initialized_hash_obj._find_duplicates_dir(path_dir=path_dir, method='phash')
    dict_hash = initialized_hash_obj.phash_dir(path_dir)
    dup_dict = initialized_hash_obj._find_duplicates_dict(dict_hash)
    assert dup_dir == dup_dict