from imagededup.retrieval import ResultSet
from imagededup.hashing import Hashing
from imagededup.retrieval import CosEval
from mock import patch
import os, pdb
import numpy as np

"""Run from project root with: python -m pytest -vs tests/test_retrieval.py --cov=imagededup.retrieval"""


def test_resultset_initialization(dummy_image={'ukbench09060.jpg': 'e064ece078d7c96a'}):
    dummy_hasher = Hashing()
    dummy_result = ResultSet(dummy_image, dummy_image, dummy_hasher.hamming_distance)
    assert dummy_result.queries and dummy_result.candidates


def test_invoker_initialization(dummy_image={'ukbench09060.jpg': 'e064ece078d7c96a'}):
    dummy_hasher = Hashing()
    dummy_result = ResultSet(dummy_image, dummy_image, dummy_hasher.hamming_distance)
    assert dummy_result.hamming_distance_invoker('e064ece078d7c96a', 'a064ece078d7c96e') == 2


def test_resultset_completeness(
        dummy_query={'ukbench00120.jpg': '2b69707551f1b87a', 'ukbench09268.jpg': 'ac9c72f8e1c2c448'}):
    dummy_db = {
        'ukbench00120_hflip.jpg': '2b69f1517570e2a1',
        'ukbench00120_resize.jpg': '2b69707551f1b87a',
        'ukbench09268.jpg': 'ac9c72f8e1c2c448'
    }
    dummy_hasher = Hashing()
    dummy_result = ResultSet(dummy_db, dummy_query, dummy_hasher.hamming_distance)
    assert len(dummy_result.retrieve_results()) == len(dummy_query)


def test_resultset_correctness(
        dummy_query={'ukbench00120.jpg': '2b69707551f1b87a', 'ukbench09268.jpg': 'ac9c72f8e1c2c448'}):
    dummy_db = {
        'ukbench00120_fake.jpg': '2b69707551f1b87d',
        'ukbench00120_resize.jpg': '2b69707551f1b87a',
        'ukbench09268_2.jpg': 'ac9c72f8e1c2c448'
    }
    dummy_hasher = Hashing()
    dummy_result = ResultSet(dummy_db, dummy_query, dummy_hasher.hamming_distance, cutoff=3)
    # pdb.set_trace()
    res = dummy_result.retrieve_results()
    print(res)
    dummy_distances = [max(res[dist].values()) for dist in res]
    print(dummy_distances)
    assert max(dummy_distances) == 3


def test_result_consistency_across_search_methods(
        dummy_query={'ukbench00120.jpg': '2b69707551f1b87a', 'ukbench09268.jpg': 'ac9c72f8e1c2c448'}):
    dummy_db = {
        'ukbench00120_hflip.jpg': '2b69f1517570e2a1',
        'ukbench00120_resize.jpg': '2b69707551f1b87a',
        'ukbench09268.jpg': 'ac9c72f8e1c2c448'
    }
    dummy_hasher = Hashing()
    left_result = ResultSet(dummy_db, dummy_query, dummy_hasher.hamming_distance, search_method='brute_force')\
        .retrieve_results()
    right_result = ResultSet(dummy_db, dummy_query, dummy_hasher.hamming_distance).retrieve_results()
    assert left_result == right_result


def test_no_self_retrieval():
    dummy_query = {'ukbench00120.jpg': '2b69707551f1b87a', 'ukbench09268.jpg': 'ac9c72f8e1c2c448'}
    dummy_db = {
    'ukbench00120_hflip.jpg': '2b69f1517570e2a1',
    'ukbench00120_resize.jpg': '2b69707551f1b87a',
    'ukbench09268.jpg': 'ac9c72f8e1c2c448'

    }
    dummy_hasher = Hashing()
    brute_res = ResultSet(dummy_db, dummy_query, dummy_hasher.hamming_distance, search_method='brute_force') \
        .retrieve_results()
    bktree_res = ResultSet(dummy_db, dummy_query, dummy_hasher.hamming_distance).retrieve_results()
    assert len(brute_res['ukbench09268.jpg']) == 0
    assert len(bktree_res['ukbench09268.jpg']) == 0


def test_max_hamming_threshold_not_violated(
        dummy_query={'ukbench00120.jpg': '2b69707551f1b87a', 'ukbench09268.jpg': 'ac9c72f8e1c2c448'}):
    dummy_db = {
        'ukbench00120_hflip.jpg': '2b69f1517570e2a1',
        'ukbench00120_resize.jpg': '2b69707551f1b87a',
        'ukbench09268_2.jpg': 'ac9c72f8e1c2c448'
    }
    dummy_hasher = Hashing()
    dummy_result = ResultSet(dummy_db, dummy_query, dummy_hasher.hamming_distance, search_method='brute_force')
    res = dummy_result.retrieve_results()
    dummy_distances = [max(res[dist].values()) for dist in res]
    assert max(dummy_distances) < 5


def test_save_true(dummy_image={'ukbench09060.jpg': 'e064ece078d7c96a'}, dummy_file='retrieved_results_map.pkl'):
    if os.path.exists(dummy_file):
        os.remove(dummy_file)
    dummy_hasher = Hashing()
    _ = ResultSet(dummy_image, dummy_image, dummy_hasher.hamming_distance, save=True)
    assert os.path.exists(dummy_file)


def test_save_false(dummy_image={'ukbench09060.jpg': 'e064ece078d7c96a'}, dummy_file='retrieved_results_map.pkl'):
    if os.path.exists(dummy_file):
        os.remove(dummy_file)
    dummy_hasher = Hashing()
    _ = ResultSet(dummy_image, dummy_image, dummy_hasher.hamming_distance)
    assert not os.path.exists(dummy_file)


def test_coseval_normalization():
    inp_arr = np.array([[1, 0], [1, 1]])
    normed_mat = CosEval.get_normalized_matrix(inp_arr)
    np.testing.assert_array_equal(normed_mat[0], inp_arr[0])
    np.testing.assert_array_equal(normed_mat[1], inp_arr[1] / np.sqrt(2))
    assert normed_mat.shape == (2, 2)


@patch('imagededup.retrieval.CosEval._normalize_vector_matrices')
def test_normalize_vector_matrices(mocker):
    inp_arr_1 = np.array([[1, 0], [1, 1]])
    inp_arr_2 = np.array([[1, 0], [1, 1]])
    cosev = CosEval(inp_arr_1, inp_arr_2)
    cosev._normalize_vector_matrices.assert_called()


def test__get_similarity():
    inp_arr_1 = np.array([[1, 0], [1, 1]])
    inp_arr_2 = np.array([[1, 0], [1, 1], [1, 0], [1, 1]])
    cosev = CosEval(inp_arr_1, inp_arr_2)
    assert cosev.sim_mat is None
    _ = cosev._get_similarity()
    assert cosev.sim_mat.shape == (2, 4)


def test__get_matches_above_threshold():
    rets_ind, rets_val = CosEval._get_matches_above_threshold(row=np.array([0.1, -0.1, 0.2, 1.9]), thresh=0.8)
    assert rets_ind == np.array([3])
    np.testing.assert_array_equal(rets_val, np.array([1.9]))


def test_get_retrievals_at_thresh():
    inp_arr_1 = np.array([[1, 0], [1, 1]])
    inp_arr_2 = np.array([[1, 0], [1, 1], [1, 0], [1, 1]])
    cosev = CosEval(inp_arr_1, inp_arr_2)
    filemapping_query = {0: 'ukbench00002.jpg', 1: 'ukbench00003.jpg'}
    filemapping_ret = {0: 'ukbench00004.jpg', 1: 'ukbench00005.jpg', 2: 'ukbench00006.jpg', 3: 'ukbench00007.jpg'}
    dict_ret = cosev.get_retrievals_at_thresh(filemapping_query, filemapping_ret, thresh=0.99)
    assert set(dict_ret['ukbench00002.jpg'].keys()) == set(['ukbench00004.jpg', 'ukbench00006.jpg'])


def test_get_retrievals_at_thresh_query_name_in_retrieval():
    inp_arr_1 = np.array([[1, 0], [1, 1]])
    inp_arr_2 = np.array([[1, 0], [1, 1], [1, 0], [1, 1]])
    cosev = CosEval(inp_arr_1, inp_arr_2)
    filemapping_query = {0: 'ukbench00002.jpg', 1: 'ukbench00003.jpg'}
    filemapping_ret = {0: 'ukbench00002.jpg', 1: 'ukbench00005.jpg', 2: 'ukbench00006.jpg', 3: 'ukbench00007.jpg'}
    dict_ret = cosev.get_retrievals_at_thresh(filemapping_query, filemapping_ret, thresh=0.99)
    assert set(dict_ret['ukbench00002.jpg'].keys()) == set(['ukbench00006.jpg'])
