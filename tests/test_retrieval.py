import sys
import pytest
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from imagededup.handlers.search.retrieval import (
    HashEval,
    cosine_similarity_chunk,
    get_cosine_similarity,
)
from imagededup.methods.hashing import Hashing

HAMMING_DISTANCE_FUNCTION = Hashing().hamming_distance


def test_cosine_similarity_chunk():
    X = np.random.rand(333, 100)
    start_idx = 10
    end_idx = 100

    input_tuple = (X, (start_idx, end_idx))

    result = cosine_similarity_chunk(input_tuple)

    np.testing.assert_array_almost_equal(
        result, cosine_similarity(X[start_idx:end_idx, :], X).astype('float16')
    )


def test_get_cosine_similarity():
    X = np.random.rand(333, 10)
    expected = cosine_similarity(X)

    # threshold not triggered
    result = get_cosine_similarity(X)

    np.testing.assert_array_almost_equal(result, expected)

    # threshold triggered
    result = get_cosine_similarity(X, threshold=20)

    np.testing.assert_array_almost_equal(result, expected.astype('float16'))

    # multiple chunks
    result = get_cosine_similarity(X, threshold=20, chunk_size=10)

    np.testing.assert_array_almost_equal(result, expected.astype('float16'))


def test_initialization():
    db = {'ukbench09060.jpg': 'e064ece078d7c96a'}
    threshold = 10
    hasheval_obj = HashEval(
        test=db,
        queries=db,
        distance_function=HAMMING_DISTANCE_FUNCTION,
        threshold=threshold,
    )
    assert hasheval_obj.queries and hasheval_obj.test
    assert hasheval_obj.threshold == threshold
    assert hasheval_obj.distance_invoker('e064ece078d7c96a', 'a064ece078d7c96e') == 2


def test_retrieve_results_dtypes():
    db = {'ukbench09060.jpg': 'e064ece078d7c96a'}
    result = HashEval(db, db, HAMMING_DISTANCE_FUNCTION).retrieve_results()
    assert isinstance(result, dict)
    assert isinstance(list(result.values())[0], list)


def test_retrieve_results_dtypes_scores():
    query = {
        'ukbench00120.jpg': '2b69707551f1b87a',
        'ukbench09268.jpg': 'ac9c72f8e1c2c448',
    }

    db = {
        'ukbench00120_hflip.jpg': '2b69f1517570e2a1',
        'ukbench00120_resize.jpg': '2b69707551f1b87a',
        'ukbench09268.jpg': 'ac9c72f8e1c2c448',
    }
    out_map = HashEval(db, query, HAMMING_DISTANCE_FUNCTION).retrieve_results(
        scores=True
    )
    assert isinstance(out_map, dict)
    assert isinstance(list(out_map.values())[0], list)
    assert isinstance(list(out_map.values())[0][0], tuple)
    assert len(out_map) == len(query)


def test_resultset_correctness():
    query = {
        'ukbench00120.jpg': '2b69707551f1b87a',
        'ukbench09268.jpg': 'ac9c72f8e1c2c448',
    }
    db = {
        'ukbench00120_fake.jpg': '2b69707551f1b87d',
        'ukbench00120_resize.jpg': '2b69707551f1b87a',
        'ukbench09268_2.jpg': 'ac9c72f8e1c2c448',
    }
    hasheval_obj = HashEval(db, query, HAMMING_DISTANCE_FUNCTION, threshold=3)
    results = hasheval_obj.retrieve_results(scores=True)
    distances = [i[1] for v in results.values() for i in v]
    assert max(distances) == 3


def test_result_consistency_across_search_methods():
    query = {
        'ukbench00120.jpg': '2b69707551f1b87a',
        'ukbench09268.jpg': 'ac9c72f8e1c2c448',
    }
    db = {
        'ukbench00120_hflip.jpg': '2b69f1517570e2a1',
        'ukbench00120_resize.jpg': '2b69707551f1b87a',
        'ukbench09268.jpg': 'ac9c72f8e1c2c448',
    }
    brute_force_result = HashEval(
        db, query, HAMMING_DISTANCE_FUNCTION, search_method='brute_force'
    ).retrieve_results()

    bktree_result = HashEval(db, query, HAMMING_DISTANCE_FUNCTION).retrieve_results()
    assert brute_force_result == bktree_result


@pytest.mark.skipif(sys.platform == 'win32', reason='Does not run on Windows.')
def test_result_consistency_across_search_methods_scores():
    query = {
        'ukbench00120.jpg': '2b69707551f1b87a',
        'ukbench09268.jpg': 'ac9c72f8e1c2c448',
    }

    db = {
        'ukbench00120_hflip.jpg': '2b69f1517570e2a1',
        'ukbench00120_resize.jpg': '2b69707551f1b87a',
        'ukbench09268.jpg': 'ac9c72f8e1c2c448',
    }
    brute_force_result = HashEval(
        db, query, HAMMING_DISTANCE_FUNCTION, search_method='brute_force'
    ).retrieve_results(scores=True)

    bktree_result = HashEval(db, query, HAMMING_DISTANCE_FUNCTION,search_method='bktree').retrieve_results(
        scores=True
    )

    brute_force_cython_result = HashEval(
        db, query, HAMMING_DISTANCE_FUNCTION, search_method='brute_force_cython'
    ).retrieve_results(scores=True)

    assert brute_force_result == bktree_result
    assert brute_force_cython_result == brute_force_result


def test_no_self_retrieval():
    query = {
        'ukbench00120.jpg': '2b69707551f1b87a',
        'ukbench09268.jpg': 'ac9c72f8e1c2c448',
    }
    db = {
        'ukbench00120_hflip.jpg': '2b69f1517570e2a1',
        'ukbench00120_resize.jpg': '2b69707551f1b87a',
        'ukbench09268.jpg': 'ac9c72f8e1c2c448',
    }
    brute_res = HashEval(
        db, query, HAMMING_DISTANCE_FUNCTION, search_method='brute_force'
    ).retrieve_results()

    bktree_res = HashEval(db, query, HAMMING_DISTANCE_FUNCTION).retrieve_results()
    assert len(brute_res['ukbench09268.jpg']) == 0
    assert len(bktree_res['ukbench09268.jpg']) == 0


def test_max_hamming_threshold_not_violated():
    query = {
        'ukbench00120.jpg': '2b69707551f1b87a',
        'ukbench09268.jpg': 'ac9c72f8e1c2c448',
    }
    db = {
        'ukbench00120_hflip.jpg': '2b69f1517570e2a1',
        'ukbench00120_resize.jpg': '2b69707551f1b870',
        'ukbench09268_2.jpg': 'ac9c72f8e1c2c448',
    }
    result = HashEval(
        db, query, HAMMING_DISTANCE_FUNCTION, search_method='bktree'
    ).retrieve_results(scores=True)
    distances = [i[1] for v in result.values() for i in v]
    assert max(distances) < 5  # 5 is the default threshold value


def test_results_sorted_in_ascending_distance_order():
    query = {'ukbench00120.jpg': '2b69707551f1b87a'}
    db = {
        'ukbench00120_hflip.jpg': '2b69707551f1b87f',
        'ukbench00120_resize.jpg': '2b69707551f1b87b',
        'ukbench09268_2.jpg': '2b69707551f1b870',
        'ukbench09268_3.jpg': '2c89709251f1b870',
    }
    result = HashEval(
        db, query, HAMMING_DISTANCE_FUNCTION, threshold=30, search_method='bktree'
    ).retrieve_results(scores=True)
    distances = [i[1] for v in result.values() for i in v]

    assert sorted(distances, reverse=False) == distances
