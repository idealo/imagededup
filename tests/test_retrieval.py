from imagededup.handlers.search.retrieval import HashEval
from imagededup.methods.hashing import Hashing


"""Run from project root with: python -m pytest -vs tests/test_retrieval.py --cov=imagededup.retrieval"""

HAMMING_DISTANCE_FUNCTION = Hashing().hamming_distance


def test_initialization():
    db = {"ukbench09060.jpg": "e064ece078d7c96a"}
    threshold = 10
    hasheval_obj = HashEval(
        test=db,
        queries=db,
        distance_function=HAMMING_DISTANCE_FUNCTION,
        threshold=threshold,
    )
    assert hasheval_obj.queries and hasheval_obj.test
    assert hasheval_obj.threshold == threshold
    assert hasheval_obj.distance_invoker("e064ece078d7c96a", "a064ece078d7c96e") == 2


def test_retrieve_results_dtypes():
    db = {"ukbench09060.jpg": "e064ece078d7c96a"}
    result = HashEval(db, db, HAMMING_DISTANCE_FUNCTION).retrieve_results()
    assert isinstance(result, dict)
    assert isinstance(list(result.values())[0], list)


def test_retrieve_results_dtypes_scores():
    query = {
        "ukbench00120.jpg": "2b69707551f1b87a",
        "ukbench09268.jpg": "ac9c72f8e1c2c448",
    }

    db = {
        "ukbench00120_hflip.jpg": "2b69f1517570e2a1",
        "ukbench00120_resize.jpg": "2b69707551f1b87a",
        "ukbench09268.jpg": "ac9c72f8e1c2c448",
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
        "ukbench00120.jpg": "2b69707551f1b87a",
        "ukbench09268.jpg": "ac9c72f8e1c2c448",
    }
    db = {
        "ukbench00120_fake.jpg": "2b69707551f1b87d",
        "ukbench00120_resize.jpg": "2b69707551f1b87a",
        "ukbench09268_2.jpg": "ac9c72f8e1c2c448",
    }
    hasheval_obj = HashEval(db, query, HAMMING_DISTANCE_FUNCTION, threshold=3)
    results = hasheval_obj.retrieve_results(scores=True)
    distances = [i[1] for v in results.values() for i in v]
    assert max(distances) == 3


def test_result_consistency_across_search_methods():
    query = {
        "ukbench00120.jpg": "2b69707551f1b87a",
        "ukbench09268.jpg": "ac9c72f8e1c2c448",
    }
    db = {
        "ukbench00120_hflip.jpg": "2b69f1517570e2a1",
        "ukbench00120_resize.jpg": "2b69707551f1b87a",
        "ukbench09268.jpg": "ac9c72f8e1c2c448",
    }
    brute_force_result = HashEval(
        db, query, HAMMING_DISTANCE_FUNCTION, search_method="brute_force"
    ).retrieve_results()
    bktree_result = HashEval(db, query, HAMMING_DISTANCE_FUNCTION).retrieve_results()
    assert brute_force_result == bktree_result


def test_result_consistency_across_search_methods_scores():
    query = {
        "ukbench00120.jpg": "2b69707551f1b87a",
        "ukbench09268.jpg": "ac9c72f8e1c2c448",
    }

    db = {
        "ukbench00120_hflip.jpg": "2b69f1517570e2a1",
        "ukbench00120_resize.jpg": "2b69707551f1b87a",
        "ukbench09268.jpg": "ac9c72f8e1c2c448",
    }
    brute_force_result = HashEval(
        db, query, HAMMING_DISTANCE_FUNCTION, search_method="brute_force"
    ).retrieve_results(scores=True)

    bktree_result = HashEval(db, query, HAMMING_DISTANCE_FUNCTION).retrieve_results(
        scores=True
    )
    assert brute_force_result == bktree_result


def test_no_self_retrieval():
    query = {
        "ukbench00120.jpg": "2b69707551f1b87a",
        "ukbench09268.jpg": "ac9c72f8e1c2c448",
    }
    db = {
        "ukbench00120_hflip.jpg": "2b69f1517570e2a1",
        "ukbench00120_resize.jpg": "2b69707551f1b87a",
        "ukbench09268.jpg": "ac9c72f8e1c2c448",
    }
    brute_res = HashEval(
        db, query, HAMMING_DISTANCE_FUNCTION, search_method="brute_force"
    ).retrieve_results()
    bktree_res = HashEval(db, query, HAMMING_DISTANCE_FUNCTION).retrieve_results()
    assert len(brute_res["ukbench09268.jpg"]) == 0
    assert len(bktree_res["ukbench09268.jpg"]) == 0


def test_max_hamming_threshold_not_violated():
    query = {
        "ukbench00120.jpg": "2b69707551f1b87a",
        "ukbench09268.jpg": "ac9c72f8e1c2c448",
    }
    db = {
        "ukbench00120_hflip.jpg": "2b69f1517570e2a1",
        "ukbench00120_resize.jpg": "2b69707551f1b870",
        "ukbench09268_2.jpg": "ac9c72f8e1c2c448",
    }
    result = HashEval(
        db, query, HAMMING_DISTANCE_FUNCTION, search_method="brute_force"
    ).retrieve_results(scores=True)
    distances = [i[1] for v in result.values() for i in v]
    assert max(distances) < 5  # 5 is the default threshold value


def test_results_sorted_in_descending_distance_order():
    query = {"ukbench00120.jpg": "2b69707551f1b87a"}
    db = {
        "ukbench00120_hflip.jpg": "2b69707551f1b87f",
        "ukbench00120_resize.jpg": "2b69707551f1b87b",
        "ukbench09268_2.jpg": "2b69707551f1b870",
        "ukbench09268_3.jpg": "2c89709251f1b870",
    }
    result = HashEval(
        db, query, HAMMING_DISTANCE_FUNCTION, threshold=30, search_method="brute_force"
    ).retrieve_results(scores=True)
    distances = [i[1] for v in result.values() for i in v]

    assert sorted(distances, reverse=False) == distances


"""def test_coseval_normalization():
    inp_arr = np.array([[1, 0], [1, 1]])
    normed_mat = CosEval.get_normalized_matrix(inp_arr)
    np.testing.assert_array_equal(normed_mat[0], inp_arr[0])
    np.testing.assert_array_equal(normed_mat[1], inp_arr[1] / np.sqrt(2))
    assert normed_mat.shape == (2, 2)


@patch("imagededup.retrieve.retrieval.CosEval._normalize_vector_matrices")
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
    rets_ind, rets_val = CosEval._get_matches_above_threshold(
        row=np.array([0.1, -0.1, 0.2, 1.9]), thresh=0.8
    )
    assert rets_ind == np.array([3])
    np.testing.assert_array_equal(rets_val, np.array([1.9]))


def test_get_retrievals_at_thresh():
    inp_arr_1 = np.array([[1, 0], [1, 1]])
    inp_arr_2 = np.array([[1, 0], [1, 1], [1, 0], [1, 1]])
    cosev = CosEval(inp_arr_1, inp_arr_2)
    filemapping_query = {0: "ukbench00002.jpg", 1: "ukbench00003.jpg"}
    filemapping_ret = {
        0: "ukbench00004.jpg",
        1: "ukbench00005.jpg",
        2: "ukbench00006.jpg",
        3: "ukbench00007.jpg",
    }
    dict_ret = cosev.get_retrievals_at_thresh(
        filemapping_query, filemapping_ret, thresh=0.99
    )
    assert set(dict_ret["ukbench00002.jpg"].keys()) == set(
        ["ukbench00004.jpg", "ukbench00006.jpg"]
    )


def test_get_retrievals_at_thresh_query_name_in_retrieval():
    inp_arr_1 = np.array([[1, 0], [1, 1]])
    inp_arr_2 = np.array([[1, 0], [1, 1], [1, 0], [1, 1]])
    cosev = CosEval(inp_arr_1, inp_arr_2)
    filemapping_query = {0: "ukbench00002.jpg", 1: "ukbench00003.jpg"}
    filemapping_ret = {
        0: "ukbench00002.jpg",
        1: "ukbench00005.jpg",
        2: "ukbench00006.jpg",
        3: "ukbench00007.jpg",
    }
    dict_ret = cosev.get_retrievals_at_thresh(
        filemapping_query, filemapping_ret, thresh=0.99
    )
    assert set(dict_ret["ukbench00002.jpg"].keys()) == set(["ukbench00006.jpg"])"""
