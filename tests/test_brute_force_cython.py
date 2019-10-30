from collections import OrderedDict

from imagededup.handlers.search.brute_force_cython import BruteForceCython
from imagededup.methods.hashing import Hashing


def initialize():
    hash_dict = OrderedDict(
        {'a': '9', 'b': 'D', 'c': 'A', 'd': 'F', 'e': '2', 'f': '6', 'g': '7', 'h': 'E'}
    )
    dist_func = Hashing.hamming_distance
    return hash_dict, dist_func


def test_search_correctness():
    hash_dict, dist_func = initialize()
    bf = BruteForceCython(hash_dict, dist_func)
    query = '5'
    valid_retrievals = bf.search(query, tol=2)
    assert set([i[0] for i in valid_retrievals]) == set(['a', 'f', 'g', 'd', 'b'])


def test_tolerance_value_effect():
    hash_dict, dist_func = initialize()
    bf = BruteForceCython(hash_dict, dist_func)
    query = '5'
    valid_retrievals_2 = bf.search(query, tol=2)
    valid_retrievals_3 = bf.search(query, tol=3)
    assert set([i[0] for i in valid_retrievals_2]) != set(
        [i[0] for i in valid_retrievals_3]
    )
