from imagededup.retrieve import BruteForce
from imagededup.hashing import Hashing
from collections import OrderedDict


def initialize():
    hash_dict = OrderedDict({'a': '9', 'b': 'D', 'c': 'A', 'd': 'F', 'e': '2', 'f': '6', 'g': '7', 'h': 'E'})
    dist_func = Hashing.hamming_distance
    return hash_dict, dist_func


def test_search_correctness():
    hash_dict, dist_func = initialize()
    bf = BruteForce(hash_dict, dist_func)
    query = '5'
    valid_retrievals = bf.search(query, tol=2)
    assert set(valid_retrievals.keys()) == set(['a', 'f', 'g', 'd', 'b'])


def test_tolerance_value_effect():
    hash_dict, dist_func = initialize()
    bf = BruteForce(hash_dict, dist_func)
    query = '5'
    valid_retrievals_2 = bf.search(query, tol=2)
    valid_retrievals_3 = bf.search(query, tol=3)
    assert set(valid_retrievals_2.keys()) != set(valid_retrievals_3.keys())