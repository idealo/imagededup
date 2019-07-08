from imagededup.retrieve.brute_force import BruteForce
from imagededup.hashing import Hashing
from collections import OrderedDict


def initialize():
    hash_dict = OrderedDict({'a': '9', 'b': 'D', 'c': 'A', 'd': 'F', 'e': '2', 'f': '6', 'g': '7', 'h': 'E'})
    dist_func = Hashing.hamming_distance
    return hash_dict, dist_func


def test_correctness():
    hash_dict, dist_func = initialize()
    bf = BruteForce(hash_dict, dist_func)
    query = '5'
    valid_retrievals = bf.search(query, tol=2)
    assert len(valid_retrievals) == 5


def test_search_correctness():
    # Input a tree and send a search query, check whether correct retrievals are returned
    hash_dict, dist_func = initialize()
    bk = BruteForce(hash_dict, dist_func)
    query = '5'
    valid_retrievals = bk.search(query, tol=2)
    assert set(valid_retrievals.keys()) == set(['a', 'f', 'g', 'd', 'b'])