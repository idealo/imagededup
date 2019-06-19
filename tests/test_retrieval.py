from imagededup.retrieval import ResultSet
from imagededup.hashing import Hashing
import os

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
    assert len(dummy_result.query_results) == len(dummy_query)


def test_resultset_correctness(
        dummy_query={'ukbench00120.jpg': '2b69707551f1b87a', 'ukbench09268.jpg': 'ac9c72f8e1c2c448'}):
    dummy_db = {
        'ukbench00120_fake.jpg': '2b69707551f1b87d',
        'ukbench00120_resize.jpg': '2b69707551f1b87a',
        'ukbench09268.jpg': 'ac9c72f8e1c2c448'
    }
    dummy_hasher = Hashing()
    dummy_result = ResultSet(dummy_db, dummy_query, dummy_hasher.hamming_distance, cutoff=3)
    dummy_distances = [max(dist) for dist in dummy_result.query_distances.values()]
    print(dummy_distances)
    assert max(dummy_distances) == 3


def test_max_hamming_threshold_not_violated(
        dummy_query={'ukbench00120.jpg': '2b69707551f1b87a', 'ukbench09268.jpg': 'ac9c72f8e1c2c448'}):
    dummy_db = {
        'ukbench00120_hflip.jpg': '2b69f1517570e2a1',
        'ukbench00120_resize.jpg': '2b69707551f1b87a',
        'ukbench09268.jpg': 'ac9c72f8e1c2c448'
    }
    dummy_hasher = Hashing()
    dummy_result = ResultSet(dummy_db, dummy_query, dummy_hasher.hamming_distance, search_method='brute_force')
    dummy_distances = [max(dist) for dist in dummy_result.query_distances.values()]
    assert max(dummy_distances) < 5


def test_identical_hash_consistency(dummy_image={'ukbench09060.jpg': 'e064ece078d7c96a'}):
    dummy_hasher = Hashing()
    dummy_result = ResultSet(dummy_image, dummy_image, dummy_hasher.hamming_distance)
    dummy_distances = [max(dist) for dist in dummy_result.query_distances.values()]
    assert set(dummy_distances) == {0}


def test_save_results(dummy_file='test_save_results.pkl'):
    if os.path.exists(dummy_file):
        os.remove(dummy_file)
    dummy_data = "010101"
    with open(dummy_file, 'w') as wh:
        wh.write(dummy_data)
    assert os.path.exists(dummy_file)


def test_saved_results_consistency(dummy_file='test_saved_results_consistency.pkl'):
    if os.path.exists(dummy_file):
        os.remove(dummy_file)
    dummy_data = "010101"
    with open(dummy_file, 'w') as wh:
        wh.write(dummy_data)
    with open(dummy_file, 'r') as rh:
        read_data = rh.read()
    assert read_data == dummy_data
