import pickle
import pytest
from pathlib import Path

from imagededup.handlers.metrics.information_retrieval import *

p = Path(__file__)

PATH_GROUND_TRUTH = p.parent / 'data/evaluation_files/ground_truth.pkl'
PATH_ALL_CORRECT_RETRIEVALS = (
    p.parent / 'data/evaluation_files/all_correct_retrievals.pkl'
)
PATH_INCORRECT_RETRIEVALS = p.parent / 'data/evaluation_files/incorrect_retrievals.pkl'


def return_ground_all_correct_retrievals():
    return load_pickle(PATH_GROUND_TRUTH), load_pickle(PATH_ALL_CORRECT_RETRIEVALS)


def return_ground_incorrect_retrievals():
    return load_pickle(PATH_GROUND_TRUTH), load_pickle(PATH_INCORRECT_RETRIEVALS)


def load_pickle(filename):
    """The path of the file below is set since the test suite is run using python -m pytest command from the image-dedup
    directory"""
    with open(filename, 'rb') as f:
        dict_loaded = pickle.load(f)
    return dict_loaded


def initialize_fake_data_retrieved_same():
    """Number of retrievals = Number of ground truth retrievals"""
    corr_dup = ['1.jpg', '2.jpg', '3.jpg', '4.jpg']
    ret_dups = ['1.jpg', '33.jpg', '2.jpg', '4.jpg']
    return corr_dup, ret_dups


def initialize_fake_data_retrieved_less():
    """Number of retrievals < Number of ground truth retrievals"""
    corr_dup = ['1.jpg', '2.jpg', '3.jpg', '4.jpg']
    ret_dups = ['1.jpg', '42.jpg']
    return corr_dup, ret_dups


def initialize_fake_data_retrieved_more():
    """Number of retrievals > Number of ground truth retrievals"""
    corr_dup = ['1.jpg', '2.jpg', '3.jpg', '4.jpg']
    ret_dups = ['1.jpg', '42.jpg', '2.jpg', '3.jpg', '4.jpg']
    return corr_dup, ret_dups


@pytest.mark.parametrize('metric_function, expected_value', [(avg_prec, 0.6041666666666666), (ndcg, 0.9060254355346823),
                                                        (jaccard_similarity, 0.6)])
def test_metrics_same_number_of_retrievals(metric_function, expected_value):
    """Number of retrievals = Number of ground truth retrievals"""
    corr_dup, ret_dups = initialize_fake_data_retrieved_same()
    avg_val = metric_function(corr_dup, ret_dups)
    assert avg_val == expected_value


@pytest.mark.parametrize('metric_function, expected_value', [(avg_prec, 0.25), (ndcg, 1.0),
                                                        (jaccard_similarity, 0.2)])
def test_metrics_less_number_of_retrievals(metric_function, expected_value):
    """Number of retrievals < Number of ground truth retrievals"""
    corr_dup, ret_dups = initialize_fake_data_retrieved_less()
    avg_val = metric_function(corr_dup, ret_dups)
    assert avg_val == expected_value

@pytest.mark.parametrize('metric_function, expected_value', [(avg_prec, 0.8041666666666667), (ndcg, 0.9047172294870751),
                                                        (jaccard_similarity, 0.8)])
def test_metrics_more_number_of_retrievals(metric_function, expected_value):
    """Number of retrievals > Number of ground truth retrievals"""
    corr_dup, ret_dups = initialize_fake_data_retrieved_more()
    avg_val = metric_function(corr_dup, ret_dups)
    assert avg_val == expected_value


@pytest.mark.parametrize('metric_func', [avg_prec, ndcg, jaccard_similarity])
def test_zero_retrieval(metric_func):
    corr_dup = ['1.jpg', '2.jpg', '3.jpg', '4.jpg']
    ret_dups = []
    av_val = metric_func(corr_dup, ret_dups)
    assert av_val == 0.0

@pytest.mark.parametrize('metric_function, expected_value', [(avg_prec, 1.0), (ndcg, 1.0),
                                                        (jaccard_similarity, 1.0)])
def test_zero_correct_and_zero_retrieved(metric_function, expected_value):
    corr_dup = []
    ret_dups = []
    assert metric_function(corr_dup, ret_dups) == expected_value


@pytest.mark.parametrize('metric_function, expected_value', [(avg_prec, 0.0), (ndcg, 0.0),
                                                        (jaccard_similarity, 0.0)])
def test_zero_correct_and_one_retrieved(metric_function, expected_value):
    corr_dup = []
    ret_dups = ['1']
    assert metric_function(corr_dup, ret_dups) == expected_value

@pytest.mark.parametrize('metric, expected_value', [('map', 0.5555555555555556), ('ndcg', 0.75),
                                                        ('jaccard', 0.6)])
def test_metric_is_not_1_for_incorrect(metric, expected_value):
    """Tests if correct MAP values are computed
    Load ground truth and dict for incorrect map prediction to have a Map less than 1.0"""
    ground_truth, retrieved = return_ground_incorrect_retrievals()
    metric_val = mean_metric(ground_truth, retrieved, metric=metric)
    assert metric_val == expected_value


def test_all_metrics_1_for_all_correct_retrievals():
    ground_truth, retrieved = return_ground_all_correct_retrievals()
    metrics = get_all_metrics(ground_truth, retrieved)
    for i in metrics.values():
        assert i == 1.0


def test_get_metrics_returns_dict():
    ground_truth, retrieved = return_ground_incorrect_retrievals()
    assert isinstance(get_all_metrics(ground_truth, retrieved), dict)
    assert len(get_all_metrics(ground_truth, retrieved).values()) == 3  # 3 metrics




