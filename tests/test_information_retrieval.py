from imagededup.handlers.metrics.information_retrieval import *
import os
import pickle
import pytest
"""Run from project root with: python -m pytest -vs tests/test_information_retrieval.py 
--cov=imagededup.handlers.metrics.information_retrieval"""


def load_pickle(filename):
    """The path of the file below is set since the test suite is run using python -m pytest command from the image-dedup
    directory"""
    with open(os.path.join('tests', 'data', filename), 'rb') as f:
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


@pytest.mark.parametrize("metric_function, expected_value", [(avg_prec, 0.6041666666666666), (ndcg, 0.75369761125927),
                                                        (jaccard_similarity, 0.6)])
def test_metrics_same_number_of_retrievals(metric_function, expected_value):
    """Number of retrievals = Number of ground truth retrievals"""
    corr_dup, ret_dups = initialize_fake_data_retrieved_same()
    avg_val = metric_function(corr_dup, ret_dups)
    assert avg_val == expected_value


@pytest.mark.parametrize("metric_function, expected_value", [(avg_prec, 0.25), (ndcg, 0.6131471927654584),
                                                        (jaccard_similarity, 0.2)])
def test_metrics_less_number_of_retrievals(metric_function, expected_value):
    """Number of retrievals < Number of ground truth retrievals"""
    corr_dup, ret_dups = initialize_fake_data_retrieved_less()
    avg_val = metric_function(corr_dup, ret_dups)
    assert avg_val == expected_value

@pytest.mark.parametrize("metric_function, expected_value", [(avg_prec, 0.8041666666666667), (ndcg, 0.9047172294870751),
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


@pytest.mark.parametrize("metric, expected_value", [('map', 0.5555555555555556), ('ndcg', 0.6173196815056892),
                                                        ('jaccard', 0.6)])
def test_metric_is_not_1_for_incorrect(metric, expected_value):
    """Tests if correct MAP values are computed
    Load ground truth and dict for incorrect map prediction to have a Map less than 1.0"""
    ground_truth = load_pickle('ground_truth.pkl')
    retrieved = load_pickle('incorrect_retrievals.pkl')
    metric_val = mean_metric(ground_truth, retrieved, metric=metric)
    assert metric_val == expected_value


def test_all_metrics_1_for_all_correct_retrievals():
    ground_truth = load_pickle('ground_truth.pkl')
    retrieved = load_pickle('all_correct_retrievals.pkl')
    metrics = get_all_metrics(ground_truth, retrieved)
    for i in metrics.values():
        assert i == 1.0


def test_get_metrics_returns_dict():
    ground_truth = load_pickle('ground_truth.pkl')
    retrieved = load_pickle('incorrect_retrievals.pkl')
    assert isinstance(get_all_metrics(ground_truth, retrieved), dict)
    assert len(get_all_metrics(ground_truth, retrieved).values()) == 3  # 3 metrics



