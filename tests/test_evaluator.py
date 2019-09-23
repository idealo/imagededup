import pytest

import numpy as np

from imagededup.evaluation.evaluation import (
    evaluate,
    _check_map_correctness,
    _transpose_checker,
)


def return_ground_all_correct_retrievals():
    ground_truth = {'1': ['2', '3', '4'], '2': ['1', '3'], '3': ['1', '2'], '4': ['1']}
    retrieved = ground_truth
    return ground_truth, retrieved


def return_ground_incorrect_retrievals():
    ground_truth = {'1': ['2', '3', '4'], '2': ['1', '3'], '3': ['1', '2'], '4': ['1']}
    retrieved = {'1': ['2', '3'], '2': ['1'], '3': ['1'], '4': []}
    return ground_truth, retrieved


def test__transpose_checker():
    mapping_non_transpose = {
        '1': ['2', '3', '4'],
        '2': ['1', '3'],
        '3': ['1', '2'],
        '4': [],
    }
    with pytest.raises(AssertionError):
        _transpose_checker(mapping_non_transpose)


def test__check_map_correctness_extra_gt_vals():
    ground_truth_map, retrieved_map = return_ground_all_correct_retrievals()
    ground_truth_map['1'].append('20')
    with pytest.raises(AssertionError):
        _check_map_correctness(ground_truth_map, retrieved_map)


def test__check_map_correctness_extra_ret_vals():
    ground_truth_map, retrieved_map = return_ground_all_correct_retrievals()
    retrieved_map['1'].append('20')
    with pytest.raises(AssertionError):
        _check_map_correctness(ground_truth_map, retrieved_map)


def test__check_map_correctness_different_keys():
    ground_truth_map = {'1': ['2'], '2': ['1']}
    retrieve_map = {'2': ['3'], '3': ['2']}
    with pytest.raises(Exception):
        _check_map_correctness(ground_truth_map, retrieve_map)


def test_default_returns_all_metrics(mocker):
    ground_truth_map, retrieve_map = return_ground_all_correct_retrievals()
    get_all_metrics_mocker = mocker.patch(
        'imagededup.evaluation.evaluation.get_all_metrics'
    )
    classification_metrics_mocker = mocker.patch(
        'imagededup.evaluation.evaluation.classification_metrics'
    )
    classification_metrics_mocker = mocker.patch(
        'imagededup.evaluation.evaluation.classification_metrics'
    )
    evaluate(ground_truth_map, retrieve_map)
    get_all_metrics_mocker.assert_called_once_with(ground_truth_map, retrieve_map)
    classification_metrics_mocker.assert_called_once_with(
        ground_truth_map, retrieve_map
    )


def test_wrong_metric_raises_valueerror():
    ground_truth_map, retrieve_map = return_ground_all_correct_retrievals()
    with pytest.raises(ValueError):
        evaluate(ground_truth_map, retrieve_map, metric='bla')


@pytest.mark.parametrize('metric_name', ['map', 'ndcg', 'jaccard'])
def test_correct_call_to_mean_metric(mocker, metric_name):
    ground_truth_map, retrieve_map = return_ground_all_correct_retrievals()
    mean_metric_mocker = mocker.patch('imagededup.evaluation.evaluation.mean_metric')
    evaluate(ground_truth_map, retrieve_map, metric=metric_name)
    mean_metric_mocker.assert_called_once_with(
        ground_truth_map, retrieve_map, metric=metric_name
    )


def test_correct_call_to_classification_metric(mocker):
    ground_truth_map, retrieve_map = return_ground_all_correct_retrievals()
    classification_metrics_mocker = mocker.patch(
        'imagededup.evaluation.evaluation.classification_metrics'
    )
    evaluate(ground_truth_map, retrieve_map, metric='classification')
    classification_metrics_mocker.assert_called_once_with(
        ground_truth_map, retrieve_map
    )


@pytest.mark.parametrize('metric_name', ['MAP', 'Ndcg', 'JacCard'])
def test_correct_call_to_mean_metric_mixed_cases(mocker, metric_name):
    ground_truth_map, retrieve_map = return_ground_all_correct_retrievals()
    mean_metric_mocker = mocker.patch('imagededup.evaluation.evaluation.mean_metric')
    evaluate(ground_truth_map, retrieve_map, metric=metric_name)
    mean_metric_mocker.assert_called_once_with(
        ground_truth_map, retrieve_map, metric=metric_name.lower()
    )


def test_correct_call_to_classification_metric_mixed_case(mocker):
    ground_truth_map, retrieve_map = return_ground_all_correct_retrievals()
    classification_metrics_mocker = mocker.patch(
        'imagededup.evaluation.evaluation.classification_metrics'
    )
    evaluate(ground_truth_map, retrieve_map, metric='Classification')
    classification_metrics_mocker.assert_called_once_with(
        ground_truth_map, retrieve_map
    )


# Integration tests


@pytest.mark.parametrize(
    'metric, expected_value',
    [('map', 0.41666666666666663), ('ndcg', 0.75), ('jaccard', 0.41666666666666663)],
)
def test_correct_values_ir(metric, expected_value):
    """Tests if correct MAP values are computed
    Load ground truth and dict for incorrect map prediction to have a Map less than 1.0"""
    ground_truth, retrieved = return_ground_incorrect_retrievals()
    score = evaluate(ground_truth, retrieved, metric=metric)
    assert isinstance(score, dict)
    assert list(score.keys())[0] == metric
    assert score[metric] == expected_value


def test_correct_values_classification():
    ground_truth, retrieved = return_ground_incorrect_retrievals()
    expected_return = {
        'precision': np.array([0.5, 1.0]),
        'recall': np.array([1.0, 0.5]),
        'f1_score': np.array([0.66666667, 0.66666667]),
        'support': np.array([2, 4]),
    }
    score = evaluate(ground_truth, retrieved, metric='classification')
    assert isinstance(score, dict)
    assert set(score.keys()) == set(['precision', 'recall', 'f1_score', 'support'])
    for k, v in score.items():
        assert isinstance(v, np.ndarray)
        np.testing.assert_almost_equal(score[k], expected_return[k])


def test_correct_values_all():
    ground_truth, retrieved = return_ground_incorrect_retrievals()
    score = evaluate(ground_truth, retrieved)
    assert isinstance(score, dict)
    assert set(score.keys()) == set(
        ['map', 'ndcg', 'jaccard', 'precision', 'recall', 'f1_score', 'support']
    )
    for v in score.values():
        assert v is not None
