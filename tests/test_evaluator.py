from imagededup.evaluation.evaluation import evaluate, _align_maps
import os
import pickle
import pytest

def load_pickle(filename):
    """The path of the file below is set since the test suite is run using python -m pytest command from the image-dedup
    directory"""
    with open(os.path.join('tests', 'data', filename), 'rb') as f:
        dict_loaded = pickle.load(f)
    return dict_loaded


def test__align_maps_same_keys():
    ground_truth_map = load_pickle('ground_truth.pkl')
    retrieve_map = load_pickle('all_correct_retrievals.pkl')
    duplicate_map = _align_maps(ground_truth_map, retrieve_map)
    assert set(duplicate_map.keys()) == set(ground_truth_map.keys())
    for i, j in zip(ground_truth_map.keys(), duplicate_map.keys()):
        assert i == j


def test__align_maps_different_keys():
    ground_truth_map = load_pickle('ground_truth.pkl')
    red_retrieve_map = {'ukbench09060.jpg': ground_truth_map['ukbench09060.jpg']}
    with pytest.raises(AssertionError):
        _align_maps(ground_truth_map, red_retrieve_map)


def test_default_returns_all_metrics(mocker):
    ground_truth_map = load_pickle('ground_truth.pkl')
    retrieve_map = load_pickle('all_correct_retrievals.pkl')
    get_all_metrics_mocker = mocker.patch('imagededup.evaluation.evaluation.get_all_metrics')
    evaluate(ground_truth_map, retrieve_map)
    get_all_metrics_mocker.assert_called_once_with(ground_truth_map, retrieve_map)


def test_wrong_metric_raises_valueerror():
    ground_truth_map = load_pickle('ground_truth.pkl')
    retrieve_map = load_pickle('all_correct_retrievals.pkl')
    with pytest.raises(ValueError):
        evaluate(ground_truth_map, retrieve_map, metric='bla')


@pytest.mark.parametrize('metric_name', ['map', 'ndcg', 'jaccard'])
def test_correct_call_to_mean_metric(mocker, metric_name):
    ground_truth_map = load_pickle('ground_truth.pkl')
    retrieve_map = load_pickle('all_correct_retrievals.pkl')
    mean_metric_mocker = mocker.patch('imagededup.evaluation.evaluation.mean_metric')
    evaluate(ground_truth_map, retrieve_map, metric=metric_name)
    mean_metric_mocker.assert_called_once_with(ground_truth_map, retrieve_map, metric=metric_name)


@pytest.mark.parametrize('metric_name', ['MAP', 'Ndcg', 'JacCard'])
def test_correct_call_to_mean_metric_mixed_cases(mocker, metric_name):
    ground_truth_map = load_pickle('ground_truth.pkl')
    retrieve_map = load_pickle('all_correct_retrievals.pkl')
    mean_metric_mocker = mocker.patch('imagededup.evaluation.evaluation.mean_metric')
    evaluate(ground_truth_map, retrieve_map, metric=metric_name)
    mean_metric_mocker.assert_called_once_with(ground_truth_map, retrieve_map, metric=metric_name.lower())

# Integration tests


@pytest.mark.parametrize("metric, expected_value", [('map', 0.5555555555555556), ('ndcg', 0.6173196815056892),
                                                        ('jaccard', 0.6)])
def test_correct_values(metric, expected_value):
    """Tests if correct MAP values are computed
    Load ground truth and dict for incorrect map prediction to have a Map less than 1.0"""
    ground_truth = load_pickle('ground_truth.pkl')
    retrieved = load_pickle('incorrect_retrievals.pkl')
    score = evaluate(ground_truth, retrieved, metric=metric)
    assert isinstance(score, dict)
    assert list(score.keys())[0] == metric
    assert score[metric] == expected_value


def test_correct_values_all():
    ground_truth = load_pickle('ground_truth.pkl')
    retrieved = load_pickle('incorrect_retrievals.pkl')
    score = evaluate(ground_truth, retrieved)
    assert isinstance(score, dict)
    assert set(score.keys()) == set({'map', 'ndcg', 'jaccard'})
    for v in score.values():
        assert v is not None