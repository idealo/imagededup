from imagededup.evaluation.evaluation import evaluate, _align_maps
import pickle
import pytest
from pathlib import Path

p = Path(__file__)

PATH_GROUND_TRUTH = p.parent / "data/evaluation_files/ground_truth.pkl"
PATH_ALL_CORRECT_RETRIEVALS = (
    p.parent / "data/evaluation_files/all_correct_retrievals.pkl"
)
PATH_INCORRECT_RETRIEVALS = p.parent / "data/evaluation_files/incorrect_retrievals.pkl"


def load_pickle(filename):
    with open(filename, "rb") as f:
        dict_loaded = pickle.load(f)
    return dict_loaded


def return_ground_all_correct_retrievals():
    return load_pickle(PATH_GROUND_TRUTH), load_pickle(PATH_ALL_CORRECT_RETRIEVALS)


def return_ground_incorrect_retrievals():
    return load_pickle(PATH_GROUND_TRUTH), load_pickle(PATH_INCORRECT_RETRIEVALS)


def test__align_maps_same_keys():
    ground_truth_map, retrieve_map = return_ground_all_correct_retrievals()
    duplicate_map = _align_maps(ground_truth_map, retrieve_map)
    assert set(duplicate_map.keys()) == set(ground_truth_map.keys())
    for i, j in zip(ground_truth_map.keys(), duplicate_map.keys()):
        assert i == j


def test__align_maps_different_keys():
    ground_truth_map = load_pickle(PATH_GROUND_TRUTH)
    red_retrieve_map = {"ukbench09060.jpg": ground_truth_map["ukbench09060.jpg"]}
    with pytest.raises(AssertionError):
        _align_maps(ground_truth_map, red_retrieve_map)


def test_default_returns_all_metrics(mocker):
    ground_truth_map, retrieve_map = return_ground_all_correct_retrievals()
    get_all_metrics_mocker = mocker.patch(
        "imagededup.evaluation.evaluation.get_all_metrics"
    )
    evaluate(ground_truth_map, retrieve_map)
    get_all_metrics_mocker.assert_called_once_with(ground_truth_map, retrieve_map)


def test_wrong_metric_raises_valueerror():
    ground_truth_map, retrieve_map = return_ground_all_correct_retrievals()
    with pytest.raises(ValueError):
        evaluate(ground_truth_map, retrieve_map, metric="bla")


@pytest.mark.parametrize("metric_name", ["map", "ndcg", "jaccard"])
def test_correct_call_to_mean_metric(mocker, metric_name):
    ground_truth_map, retrieve_map = return_ground_all_correct_retrievals()
    mean_metric_mocker = mocker.patch("imagededup.evaluation.evaluation.mean_metric")
    evaluate(ground_truth_map, retrieve_map, metric=metric_name)
    mean_metric_mocker.assert_called_once_with(
        ground_truth_map, retrieve_map, metric=metric_name
    )


@pytest.mark.parametrize("metric_name", ["MAP", "Ndcg", "JacCard"])
def test_correct_call_to_mean_metric_mixed_cases(mocker, metric_name):
    ground_truth_map, retrieve_map = return_ground_all_correct_retrievals()
    mean_metric_mocker = mocker.patch("imagededup.evaluation.evaluation.mean_metric")
    evaluate(ground_truth_map, retrieve_map, metric=metric_name)
    mean_metric_mocker.assert_called_once_with(
        ground_truth_map, retrieve_map, metric=metric_name.lower()
    )


# Integration tests


@pytest.mark.parametrize(
    "metric, expected_value",
    [("map", 0.5555555555555556), ("ndcg", 0.6173196815056892), ("jaccard", 0.6)],
)
def test_correct_values(metric, expected_value):
    """Tests if correct MAP values are computed
    Load ground truth and dict for incorrect map prediction to have a Map less than 1.0"""
    ground_truth, retrieved = return_ground_incorrect_retrievals()
    score = evaluate(ground_truth, retrieved, metric=metric)
    assert isinstance(score, dict)
    assert list(score.keys())[0] == metric
    assert score[metric] == expected_value


def test_correct_values_all():
    ground_truth, retrieved = return_ground_incorrect_retrievals()
    score = evaluate(ground_truth, retrieved)
    assert isinstance(score, dict)
    assert set(score.keys()) == set({"map", "ndcg", "jaccard"})
    for v in score.values():
        assert v is not None
