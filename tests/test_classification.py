import numpy as np

from imagededup.handlers.metrics.classification import (
    _get_unique_ordered_tuples,
    _make_all_unique_possible_pairs,
    _make_positive_duplicate_pairs,
    _prepare_labels,
    classification_metrics,
)


def test__get_unique_ordered_tuples():
    non_unique_pairs = [('1', '3'), ('1', '4'), ('3', '1'), ('1', '3'), ('3', '4')]
    obtained_unique_ordered_pairs = _get_unique_ordered_tuples(non_unique_pairs)

    # test order
    for i in obtained_unique_ordered_pairs:
        assert i[0] <= i[1]

    # test membership
    set_list = []
    for j in obtained_unique_ordered_pairs:
        if set(j) not in set_list:
            set_list.append(set(j))
    assert len(set_list) == 3


def test__make_all_unique_possible_pairs(mocker):
    ground_truth = {'1': ['2', '3', '4'], '2': ['1', '3'], '3': ['1', '2'], '4': ['1']}
    all_pairs = [
        ('1', '2'),
        ('1', '3'),
        ('1', '4'),
        ('2', '1'),
        ('2', '3'),
        ('2', '4'),
        ('3', '1'),
        ('3', '2'),
        ('3', '4'),
        ('4', '1'),
        ('4', '2'),
        ('4', '3'),
    ]
    get_unique_ordered_tuples_mocker = mocker.patch(
        'imagededup.handlers.metrics.classification._get_unique_ordered_tuples'
    )
    _make_all_unique_possible_pairs(ground_truth)
    get_unique_ordered_tuples_mocker.assert_called_once_with(all_pairs)


def test__make_positive_duplicate_pairs(mocker):
    ground_truth = {'1': ['2', '3', '4'], '2': ['1', '3'], '3': ['1', '2'], '4': ['1']}
    valid_pairs = [
        ('1', '2'),
        ('1', '3'),
        ('1', '4'),
        ('2', '1'),
        ('2', '3'),
        ('3', '1'),
        ('3', '2'),
        ('4', '1'),
    ]
    get_unique_ordered_tuples_mocker = mocker.patch(
        'imagededup.handlers.metrics.classification._get_unique_ordered_tuples'
    )
    _make_positive_duplicate_pairs(ground_truth, ground_truth)
    get_unique_ordered_tuples_mocker.assert_called_with(valid_pairs)


def test__prepare_labels():
    all_possible_pairs = [
        ('1', '3'),
        ('2', '3'),
        ('1', '4'),
        ('2', '4'),
        ('1', '2'),
        ('3', '4'),
    ]
    ground_truth_pairs = [('1', '3'), ('2', '3'), ('1', '2'), ('1', '4')]
    retrieved_pairs = [('1', '3'), ('1', '2')]
    y_true_obtained, y_pred_obtained = _prepare_labels(
        all_possible_pairs, ground_truth_pairs, retrieved_pairs
    )
    assert y_true_obtained == [1, 1, 1, 0, 1, 0]
    assert y_pred_obtained == [1, 0, 0, 0, 1, 0]


def test_classification_metrics(mocker):
    ground_truth = {'1': ['2', '3', '4'], '2': ['1', '3'], '3': ['1', '2'], '4': ['1']}
    retrieved = {'1': ['2', '3'], '2': ['1'], '3': ['1'], '4': []}
    all_possible_pairs_ret = [
        ('1', '3'),
        ('2', '3'),
        ('1', '4'),
        ('2', '4'),
        ('1', '2'),
        ('3', '4'),
    ]
    ground_truth_pairs_ret = [('1', '3'), ('2', '3'), ('1', '2'), ('1', '4')]
    retrieved_pairs_ret = [('1', '3'), ('1', '2')]
    y_true_ret = [1, 1, 1, 0, 1, 0]
    y_pred_ret = [1, 0, 0, 0, 1, 0]

    make_all_unique_possible_pairs_mocker = mocker.patch(
        'imagededup.handlers.metrics.classification._make_all_unique_possible_pairs'
    )
    make_all_unique_possible_pairs_mocker.return_value = all_possible_pairs_ret

    make_positive_duplicate_pairs_mocker = mocker.patch(
        'imagededup.handlers.metrics.classification._make_positive_duplicate_pairs'
    )
    make_positive_duplicate_pairs_mocker.return_value = (
        ground_truth_pairs_ret,
        retrieved_pairs_ret,
    )
    prepare_labels_mocker = mocker.patch(
        'imagededup.handlers.metrics.classification._prepare_labels'
    )
    prepare_labels_mocker.return_value = y_true_ret, y_pred_ret
    precision_recall_fscore_support_mocker = mocker.patch(
        'imagededup.handlers.metrics.classification.precision_recall_fscore_support'
    )
    classification_metrics(ground_truth, retrieved)
    make_all_unique_possible_pairs_mocker.assert_called_once_with(ground_truth)
    make_positive_duplicate_pairs_mocker.assert_called_once_with(
        ground_truth, retrieved
    )
    prepare_labels_mocker.assert_called_once_with(
        all_possible_pairs_ret, ground_truth_pairs_ret, retrieved_pairs_ret
    )
    precision_recall_fscore_support_mocker.assert_called_once_with(
        y_true_ret, y_pred_ret
    )


# Integration test


def test_classification_metrics_integrated():
    ground_truth = {'1': ['2', '3', '4'], '2': ['1', '3'], '3': ['1', '2'], '4': ['1']}
    retrieved = {'1': ['2', '3'], '2': ['1'], '3': ['1'], '4': []}
    expected_return = {
        'precision': np.array([0.5, 1.0]),
        'recall': np.array([1.0, 0.5]),
        'f1_score': np.array([0.66666667, 0.66666667]),
        'support': np.array([2, 4]),
    }
    metrics = classification_metrics(ground_truth, retrieved)
    assert isinstance(metrics, dict)

    for k, v in metrics.items():
        assert isinstance(v, np.ndarray)
        np.testing.assert_almost_equal(metrics[k], expected_return[k])
