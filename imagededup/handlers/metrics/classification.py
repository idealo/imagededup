import itertools
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
)
from typing import List, Dict


def _order_tuples(unique_tuples):
    ordered_tuples = []

    for i in unique_tuples:
        ordered_tuples.append(tuple(sorted(i)))
    return ordered_tuples


def _get_unique_tuples(tuple_list):
    return [tuple(x) for x in set(map(frozenset, tuple_list))]


def _make_all_unique_possible_pairs(in_dict):
    # get all elements of the dictionary
    all_files = list(in_dict.keys())

    # make all possible pairs (remove pairs with same elements)
    all_tuples = []

    for i in itertools.product(all_files, all_files):
        if not i[0] == i[1]:
            all_tuples.append(i)

    # get unique pairs by disregrading order
    all_unique_tuples = _get_unique_tuples(all_tuples)

    # sort alphabetically the elements in pairs
    return _order_tuples(all_unique_tuples)


def _make_duplicate_pairs(in_dict):
    valid_pairs = []

    for k, v in in_dict.items():
        for j in v:
            valid_pairs.append((k, j))

    unique_tuples = _get_unique_tuples(valid_pairs)
    return _order_tuples(unique_tuples)


def _get_labels(complete_pairs, ground_truth_pairs, retrieved_pairs):
    y_true = [1 if i in ground_truth_pairs else 0 for i in complete_pairs]
    y_pred = [1 if i in retrieved_pairs else 0 for i in complete_pairs]
    return y_true, y_pred


def classification_metrics(ground_truth: Dict, retrieved: Dict):
    all_pairs = _make_all_unique_possible_pairs(ground_truth, retrieved)
    ground_truth_duplicate_pairs = _make_duplicate_pairs(ground_truth)
    retrieved_duplicate_pairs = _make_duplicate_pairs(retrieved)
    y_true, y_pred = _get_labels(
        all_pairs, ground_truth_duplicate_pairs, retrieved_duplicate_pairs
    )
    print(classification_report(y_true, y_pred))
    return precision_recall_fscore_support(y_true, y_pred)
