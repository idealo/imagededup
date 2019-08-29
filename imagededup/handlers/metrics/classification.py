import itertools
import numpy as np
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
)
from typing import List, Dict, Tuple


def _order_tuples(unique_tuples: List[Tuple]) -> List[Tuple]:
    """Sorts each tuple given a list of tuples.
    Eg: [(2, 1), (3, 4)]  becomes [(1, 2), (3, 4)]"""
    ordered_tuples = []

    for i in unique_tuples:
        ordered_tuples.append(tuple(sorted(i)))
    return ordered_tuples


def _get_unique_tuples(tuple_list: List[Tuple]) -> List[Tuple]:
    """Gets unique tuples disregarding the order of elements in the tuple given a list of tuples.
        Eg: [(2, 1), (1, 2), (3, 4)]  becomes [(1, 2), (3, 4)] or [(2, 1), (3, 4)]"""
    return [tuple(x) for x in set(map(frozenset, tuple_list))]


def _make_all_unique_possible_pairs(ground_truth_dict: Dict) -> List[Tuple]:
    """
    Given a ground truth dictionary, generates all possible unique image pairs (both negative and positive pairs).
    """
    # get all elements of the dictionary
    all_files = list(ground_truth_dict.keys())

    # make all possible pairs (remove pairs with same elements)
    all_tuples = []

    for i in itertools.product(all_files, all_files):
        if not i[0] == i[1]:
            all_tuples.append(i)

    # get unique pairs by disregrading order
    all_unique_tuples = _get_unique_tuples(all_tuples)

    # sort alphabetically the elements in pairs
    return _order_tuples(all_unique_tuples)


def _make_duplicate_pairs(duplicate_mapping: Dict) -> List[Tuple]:
    """
    Given a dictionary, generates all unique positive pairs.
    """
    valid_pairs = []

    for k, v in duplicate_mapping.items():
        for j in v:
            valid_pairs.append((k, j))

    unique_tuples = _get_unique_tuples(valid_pairs)
    return _order_tuples(unique_tuples)


def _prepare_labels(
    complete_pairs: List[Tuple],
    ground_truth_pairs: List[Tuple],
    retrieved_pairs: List[Tuple],
) -> Tuple[List, List]:
    """
    Given all possible unique pairs, ground truth positive pairs and retrieved positive pairs, generates true and
    predicted labels to feed into classification metrics functions.
    """
    y_true = [1 if i in ground_truth_pairs else 0 for i in complete_pairs]
    y_pred = [1 if i in retrieved_pairs else 0 for i in complete_pairs]
    return y_true, y_pred


def classification_metrics(ground_truth: Dict, retrieved: Dict) -> np.ndarray:
    """
    Given ground truth dictionary and retrieved dictionary, returns per class precision, recall and f1 score.
    """
    all_pairs = _make_all_unique_possible_pairs(ground_truth, retrieved)
    ground_truth_duplicate_pairs = _make_duplicate_pairs(ground_truth)
    retrieved_duplicate_pairs = _make_duplicate_pairs(retrieved)
    y_true, y_pred = _prepare_labels(
        all_pairs, ground_truth_duplicate_pairs, retrieved_duplicate_pairs
    )
    print(classification_report(y_true, y_pred))
    return precision_recall_fscore_support(y_true, y_pred)
