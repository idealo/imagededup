import itertools
from typing import Dict, List, Tuple

import numpy as np

from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
)
from imagededup.utils.logger import return_logger

logger = return_logger(__name__)


def _get_unique_ordered_tuples(unique_tuples: List[Tuple]) -> List[Tuple]:
    """Sort each tuple given a list of tuples and retain only unique pairs regardless of order within the tuple.
    Eg: [(2, 1), (1, 2), (3, 4)]  becomes [(1, 2), (3, 4)]"""

    return list(set([tuple(sorted(i)) for i in unique_tuples]))


def _make_all_unique_possible_pairs(ground_truth_dict: Dict) -> List[Tuple]:
    """
    Given a ground truth dictionary, generate all possible unique image pairs (both negative and positive pairs).
    """
    # get all elements of the dictionary
    all_files = list(ground_truth_dict.keys())

    # make all possible pairs (remove pairs with same elements)
    all_tuples = [i for i in itertools.product(all_files, all_files) if i[0] != i[1]]
    return _get_unique_ordered_tuples(all_tuples)


def _make_positive_duplicate_pairs(ground_truth: Dict, retrieved: Dict) -> List[Tuple]:
    """
    Given ground_truth and retrieved dictionary, generate all unique positive pairs.
    """
    pairs = []

    for mapping in [ground_truth, retrieved]:
        valid_pairs = []

        for k, v in mapping.items():
            valid_pairs.extend(list(zip([k]*len(v), v)))
        pairs.append(_get_unique_ordered_tuples(valid_pairs))

    return pairs[0], pairs[1]


def _prepare_labels(
    complete_pairs: List[Tuple],
    ground_truth_pairs: List[Tuple],
    retrieved_pairs: List[Tuple],
) -> Tuple[List, List]:
    """
    Given all possible unique pairs, ground truth positive pairs and retrieved positive pairs, generate true and
    predicted labels to feed into classification metrics functions.
    """
    ground_truth_pairs = set(ground_truth_pairs)
    retrieved_pairs = set(retrieved_pairs)

    y_true = [1 if i in ground_truth_pairs else 0 for i in complete_pairs]
    y_pred = [1 if i in retrieved_pairs else 0 for i in complete_pairs]
    return y_true, y_pred


def classification_metrics(ground_truth: Dict, retrieved: Dict) -> np.ndarray:
    """
    Given ground truth dictionary and retrieved dictionary, return per class precision, recall and f1 score. Class 1 is
    assigned to duplicate file pairs while class 0 is for non-duplicate file pairs.

    Args:
        ground_truth: A dictionary representing ground truth with filenames as key and a list of duplicate filenames
        as value.
        retrieved: A dictionary representing retrieved duplicates with filenames as key and a list of retrieved
        duplicate filenames as value.

    Returns:
        Dictionary of precision, recall and f1 score for both classes.
    """
    all_pairs = _make_all_unique_possible_pairs(ground_truth)
    ground_truth_duplicate_pairs, retrieved_duplicate_pairs = _make_positive_duplicate_pairs(
        ground_truth, retrieved
    )
    y_true, y_pred = _prepare_labels(
        all_pairs, ground_truth_duplicate_pairs, retrieved_duplicate_pairs
    )
    logger.info(classification_report(y_true, y_pred))
    prec_rec_fscore_support = dict(
        zip(
            ('precision', 'recall', 'f1_score', 'support'),
            precision_recall_fscore_support(y_true, y_pred),
        )
    )
    return prec_rec_fscore_support
