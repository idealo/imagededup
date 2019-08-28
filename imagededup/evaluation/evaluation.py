from imagededup.handlers.metrics.information_retrieval import (
    mean_metric,
    get_all_metrics,
)
from imagededup.handlers.metrics.classification import classification_metrics
from typing import Dict


def _check_map_completeness(ground_truth_map: Dict, duplicate_map: Dict):
    ground_truth_keys_set = set(ground_truth_map.keys())
    duplicate_map_keys_set = set(duplicate_map.keys())

    if not ground_truth_keys_set == duplicate_map_keys_set:
        diff = ground_truth_keys_set.symmetric_difference(duplicate_map_keys_set)
        raise Exception(f'Please ensure that ground truth and duplicate map have the same keys!'
                        f' Following keys uncommon between ground truth and duplicate_map:\n{diff}')


def evaluate(
    ground_truth_map: Dict = None, duplicate_map: Dict = None, metric: str = 'all'
):
    metric = metric.lower()
    _check_map_completeness(ground_truth_map, duplicate_map)

    if metric in ['map', 'ndcg', 'jaccard']:
        return {metric: mean_metric(ground_truth_map, duplicate_map, metric=metric)}
    elif metric == "classification":
        return classification_metrics(ground_truth_map, duplicate_map)
    elif metric == "all":
        return (
            get_all_metrics(ground_truth_map, duplicate_map),
            classification_metrics(ground_truth_map, duplicate_map),
        )
    else:
        raise ValueError("Acceptable metrics are: 'map', 'ndcg', 'jaccard', 'classification', 'all'")

