from imagededup.handlers.metrics.information_retrieval import mean_metric, get_all_metrics
from typing import Dict


def _align_maps(ground_truth_map: Dict = None, duplicate_map: Dict = None):
    ground_truth_keys_set = set(ground_truth_map.keys())
    duplicate_map_keys_set = set(duplicate_map.keys())

    if not ground_truth_keys_set == duplicate_map_keys_set:
        diff = ground_truth_keys_set.difference(duplicate_map_keys_set)
        assert diff == set(), f'Following keys not in duplicate_map:\n{diff}'
    duplicate_map = {k: duplicate_map[k] for k in ground_truth_map.keys()}
    return duplicate_map


def evaluate(ground_truth_map: Dict = None, duplicate_map: Dict = None, metric: str = 'all'):
    metric = metric.lower()
    duplicate_map = _align_maps(ground_truth_map, duplicate_map)

    if metric in ['map', 'ndcg', 'jaccard']:
        return {metric: mean_metric(ground_truth_map, duplicate_map, metric=metric)}
    elif metric == 'all':
        return get_all_metrics(ground_truth_map, duplicate_map)
    else:
        raise ValueError('Acceptable metrics are: \'map\', \'ndcg\', \'jaccard\', \'all\'')


