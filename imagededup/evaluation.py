import pickle
import numpy as np
from types import FunctionType


class EvalPerformance:
    def __init__(self, dict_correct: dict, dict_retrieved: dict) -> None:
        self.dict_correct = dict_correct # dict of correct retrievals for each query(= ground truth), {'1.jpg': 'correct_dup1.jpg'}
        self.dict_retrieved = dict_retrieved # dict of all retrievals for each query, {'1.jpg': 'retrieval_1.jpg'}

    @staticmethod
    def avg_prec(correct_duplicates: list, retrieved_duplicates: list) -> float:
        """Input: (list of correct duplicates (i.e., ground truth), list of retrieved duplicates) for one single query
        return: float representing average precision for one input query"""

        count_real_correct = len(correct_duplicates)
        relevance = np.array([1 if i in correct_duplicates else 0 for i in retrieved_duplicates])
        relevance_cumsum = np.cumsum(relevance)
        prec_k = [relevance_cumsum[k] / (k + 1) for k in range(len(relevance))]
        prec_and_relevance = [relevance[k] * prec_k[k] for k in range(len(relevance))]
        avg_precision = np.sum(prec_and_relevance) / count_real_correct
        return avg_precision

    @staticmethod
    def ndcg(correct_duplicates: list, retrieved_duplicates: list) -> float:
        """Input: (list of correct duplicates (i.e., ground truth), list of retrieved duplicates) for one single query
                return: float representing Normalized discounted Cumulative Gain (NDCG) for one input query"""
        relevance = np.array([1 if i in correct_duplicates else 0 for i in retrieved_duplicates])
        relevance_numerator = [2 ** (k) - 1 for k in relevance]
        relevance_denominator = [np.log2(k + 2) for k in
                                 range(len(relevance))]  # first value of denominator term should be 2

        dcg_terms = [relevance_numerator[k] / relevance_denominator[k] for k in range(len(relevance))]
        dcg_k = np.sum(dcg_terms)

        # get #retrievals
        # if #retrievals <= #ground truth retrievals, set score=1 for calculating idcg
        # else score=1 for first #ground truth retrievals entries, score=0 for remaining positions

        if len(dcg_terms) <= len(correct_duplicates):
            ideal_dcg = np.sum([1 / np.log2(k + 2) for k in range(len(dcg_terms))])
            ndcg = dcg_k / ideal_dcg
        else:
            ideal_dcg_terms = [1] * len(correct_duplicates) + [0] * (len(dcg_terms) - len(correct_duplicates))
            ideal_dcg_numerator = [(2 ** ideal_dcg_terms[k]) - 1 for k in range(len(ideal_dcg_terms))]
            ideal_dcg_denominator = [np.log2(k + 2) for k in range(len(ideal_dcg_terms))]
            ideal_dcg = np.sum([ideal_dcg_numerator[k] / ideal_dcg_denominator[k] for k in range(len(ideal_dcg_numerator))])
            ndcg = dcg_k / ideal_dcg
        return ndcg

    @staticmethod
    def jaccard_similarity(correct_duplicates: list, retrieved_duplicates: list) -> float:
        """Input: (list of correct duplicates (i.e., ground truth), list of retrieved duplicates) for one single query
                return: float representing jaccard similarity for one input query"""
        set_correct_duplicates = set(correct_duplicates)
        set_retrieved_duplicates = set(retrieved_duplicates)

        intersection_dups = set_retrieved_duplicates.intersection(set_correct_duplicates)
        union_dups = set_retrieved_duplicates.union(set_correct_duplicates)

        jacc_sim = len(intersection_dups) / len(union_dups)
        return jacc_sim

    def mean_all_func(self, metric_func: FunctionType) -> float:
        """Input: metric function on which mean is to be calculated across all queries
                return: float representing mean of the metric across all queries"""
        all_metrics = []
        for k in self.dict_correct.keys():
            all_metrics.append(metric_func(self.dict_correct[k], self.dict_retrieved[k]))
        return np.mean(all_metrics)

    def get_all_metrics(self, save: bool=True) -> dict:
        """Input: Save flag indicating whether the dictionary below should be saved
        return: dictionary of all mean metrics"""
        dict_average_metrics = {
            'MAP': self.mean_all_func(self.avg_prec),
            'NDCG': self.mean_all_func(self.ndcg),
            'Jaccard': self.mean_all_func(self.jaccard_similarity)
        }

        if save:
            with open('all_average_metrics.pkl', 'wb') as f:
                pickle.dump(dict_average_metrics, f)
        return dict_average_metrics

