import pickle
import numpy as np


class EvalPerformance:
    def __init__(self, dict_correct, dict_retrieved):
        self.dict_correct =  dict_correct
        self.dict_retrieved  = dict_retrieved

    @staticmethod
    def avg_prec(correct_duplicates, retrieved_duplicates):
        count_real_correct = len(correct_duplicates)
        relevance = np.array([1 if i in correct_duplicates else 0 for i in retrieved_duplicates])
        relevance_cumsum = np.cumsum(relevance)
        prec_k = [relevance_cumsum[k] / (k + 1) for k in range(len(relevance))]
        prec_and_relevance = [relevance[k] * prec_k[k] for k in range(len(relevance))]
        avg_precision = np.sum(prec_and_relevance) / count_real_correct
        return avg_precision

    @staticmethod
    def ndcg(correct_duplicates, retrieved_duplicates):
        relevance = np.array([1 if i in correct_duplicates else 0 for i in retrieved_duplicates])
        relevance_numerator = [2 ** (k) - 1 for k in relevance]
        relevance_denominator = [np.log2(k + 2) for k in
                                 range(len(relevance))]  # first value of denominator term should be 2

        dcg_terms = [relevance_numerator[k] / relevance_denominator[k] for k in range(len(relevance))]
        dcg_k = np.sum(dcg_terms)
        ideal_dcg = np.sum([1 / np.log2(k + 2) for k in range(len(relevance))])
        ndcg = dcg_k / ideal_dcg
        return ndcg

    @staticmethod
    def jaccard_similarity(correct_duplicates, retrieved_duplicates):
        set_correct_duplicates = set(correct_duplicates)
        set_retrieved_duplicates = set(retrieved_duplicates)

        intersection_dups = set_retrieved_duplicates.intersection(set_correct_duplicates)
        union_dups = set_retrieved_duplicates.union(set_correct_duplicates)

        jacc_sim = len(intersection_dups) / len(union_dups)
        return jacc_sim

    def mean_all_func(self, metric_func):
        all_metrics = []
        for k in self.dict_correct.keys():
            all_metrics.append(metric_func(self.dict_correct[k], self.dict_retrieved[k]))
        return np.mean(all_metrics)

    def get_all_metrics(self, save=True):
        dict_average_metrics = {
            'MAP': self.mean_all_func(self.avg_prec),
            'NDCG': self.mean_all_func(self.ndcg),
            'Jaccard': self.mean_all_func(self.jaccard_similarity)
        }

        if save:
            with open('all_average_metrics.pkl', 'wb') as f:
                pickle.dump(f)
        return dict_average_metrics

