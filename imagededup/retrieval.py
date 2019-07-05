from imagededup.bktree import BKTree
from imagededup.brute_force import BruteForce
from imagededup.logger import return_logger
from types import FunctionType
from numpy.linalg import norm
from typing import Tuple, Dict
import os
import numpy as np
import pickle


class HashEval:
    def __init__(self, test: Dict, queries: Dict, hammer: FunctionType, cutoff: int = 5, search_method: str = 'bktree',
                 save: bool = False) -> None:
        """
        Initializes a HashEval object which offers an interface to control hashing and search methods for desired
        dataset. Computes a map of duplicate images in the document space given certain input control parameters.
        """
        self.candidates = test  # database
        self.queries = queries
        self.hamming_distance_invoker = hammer
        self.max_d = cutoff
        self.logger = return_logger(__name__, os.getcwd())
        if search_method == 'bktree':
            self.fetch_nearest_neighbors_bktree()  # bktree is the default search method
        else:
            self.fetch_nearest_neighbors_brute_force()
        if save:
            self.save_results()

    def _get_query_results(self, search_method_object):
        sorted_result_list, result_map = {}, {}
        for each in self.queries:
            res = search_method_object.search(query=self.queries[each], tol=self.max_d)
            if each in res:  # to avoid self retrieval
                res.pop(each)
            result_map[each] = res
            sorted_result_list[each] = sorted(res, key=lambda x: res[x], reverse=False)
        self.query_results_map = result_map
        self.query_results_list = sorted_result_list

    def fetch_nearest_neighbors_brute_force(self) -> None:
        """
        Wrapper function to retrieve results for all queries in dataset using brute-force search
        """
        self.logger.info('Start: Retrieving duplicates using Brute force algorithm')  # TODO: Add max hamming distance
        # after it is parmatrized

        bruteforce = BruteForce(self.candidates, self.hamming_distance_invoker)
        self._get_query_results(bruteforce)

    def fetch_nearest_neighbors_bktree(self) -> None:
        """
        Wrapper function to retrieve results for all queries in dataset using a BKTree search.
        """
        self.logger.info('Start: Retrieving duplicates using BKTree algorithm')  # TODO: Add max hamming distance after
        # it is parametrized
        built_tree = BKTree(self.candidates, self.hamming_distance_invoker)  # construct bktree
        self._get_query_results(built_tree)

    def retrieve_results(self) -> Dict:
        """
        Accessor function returning all results.

        :return: A dictionary of dictionaries.
        """
        return self.query_results_map

    def save_results(self) -> None:
        """
        Accessor function to write results to a default path

        :return: A dictionary of dictionaries.
        """
        with open('retrieved_results_map.pkl', 'wb') as f:
            pickle.dump(self.query_results_map, f)
        return self.query_results_map

    def retrieve_result_list(self) -> Dict:
        """
        Accessor function returning all results.

        :return: A dictionary of lists.
        """
        return self.query_results_list


class CosEval:
    """
    Calculates cosine similarity given matrices of query and retrieval features and returns valid retrieval files
    that exceed a similarity threshold for each query.

    Needs to be initialized with a query matrix and a retrieval matrix.
    """

    def __init__(self, query_vector: np.ndarray, ret_vector: np.ndarray) -> None:
        """
        Initializes local query and retrieval vectors and performs row-wise normalization of both the vectors.
        :param query_vector: A numpy array of shape (number_of_query_images, number_of_features)
        :param ret_vector: A numpy array of shape (number_of_retrieval_images, number_of_features)
        """

        self.query_vector = query_vector
        self.ret_vector = ret_vector
        self.logger = return_logger(__name__, os.getcwd())
        self._normalize_vector_matrices()
        self.sim_mat = None

    def _normalize_vector_matrices(self) -> None:
        """
        Perform row-wise normalization of both self.query_vector and self.ret_vector.
        """

        self.logger.info('Start: Vector normalization for computing cosine similarity')
        self.normed_query_vector = self.get_normalized_matrix(self.query_vector)
        self.normed_ret_vector = self.get_normalized_matrix(self.ret_vector)
        self.logger.info('Completed: Vector normalization for computing cosine similarity')

    @staticmethod
    def get_normalized_matrix(x: np.ndarray) -> np.ndarray:
        """
        Perform row-wise normalization of a given matrix.
        :param x: numpy ndarray that needs to be row normalized.
        :return: normalized ndarray.
        """

        x_norm_per_row = norm(x, axis=1)
        x_norm_per_row = x_norm_per_row[:, np.newaxis]  # adding another axis
        x_norm_per_row_tiled = np.tile(x_norm_per_row, (1, x.shape[1]))
        x_normalized = x / x_norm_per_row_tiled
        return x_normalized

    def _get_similarity(self) -> None:
        """
        Obtains a similarity matrix between self.query_vector and self.ret_vector.
        Populates the self.sim_mat variable.
        """

        self.logger.info('Start: Cosine similarity matrix computation')
        self.sim_mat = np.dot(self.normed_query_vector, self.normed_ret_vector.T)
        self.logger.info('End: Cosine similarity matrix computation')

    @staticmethod
    def _get_matches_above_threshold(row: np.ndarray, thresh: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given a single vector, return all indices and values of its elements that exceed or are equal to a value
        (threshold).
        :param row: numpy ndarray for which values need to be obtained.
        :param thresh: Value above which elements are considered valid to be returned.
        :return: valid_inds: numpy array, indices in the row where element exceed to is equal to the threshold value.
        :return: valid_vals: numpy array, values of elements that exceed or are equal to the threshold value.
        """

        valid_inds = np.where(row >= thresh)[0]
        valid_vals = row[valid_inds]
        return valid_inds, valid_vals

    def get_retrievals_at_thresh(self, file_mapping_query: Dict, file_mapping_ret: Dict, thresh=0.8) -> Dict:
        """
        Get valid retrievals for all queries given a similarity threshold.
        :param file_mapping_query: Dictionary mapping row number of query vector to filename.
        :param file_mapping_ret: Dictionary mapping row number of retrieval vector to filename.
        :param thresh: Cosine similarity above which retrieved duplicates are valid.
        :return: dict_ret: Dictionary of query and corresponding valid retrievals along with similarity scores.
        {'image1.jpg': {'image1_duplicate1.jpg':<similarity-score>,
        'image1_duplicate2.jpg':<similarity-score>, ..}, 'image2.jpg':{'image1_duplicate1.jpg':<similarity-score>,..}}
        """

        self.logger.info(f'Start: Getting duplicates with similarity above threshold = {thresh}')
        dict_ret = {}
        self._get_similarity()
        for i in range(self.sim_mat.shape[0]):
            valid_inds, valid_vals = self._get_matches_above_threshold(self.sim_mat[i, :], thresh)
            retrieved_files = [file_mapping_ret[j] for j in valid_inds]
            query_name = file_mapping_query[i]
            if query_name in retrieved_files:
                retrieved_files.remove(query_name)
            dict_ret[query_name] = dict(zip(retrieved_files, valid_vals))
        self.logger.info(f'End: Getting duplicates with similarity above threshold = {thresh}')
        return dict_ret
