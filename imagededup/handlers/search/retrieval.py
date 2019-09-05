from imagededup.handlers.search.bktree import BKTree
from imagededup.handlers.search.brute_force import BruteForce
from imagededup.utils.logger import return_logger
from types import FunctionType
from numpy.linalg import norm
from typing import Tuple, Dict, List, Union
import os
import numpy as np


class HashEval:
    def __init__(self, test: Dict, queries: Dict, hammer: FunctionType, cutoff: int = 5, search_method: str = 'bktree')\
            -> None:
        """
        Initializes a HashEval object which offers an interface to control hashing and search methods for desired
        dataset. Computes a map of duplicate images in the document space given certain input control parameters.
        """
        self.candidates = test  # database
        self.queries = queries
        self.hamming_distance_invoker = hammer
        self.max_d = cutoff
        self.logger = return_logger(__name__, os.getcwd())
        self.query_results_map = None
        self.query_results_list = None

        if search_method == 'bktree':
            self.fetch_nearest_neighbors_bktree()  # bktree is the default search method
        else:
            self.fetch_nearest_neighbors_brute_force()

    def _get_query_results(self, search_method_object: Union[BruteForce, BKTree]) -> None:
        """
        Gets result for the query using specified search object. Populates the global query_results_map and
        query_results_list attributes.
        :param search_method_object: BruteForce or BKTree object to get results for the query.
        """
        sorted_result_list, result_map = {}, {}
        for each in self.queries:
            res = search_method_object.search(query=self.queries[each], tol=self.max_d)  # list of tuples
            res = [i for i in res if i[0] != each]  # to avoid self retrieval
            result_map[each] = res
        self.query_results_map = result_map  # {'filename.jpg': [('dup1.jpg', 3)], 'filename2.jpg': [('dup2.jpg', 10)]}

    def fetch_nearest_neighbors_brute_force(self) -> None:
        """
        Wrapper function to retrieve results for all queries in dataset using brute-force search.
        """
        self.logger.info('Start: Retrieving duplicates using Brute force algorithm')
        bruteforce = BruteForce(self.candidates, self.hamming_distance_invoker)
        self._get_query_results(bruteforce)
        self.logger.info('End: Retrieving duplicates using Brute force algorithm')

    def fetch_nearest_neighbors_bktree(self) -> None:
        """
        Wrapper function to retrieve results for all queries in dataset using a BKTree search.
        """
        self.logger.info('Start: Retrieving duplicates using BKTree algorithm')
        built_tree = BKTree(self.candidates, self.hamming_distance_invoker)  # construct bktree
        self._get_query_results(built_tree)
        self.logger.info('End: Retrieving duplicates using BKTree algorithm')

    def retrieve_results(self, scores: bool = False) -> Dict:
        if scores:
            return self.query_results_map
        else:
            return {k: [i[0] for i in v] for k, v in self.query_results_map.items()}

