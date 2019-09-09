import os
from typing import Dict, Union
from types import FunctionType

from imagededup.utils.logger import return_logger
from imagededup.handlers.search.bktree import BKTree
from imagededup.handlers.search.brute_force import BruteForce


class HashEval:
    def __init__(
        self,
        test: Dict,
        queries: Dict,
        distance_function: FunctionType,
        threshold: int = 5,
        search_method: str = "bktree",
    ) -> None:
        """
        Initializes a HashEval object which offers an interface to control hashing and search methods for desired
        dataset. Computes a map of duplicate images in the document space given certain input control parameters.
        """
        self.test = test  # database
        self.queries = queries
        self.distance_invoker = distance_function
        self.threshold = threshold
        self.logger = return_logger(__name__, os.getcwd())
        self.query_results_map = None
        self.query_results_list = None

        if search_method == "bktree":
            self._fetch_nearest_neighbors_bktree()  # bktree is the default search method
        else:
            self._fetch_nearest_neighbors_brute_force()

    def _get_query_results(
        self, search_method_object: Union[BruteForce, BKTree]
    ) -> None:
        """
        Gets result for the query using specified search object. Populates the global query_results_map and
        query_results_list attributes.
        :param search_method_object: BruteForce or BKTree object to get results for the query.
        """
        result_map = {}

        for each in self.queries:
            res = search_method_object.search(
                query=self.queries[each], tol=self.threshold
            )  # list of tuples
            res = [i for i in res if i[0] != each]  # to avoid self retrieval
            result_map[each] = res

        self.query_results_map = {
            k: [i for i in sorted(v, key=lambda tup: tup[1], reverse=False)]
            for k, v in result_map.items()
        }  # {'filename.jpg': [('dup1.jpg', 3)], 'filename2.jpg': [('dup2.jpg', 10)]}

    def _fetch_nearest_neighbors_brute_force(self) -> None:
        """
        Wrapper function to retrieve results for all queries in dataset using brute-force search.
        """
        self.logger.info("Start: Retrieving duplicates using Brute force algorithm")
        bruteforce = BruteForce(self.test, self.distance_invoker)
        self._get_query_results(bruteforce)
        self.logger.info("End: Retrieving duplicates using Brute force algorithm")

    def _fetch_nearest_neighbors_bktree(self) -> None:
        """
        Wrapper function to retrieve results for all queries in dataset using a BKTree search.
        """
        self.logger.info("Start: Retrieving duplicates using BKTree algorithm")
        built_tree = BKTree(self.test, self.distance_invoker)  # construct bktree
        self._get_query_results(built_tree)
        self.logger.info("End: Retrieving duplicates using BKTree algorithm")

    def retrieve_results(self, scores: bool = False) -> Dict:
        if scores:
            return self.query_results_map
        else:
            return {k: [i[0] for i in v] for k, v in self.query_results_map.items()}
