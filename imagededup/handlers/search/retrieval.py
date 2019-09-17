import os
from typing import Dict, Union
from types import FunctionType

from imagededup.utils.logger import return_logger
from imagededup.handlers.search.bktree import BKTree
from imagededup.handlers.search.brute_force import BruteForce

from imagededup.utils.logger import return_logger
from types import FunctionType
from numpy.linalg import norm
from typing import Tuple, Dict, List, Union
import os
import numpy as np
from imagededup.utils.general_utils import parallelise


class HashEval:
    def __init__(
        self,
        test: Dict,
        queries: Dict,
        distance_function: FunctionType,
        threshold: int = 5,
        search_method: str = 'bktree',
    ) -> None:
        """
        Initialize a HashEval object which offers an interface to control hashing and search methods for desired
        dataset. Compute a map of duplicate images in the document space given certain input control parameters.
        """
        self.test = test  # database
        self.queries = queries
        self.distance_invoker = distance_function
        self.threshold = threshold
        self.logger = return_logger(__name__, os.getcwd())
        self.query_results_map = None
        self.query_results_list = None

        if search_method == 'bktree':
            self._fetch_nearest_neighbors_bktree()  # bktree is the default search method
        else:
            self._fetch_nearest_neighbors_brute_force()

    def _searcher(self, query_list: List) -> None:
        """
        Perform image encoding on a sublist passed in by encode_images multiprocessing part.
        Args:
            hash_dict: Global dictionary that gets shared by all processes
            filenames: Sublist of file names on which hashes are to be generated.
        """
        result_map = {}

        # hashes = parallelise(self._searcher, files)
        # hash_initial_dict = dict(zip([f.name for f in files], hashes))
        # hash_dict = {k: v for k, v in hash_initial_dict.items() if v}

        for each in self.queries:
            res = search_method_object.search(
                query=self.queries[each], tol=self.threshold
            )  # list of tuples
            res = [i for i in res if i[0] != each]  # to avoid self retrieval
            result_map[each] = res

        # result_map = parallelise()

    def _get_query_results(
        self, search_method_object: Union[BruteForce, BKTree]
    ) -> None:
        """
        Get result for the query using specified search object. Populate the global query_results_map.

        Args:
            search_method_object: BruteForce or BKTree object to get results for the query.
        """
        result_map = {}

        # hashes = parallelise(self._searcher, files)
        # hash_initial_dict = dict(zip([f.name for f in files], hashes))
        # hash_dict = {k: v for k, v in hash_initial_dict.items() if v}

        for each in self.queries:
            res = search_method_object.search(
                query=self.queries[each], tol=self.threshold
            )  # list of tuples
            res = [i for i in res if i[0] != each]  # to avoid self retrieval
            result_map[each] = res

        # result_map = parallelise()
        self.query_results_map = {
            k: [i for i in sorted(v, key=lambda tup: tup[1], reverse=False)]
            for k, v in result_map.items()
        }  # {'filename.jpg': [('dup1.jpg', 3)], 'filename2.jpg': [('dup2.jpg', 10)]}

    def _fetch_nearest_neighbors_brute_force(self) -> None:
        """
        Wrapper function to retrieve results for all queries in dataset using brute-force search.
        """
        self.logger.info('Start: Retrieving duplicates using Brute force algorithm')
        bruteforce = BruteForce(self.test, self.distance_invoker)
        self._get_query_results(bruteforce)
        self.logger.info('End: Retrieving duplicates using Brute force algorithm')

    def _fetch_nearest_neighbors_bktree(self) -> None:
        """
        Wrapper function to retrieve results for all queries in dataset using a BKTree search.
        """
        self.logger.info('Start: Retrieving duplicates using BKTree algorithm')
        built_tree = BKTree(self.test, self.distance_invoker)  # construct bktree
        self._get_query_results(built_tree)
        self.logger.info('End: Retrieving duplicates using BKTree algorithm')

    def retrieve_results(self, scores: bool = False) -> Dict:
        """
        Return results with or without scores.

        Args:
            scores: Boolean indicating whether results are to eb returned with or without scores.

        Returns:
            if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
            score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}
            if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg',
            'image1_duplicate2.jpg'], 'image2.jpg':['image1_duplicate1.jpg',..], ..}
        """
        if scores:
            return self.query_results_map
        else:
            return {k: [i[0] for i in v] for k, v in self.query_results_map.items()}
