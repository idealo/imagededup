import sys
from typing import Callable, Dict, Union, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from imagededup.handlers.search.bktree import BKTree
from imagededup.handlers.search.brute_force import BruteForce
from imagededup.handlers.search.brute_force_cython import BruteForceCython
from imagededup.utils.general_utils import parallelise
from imagededup.utils.logger import return_logger

logger = return_logger(__name__)


def cosine_similarity_chunk(t: Tuple) -> np.ndarray:
    return cosine_similarity(t[0][t[1][0] : t[1][1]], t[0]).astype('float16')


def get_cosine_similarity(
    X: np.ndarray, verbose: bool = True, chunk_size: int = 1000, threshold: int = 10000
) -> np.ndarray:
    n_rows = X.shape[0]

    if n_rows <= threshold:
        return cosine_similarity(X)

    else:
        logger.info(
            'Large feature matrix thus calculating cosine similarities in chunks...'
        )
        start_idxs = list(range(0, n_rows, chunk_size))
        end_idxs = start_idxs[1:] + [n_rows]
        cos_sim = parallelise(
            cosine_similarity_chunk,
            [(X, idxs) for i, idxs in enumerate(zip(start_idxs, end_idxs))],
            verbose,
        )

        return np.vstack(cos_sim)


class HashEval:
    def __init__(
        self,
        test: Dict,
        queries: Dict,
        distance_function: Callable,
        verbose: bool = True,
        threshold: int = 5,
        search_method: str = 'brute_force_cython' if not sys.platform == 'win32' else 'bktree',
    ) -> None:
        """
        Initialize a HashEval object which offers an interface to control hashing and search methods for desired
        dataset. Compute a map of duplicate images in the document space given certain input control parameters.
        """
        self.test = test  # database
        self.queries = queries
        self.distance_invoker = distance_function
        self.verbose = verbose
        self.threshold = threshold
        self.query_results_map = None

        if search_method == 'bktree':
            self._fetch_nearest_neighbors_bktree()
        elif search_method == 'brute_force':
            self._fetch_nearest_neighbors_brute_force()
        else:
            self._fetch_nearest_neighbors_brute_force_cython()

    def _searcher(self, data_tuple) -> None:
        """
        Perform search on a query passed in by _get_query_results multiprocessing part.

        Args:
            data_tuple: Tuple of (query_key, query_val, search_method_object, thresh)

        Returns:
           List of retrieved duplicate files and corresponding hamming distance for the query file.
        """
        query_key, query_val, search_method_object, thresh = data_tuple
        res = search_method_object.search(query=query_val, tol=thresh)
        res = [i for i in res if i[0] != query_key]  # to avoid self retrieval
        return res

    def _get_query_results(
        self, search_method_object: Union[BruteForce, BKTree]
    ) -> None:
        """
        Get result for the query using specified search object. Populate the global query_results_map.

        Args:
            search_method_object: BruteForce or BKTree object to get results for the query.
        """
        args = list(
            zip(
                list(self.queries.keys()),
                list(self.queries.values()),
                [search_method_object] * len(self.queries),
                [self.threshold] * len(self.queries),
            )
        )
        result_map_list = parallelise(self._searcher, args, self.verbose)
        result_map = dict(zip(list(self.queries.keys()), result_map_list))

        self.query_results_map = {
            k: [i for i in sorted(v, key=lambda tup: tup[1], reverse=False)]
            for k, v in result_map.items()
        }  # {'filename.jpg': [('dup1.jpg', 3)], 'filename2.jpg': [('dup2.jpg', 10)]}

    def _fetch_nearest_neighbors_brute_force(self) -> None:
        """
        Wrapper function to retrieve results for all queries in dataset using brute-force search.
        """
        logger.info('Start: Retrieving duplicates using Brute force algorithm')
        brute_force = BruteForce(self.test, self.distance_invoker)
        self._get_query_results(brute_force)
        logger.info('End: Retrieving duplicates using Brute force algorithm')

    def _fetch_nearest_neighbors_brute_force_cython(self) -> None:
        """
        Wrapper function to retrieve results for all queries in dataset using brute-force search.
        """
        logger.info('Start: Retrieving duplicates using Cython Brute force algorithm')
        brute_force_cython = BruteForceCython(self.test, self.distance_invoker)
        self._get_query_results(brute_force_cython)
        logger.info('End: Retrieving duplicates using Cython Brute force algorithm')

    def _fetch_nearest_neighbors_bktree(self) -> None:
        """
        Wrapper function to retrieve results for all queries in dataset using a BKTree search.
        """
        logger.info('Start: Retrieving duplicates using BKTree algorithm')
        built_tree = BKTree(self.test, self.distance_invoker)  # construct bktree
        self._get_query_results(built_tree)
        logger.info('End: Retrieving duplicates using BKTree algorithm')

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
