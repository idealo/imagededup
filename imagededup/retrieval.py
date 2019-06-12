from imagededup.bktree import BKTree
from types import FunctionType

"""
TODO: Choose whether to run brute force or bktree search.
"""


class ResultSet:
    def __init__(self, test: dict, queries: dict, hammer: FunctionType) -> None:
        self.candidates = test
        self.queries = queries
        self.hamming_distance_invoker = hammer
        # self.fetch_nearest_neighbors_brute_force()
        self.fetch_nearest_neighbors_bktree() # Keep bktree as the default search method instead of brute force

    def fetch_query_result_brute_force(self, query) -> dict:
        hammer = self.hamming_distance_invoker
        candidates = self.candidates
        return {item: hammer(query, candidates[item]) for item in candidates if hammer(query, candidates[item]) < 5}

    def fetch_nearest_neighbors_brute_force(self) -> None:
        sorted_results, sorted_distances = {}, {}
        for each in self.queries.values():
            res = self.fetch_query_result_brute_force(each)
            sorted_results[each] = sorted(res, key=lambda x: res[x], reverse=False)
            sorted_distances[each] = res.values() # REQUEST: Sort values too
        self.query_results = sorted_results # REQUEST: Have key as filenames and not hashes
        self.query_distances = sorted_distances # REQUEST: Change return types from dict_values to list, also have key as filenames and not hashes

    def fetch_nearest_neighbors_bktree(self) -> None:
        dist_func = self.hamming_distance_invoker
        built_tree = BKTree(self.candidates, dist_func)  # construct bktree

        sorted_results, sorted_distances = {}, {}
        for each in self.queries.values():
            res = built_tree.search(each)
            sorted_results[each] = sorted(res, key=lambda x: res[x], reverse=False)
            sorted_distances[each] = sorted(res.values())
        self.query_results = sorted_results
        self.query_distances = sorted_distances

    def retrieve_results(self) -> dict:
        return self.query_results
    
    def retrieve_distances(self) -> dict:
        return self.query_distances
