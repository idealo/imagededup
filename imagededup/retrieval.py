from types import FunctionType


class ResultSet:
    def __init__(self, test: dict, queries: dict, hammer: FunctionType) -> None:
        self.candidates = test
        self.queries = queries
        self.hamming_distance_invoker = hammer
        self.fetch_nearest_neighbors()

    def fetch_query_result(self, query) -> dict:
        hammer = self.hamming_distance_invoker
        candidates = self.candidates
        return {item: hammer(query, candidates[item]) for item in candidates if hammer(query, candidates[item]) < 5}

    def fetch_nearest_neighbors(self) -> None:
        sorted_results, sorted_distances = {}, {}
        for each in self.queries.values():
            res = self.fetch_query_result(each)
            sorted_results[each] = sorted(res, key=lambda x: res[x], reverse=False)
            sorted_distances[each] = res.values()
        self.query_results = sorted_results
        self.query_distances = sorted_distances

    def retrieve_results(self) -> dict:
        return self.query_results

    def retrieve_distances(self) -> dict:
        return self.query_distances
