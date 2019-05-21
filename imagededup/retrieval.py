
from imagededup.hashing import Hashing 

class ResultSet:
    def __init__(self, test:dict, queries: dict) -> None:
        self.candidates = test
        self.queries = queries
        self.fetch_nearest_neighbors()


    def fetch_query_result(self, query) -> dict:
        hammer = Hashing().hamming_distance
        return {item: hammer(query, item) for item in self.candidates if hammer(query, item) < 5}


    def fetch_nearest_neighbors(self) -> None:
        sorted_results = {}
        for each in self.queries:
            res = self.fetch_query_result(each)
            sorted_results[each] = sorted(res, key=lambda x: res[x], reverse=False)
        self.query_results = sorted_results

            
    def retrieve_results(self) -> dict:
        return self.query_results