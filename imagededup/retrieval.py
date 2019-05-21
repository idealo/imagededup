from hashing import Hashing

class ResultSet:
    def __init__(self, test:dict, queries: dict) -> None:
        # self.db_path = f'{index_save_path}.db'
        # self.db = self.create_db_index(index_save_path)
        self.candidates = test
        self.queries = queries
        self.fetch_nearest_neighbors()
        #self.destroy_db_index()

#     @staticmethod
#     def create_db_index(path) -> shelve.DbfilenameShelf:
#         return shelve.open(path, writeback=True)
   

    @staticmethod
    def build_query_vector(query) -> list:
        return [query] * self.n_candidates


#     def refresh_db_buffer(self) -> shelve.DbfilenameShelf:
#         return shelve.open(self.db_path)

    def fetch_query_result(self, query: str) -> dict:
        return {item: Hashing().hamming_distance(query, item) for item in self.candidates}


#     def populate_db(self, candidates: dict):
#         for each in candidates:
#             self.db[candidates[each]] = self.db.get(candidates[each], []) + [each]
#         # Close the shelf database
#         self.db.close()


    def fetch_nearest_neighbors(self) -> None:
        results = []
        for each in self.queries:
            res = self.fetch_query_result(each)
            sorted_res = {each: sorted(res, key=lambda x: res[x], reverse=False)}
            results.append(sorted_res)
        self.query_results = results

        
#     def destroy_db_index(self) -> None:
#         if self.query_results and os.path.exists(self.db_path):
#             os.remove(self.db_path)
            
    def retrieve_results(self) -> dict:
        return self.query_results