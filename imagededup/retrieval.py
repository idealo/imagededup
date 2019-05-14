import shelve

class ResultSet:
    """In order to retrieve duplicate images an index needs to be built against which
    search operations are run. The ResultSe Class serves as a search and retrieval
    interface, essential for driving interfacing for downstream tasks.

    Takes input dictionary of image hashes for which DB has to be created."""
    def __init__(self, index_save_path: str, candidates:dict, queries: dict) -> None:
        self.db = self.create_db_index(index_save_path)
        self.populate_db(candidates)
        self.query_results = self.find_nearest_neighbors(queries)
    
    
    @staticmethod
    def create_db_index(path) -> shelve.DbfilenameShelf:
        return shelve.open(path, writeback = True)


    def populate_db(self, candidates: dict):
        for each in candidates:
            self.db[candidates[each]] = self.db.get(candidates[each], []) + [each]
        # Close the shelf database
        self.db.close()


    def find_nearest_neighbors(self, queries):
        return {query: self.db[queries[query]][:5] for query in queries}


    def retrieve_results(self):
        return self.query_results