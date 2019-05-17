import shelve
import os

class ResultSet:
    """In order to retrieve duplicate images an index needs to be built against which
    search operations are run. The ResultSe Class serves as a search and retrieval
    interface, essential for driving interfacing for downstream tasks.

    Takes input dictionary of image hashes for which DB has to be created."""
    def __init__(self, index_save_path: str, candidates:dict, queries: dict) -> None:
        self.db_path = f'{index_save_path}.db'
        self.db = self.create_db_index(index_save_path)
        self.populate_db(candidates)
        self.fetch_nearest_neighbors(queries)
        self.destroy_db_index()

    @staticmethod
    def create_db_index(path) -> shelve.DbfilenameShelf:
        return shelve.open(path, writeback=True)


    def refresh_db_buffer(self) -> shelve.DbfilenameShelf:
        return shelve.open(self.db_path)


    def populate_db(self, candidates: dict):
        for each in candidates:
            self.db[candidates[each]] = self.db.get(candidates[each], []) + [each]
        # Close the shelf database
        self.db.close()


    def fetch_nearest_neighbors(self, queries) -> None:
        self.db = self.refresh_db_buffer()
        self.query_results = {query: self.db[queries[query]] for query in queries}
        self.db.close()

        
    def destroy_db_index(self) -> None:
        if self.query_results and os.path.exists(self.db_path):
            os.remove(self.db_path)
            
    def retrieve_results(self):
        return self.query_results