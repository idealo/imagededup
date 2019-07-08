from typing import Dict


class BruteForce:
    def __init__(self, hash_dict: Dict, distance_function) -> None:
        self.distance_function = distance_function
        self.hash_dict = hash_dict  # database

    def search(self, query: str, tol: int = 10) -> Dict:
        return {item: self.distance_function(query, self.hash_dict[item]) for item in self.hash_dict if
                self.distance_function(query, self.hash_dict[item]) <= tol}
