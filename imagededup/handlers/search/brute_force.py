from typing import Dict
from types import FunctionType


class BruteForce:
    """
    Class to perform search using a Brute force.
    """
    def __init__(self, hash_dict: Dict, distance_function: FunctionType) -> None:
        """
        Initialises a dictionary for mapping file names and corresponding hashes anda  distance function to be used for
        getting distance between two hash strings.
        :param hash_dict: Dictionary mapping file names to corresponding hash strings {filename: hash}
        :param distance_function:  A function for calculating distance between the hashes.
        """
        self.distance_function = distance_function
        self.hash_dict = hash_dict  # database

    def search(self, query: str, tol: int = 10) -> Dict[str, int]:
        """
        Function for searching using brute force.
        :param query: hash string for which brute force needs to work.
        :param tol: distance upto which duplicate is valid.
        :return: Dictionary of retrieved file names and corresponding distances {valid_retrieval_filename: distance, ..}
        """
        return [(item, self.distance_function(query, self.hash_dict[item])) for item in self.hash_dict if
                self.distance_function(query, self.hash_dict[item]) <= tol]

