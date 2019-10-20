from typing import Callable, Dict


class BruteForce:
    """
    Class to perform search using a Brute force.
    """

    def __init__(self, hash_dict: Dict, distance_function: Callable) -> None:
        """
        Initialize a dictionary for mapping file names and corresponding hashes and a distance function to be used for
        getting distance between two hash strings.

        Args:
            hash_dict: Dictionary mapping file names to corresponding hash strings {filename: hash}
            distance_function:  A function for calculating distance between the hashes.
        """
        self.distance_function = distance_function
        self.hash_dict = hash_dict  # database

    def search(self, query: str, tol: int = 10) -> Dict[str, int]:
        """
        Function for searching using brute force.

        Args:
            query: hash string for which brute force needs to work.
            tol: distance upto which duplicate is valid.

        Returns:
            List of tuples of the form [(valid_retrieval_filename1: distance), (valid_retrieval_filename2: distance)]
        """
        return [
            (item, self.distance_function(query, self.hash_dict[item]))
            for item in self.hash_dict
            if self.distance_function(query, self.hash_dict[item]) <= tol
        ]
