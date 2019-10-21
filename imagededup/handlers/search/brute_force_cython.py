from typing import Callable, Dict

import brute_force_cython_ext


class BruteForceCython:
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

        brute_force_cython_ext.clear()

        for filename, hash_val in self.hash_dict.items():
            brute_force_cython_ext.add(
                int(hash_val, 16), filename.encode('utf-8')
            )  # cast hex hash_val to decimals for __builtin_popcountll function

    def search(self, query: str, tol: int = 10) -> Dict[str, int]:
        """
        Function for searching using brute force.

        Args:
            query: hash string for which brute force needs to work.
            tol: distance upto which duplicate is valid.

        Returns:
            List of tuples of the form [(valid_retrieval_filename1: distance), (valid_retrieval_filename2: distance)]
        """

        return brute_force_cython_ext.query(
            int(query, 16), tol
        )  # cast hex hash_val to decimals for __builtin_popcountll function
