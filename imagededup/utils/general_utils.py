from imagededup.utils.logger import return_logger
from typing import Dict, List
import json
import os

logger = return_logger(__name__, os.getcwd())


def get_files_to_remove(dict_ret: Dict[str, List]) -> List:
    """
    Get a list of files to remove
    :param dict_ret: A dcitionary with fie name as key and a list of duplicate file names as value.
    :return: A list of files that should be removed.
    """
    # iterate over dict_ret keys, get value for the key and delete the dict keys that are in the value list
    list_of_files_to_remove = []

    for k, v in dict_ret.items():
        if k not in list_of_files_to_remove:
            list_of_files_to_remove.extend(v)
    return list(set(list_of_files_to_remove))  # set to remove duplicates


def save_json(results: Dict, filename: str) -> None:
    """
    Save results.
    :param results: Dictionary of results to be saved.
    :param filename: Name of the file to be saved.
    """
    logger.info('Start: Saving duplicates as json!')
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)
    logger.info('End: Saving duplicates as json!')
