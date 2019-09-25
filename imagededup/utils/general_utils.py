import json
import tqdm
from multiprocessing import cpu_count, Pool
from typing import Callable, Dict, List


def get_files_to_remove(duplicates: Dict[str, List]) -> List:
    """
    Get a list of files to remove.
    
    Args:
        duplicates: A dictionary with file name as key and a list of duplicate file names as value.

    Returns:
        A list of files that should be removed.
    """
    # iterate over dict_ret keys, get value for the key and delete the dict keys that are in the value list
    files_to_remove = set()

    for k, v in duplicates.items():
        tmp = [
            i[0] if isinstance(i, tuple) else i for i in v
        ]  # handle tuples (image_id, score)

        if k not in files_to_remove:
            files_to_remove.update(tmp)

    return list(files_to_remove)


def save_json(results: Dict, filename: str) -> None:
    """
    Save results with a filename.

    Args:
        results: Dictionary of results to be saved.
        filename: Name of the file to be saved.
    """
    print('Start: Saving duplicates as json!')
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print('End: Saving duplicates as json!')


def parallelise(function: Callable, data: List) -> List:
    pool = Pool(processes=cpu_count())
    results = list(tqdm.tqdm(pool.imap(function, data), total=len(data)))
    pool.close()
    pool.join()
    return results
