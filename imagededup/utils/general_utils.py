import os
import numpy as np
from types import FunctionType
from typing import Tuple
from multiprocessing import Manager, Process, cpu_count, Pool
import tqdm


import json
from typing import Dict, List

from imagededup.utils.logger import return_logger

import multiprocessing
import tqdm


logger = return_logger(__name__, os.getcwd())


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
    logger.info('Start: Saving duplicates as json!')
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)
    logger.info('End: Saving duplicates as json!')


# def parallelize(target_function: FunctionType, all_tasks: List, args_target_function: Tuple = ()):
#     manager = Manager()  # used to share the global variable (result dictionary) across different processes
#     result = manager.dict()  # Only works for dictionary returns
#     chunks = [i for i in np.array_split(all_tasks, cpu_count())]  # Each process gets a
#     # sublist of tasks
#
#     args_in = (result, *args_target_function)
#     job = [Process(target=target_function, args=(*args_in, sublist)) for sublist in chunks]
#     _ = [p.start() for p in job]
#     _ = [p.join() for p in job]
#     return result


def parallelise(function, data):
    vcore_count = cpu_count()
    chunksize = len(data) // vcore_count

    if chunksize == 0:
        n_proc = 1
        chunksize = 1
    else:
        n_proc = vcore_count

    pool = Pool(processes=n_proc)
    results = list(tqdm.tqdm(pool.imap(function, data, 1), total=len(data)))
    pool.close()
    pool.join()
    return results
