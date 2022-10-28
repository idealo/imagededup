import json
from multiprocessing import Pool
from pathlib import Path, PurePath
from typing import Callable, Dict, List, Union

import tqdm

from imagededup.utils.logger import return_logger

logger = return_logger(__name__)


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


def save_json(results: Dict, filename: str, float_scores: bool = False) -> None:
    """
    Save results with a filename.

    Args:
        results: Dictionary of results to be saved.
        filename: Name of the file to be saved.
        float_scores: boolean to indicate if scores are floats.
    """
    logger.info('Start: Saving duplicates as json!')

    if float_scores:
        for _file, dup_list in results.items():
            if dup_list:
                typecasted_dup_list = []
                for dup in dup_list:
                    typecasted_dup_list.append((dup[0], float(dup[1])))

                results[_file] = typecasted_dup_list

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)

    logger.info('End: Saving duplicates as json!')


def parallelise(function: Callable, data: List, verbose: bool, num_workers: int) -> List:
    num_workers = 1 if num_workers < 1 else num_workers  # Pool needs to have at least 1 worker.
    pool = Pool(processes=num_workers)
    results = list(
        tqdm.tqdm(pool.imap(function, data, 100), total=len(data), disable=not verbose)
    )
    pool.close()
    pool.join()
    return results


def generate_files(image_dir: Union[PurePath, str], recursive: bool) -> List:
    if recursive:
        glob_pattern = '**/*'
    else:
        glob_pattern = '*'

    return [
        i.absolute()
        for i in Path(image_dir).glob(glob_pattern)
        if not (i.name.startswith('.') or i.is_dir())
    ]


def generate_relative_names(image_dir: Union[PurePath, str], files: List) -> List:
    return [str(f.relative_to(Path(image_dir).absolute())) for f in files]
