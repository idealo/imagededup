from typing import Dict, List


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
