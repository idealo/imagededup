from typing import Dict, List


def get_files_to_remove(dict_ret: Dict) -> List:
    # iterate over dict_ret keys, get value for the key and delete the dict keys that are in the value list
    list_of_files_to_remove = []

    for k, v in dict_ret.items():
        if k not in list_of_files_to_remove:
            list_of_files_to_remove.extend(v)
    return list(set(list_of_files_to_remove))  # set to remove duplicates
