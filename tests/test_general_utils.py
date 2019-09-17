from imagededup.utils import general_utils

"""Run from project root with: python -m pytest -vs tests/test_general_utils.py"""


def test_get_files_to_remove():
    from collections import OrderedDict

    dict_a = OrderedDict({'1': ['2'], '2': ['1', '3'], '3': ['4'], '4': ['3'], '5': []})
    dups_to_remove = general_utils.get_files_to_remove(dict_a)
    assert set(dups_to_remove) == set(['2', '4'])
