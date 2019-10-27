import os
import json

import numpy as np

from imagededup.utils import general_utils


def test_get_files_to_remove():
    from collections import OrderedDict

    dict_a = OrderedDict({'1': ['2'], '2': ['1', '3'], '3': ['4'], '4': ['3'], '5': []})
    dups_to_remove = general_utils.get_files_to_remove(dict_a)
    assert set(dups_to_remove) == set(['2', '4'])


def test_correct_saving_floats():
    res = {
        'image1.jpg': [
            ('image1_duplicate1.jpg', np.float16(0.324)),
            ('image1_duplicate2.jpg', np.float16(0.324)),
        ],
        'image2.jpg': [],
        'image3.jpg': [('image1_duplicate1.jpg', np.float32(0.324))],
    }
    save_file = 'myduplicates.json'
    general_utils.save_json(results=res, filename=save_file, float_scores=True)
    with open(save_file, 'r') as f:
        saved_json = json.load(f)

    assert len(saved_json) == 3  # all valid files present as keys
    assert isinstance(
        saved_json['image1.jpg'][0][1], float
    )  # saved score is of type 'float' for np.float16 score
    assert isinstance(
        saved_json['image3.jpg'][0][1], float
    )  # saved score is of type 'float' for np.float32 score

    os.remove(save_file)  # clean up


def test_correct_saving_ints():
    res = {
        'image1.jpg': [('image1_duplicate1.jpg', 2), ('image1_duplicate2.jpg', 22)],
        'image2.jpg': [],
        'image3.jpg': [('image1_duplicate1.jpg', 43)],
    }
    save_file = 'myduplicates.json'
    general_utils.save_json(results=res, filename=save_file)
    with open(save_file, 'r') as f:
        saved_json = json.load(f)

    assert len(saved_json) == 3  # all valid files present as keys
    assert isinstance(
        saved_json['image1.jpg'][0][1], int
    )  # saved score is of type 'int'

    os.remove(save_file)  # clean up
