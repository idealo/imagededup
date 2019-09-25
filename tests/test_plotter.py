import os
import pytest
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np

from imagededup.utils.plotter import _formatter, _validate_args, plot_duplicates

p = Path(__file__)

PATH_DIR_POSIX = p.parent / 'data/mixed_images'
PATH_DIR = os.path.join(os.getcwd(), 'tests/data/mixed_images')
PATH_DIR_INVALID = 'some_invalid_path/'


def test__formatter_float():
    assert _formatter(np.float32(0.81342537)) == '0.813'


def test__formatter_int():
    assert _formatter(3) == 3


def test__formatter_float_not_npfloat32():
    assert _formatter(0.81342537) == 0.81342537


# test_validate_args (n tcs)

def test__validate_args_nonposixpath():
    assert _validate_args(image_dir=PATH_DIR, duplicate_map={'1': ['2']}, filename='1') == PATH_DIR_POSIX


def test__validate_args_image_dir():
    with pytest.raises(AssertionError) as e:
        _validate_args(image_dir=PATH_DIR_INVALID, duplicate_map=None, filename=None)
    assert (
        str(e.value)
        == 'Provided image directory does not exist! Please provide the image directory where all files are present!'
    )


def test__validate_args_duplicate_map():
    with pytest.raises(ValueError) as e:
        _validate_args(image_dir=PATH_DIR, duplicate_map=None, filename=None)
    assert str(e.value) == 'Please provide a valid Duplicate map!'


def test__validate_args_filename():
    with pytest.raises(ValueError) as e:
        _validate_args(image_dir=PATH_DIR, duplicate_map={'1': ['2']}, filename='2')
    assert (
        str(e.value)
        == 'Please provide a valid filename present as a key in the duplicate_map!'
    )


# test plot_duplicates, assert calls
@pytest.fixture
def mocker_validate_args(mocker):
    return mocker.patch('imagededup.utils.plotter._validate_args', return_value=PATH_DIR_POSIX)


@pytest.fixture
def mocker_plot_images(mocker):
    return mocker.patch('imagededup.utils.plotter._plot_images')


def test_plot_duplicates(mocker_validate_args, mocker_plot_images):
    plot_duplicates(image_dir=PATH_DIR_POSIX, duplicate_map={'1': ['2']}, filename='1')
    mocker_validate_args.assert_called_once_with(
        image_dir=PATH_DIR_POSIX, duplicate_map={'1': ['2']}, filename='1'
    )
    mocker_plot_images.assert_called_once_with(
        image_dir=PATH_DIR_POSIX, orig='1', image_list=['2'], scores=False, outfile=None
    )


def test_plot_duplicates_outfile(mocker_validate_args, mocker_plot_images):
    plot_duplicates(
        image_dir=PATH_DIR_POSIX,
        duplicate_map={'1': ['2']},
        filename='1',
        outfile='bla.png',
    )
    mocker_validate_args.assert_called_once_with(
        image_dir=PATH_DIR_POSIX, duplicate_map={'1': ['2']}, filename='1'
    )
    mocker_plot_images.assert_called_once_with(
        image_dir=PATH_DIR_POSIX,
        orig='1',
        image_list=['2'],
        scores=False,
        outfile='bla.png',
    )


def test_plot_duplicates_scores(mocker_validate_args, mocker_plot_images):
    plot_duplicates(
        image_dir=PATH_DIR_POSIX, duplicate_map={'1': [('2', 0.6)]}, filename='1'
    )
    mocker_validate_args.assert_called_once_with(
        image_dir=PATH_DIR_POSIX, duplicate_map={'1': [('2', 0.6)]}, filename='1'
    )
    mocker_plot_images.assert_called_once_with(
        image_dir=PATH_DIR_POSIX,
        orig='1',
        image_list=[('2', 0.6)],
        scores=True,
        outfile=None,
    )


def test_plot_duplicates_no_duplicates():
    with pytest.raises(AssertionError) as e:
        plot_duplicates(
            image_dir=PATH_DIR_POSIX, duplicate_map={'1': [], '2': []}, filename='2'
        )
    assert str(e.value) == 'Provided filename has no duplicates!'


# def test_plot_duplicates_integrated():
#     plot_duplicates(image_dir=PATH_DIR_POSIX, duplicate_map={'ukbench00120.jpg': ['ukbench00120_hflip.jpg']}, filename='ukbench00120.jpg')
#     plt.gcf().canvas.draw()
