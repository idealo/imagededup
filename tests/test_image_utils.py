from imagededup import image_utils
from pathlib import Path
import pytest


def test_check_valid_file_file_not_exists():
    path_file = Path('tests/data/bla.jpg')
    with pytest.raises(FileNotFoundError):
        image_utils.check_valid_file(path_file)


def test_check_valid_file_unsupported_format():
    path_file = Path('tests/data/ukbench09380.svg')
    with pytest.raises(TypeError):
        image_utils.check_valid_file(path_file)


def test_check_valid_file_correct_formats_loaded():
    path_file = Path('tests/data/base_images/ukbench00120.jpg')
    assert image_utils.check_valid_file(path_file) == 1
    path_file = Path('tests/data/ukbench09380.bmp')
    assert image_utils.check_valid_file(path_file) == 1
    path_file = Path('tests/data/ukbench09380.png')
    assert image_utils.check_valid_file(path_file) == 1
    path_file = Path('tests/data/ukbench09380.jpeg')
    assert image_utils.check_valid_file(path_file) == 1