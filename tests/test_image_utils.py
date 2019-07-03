from imagededup.utils import image_utils
from pathlib import Path
import pytest


def test_check_valid_file_file_not_exists():
    path_file = Path('tests/data/bla.jpg')
    with pytest.raises(FileNotFoundError):
        image_utils._validate_single_image(path_file)


def test_check_valid_file_unsupported_format():
    path_file = Path('tests/data/ukbench09380.svg')
    with pytest.raises(TypeError):
        image_utils._validate_single_image(path_file)


def test_check_valid_file_correct_formats_passes():
    path_file = Path('tests/data/base_images/ukbench00120.jpg')
    assert image_utils._validate_single_image(path_file) == 1
    path_file = Path('tests/data/formats_images/ukbench09380.bmp')
    assert image_utils._validate_single_image(path_file) == 1
    path_file = Path('tests/data/formats_images/ukbench09380.png')
    assert image_utils._validate_single_image(path_file) == 1
    path_file = Path('tests/data/formats_images/ukbench09380.jpeg')
    assert image_utils._validate_single_image(path_file) == 1


def test__load_image_valid_image_loads():
    path_file = Path('tests/data/base_images/ukbench00120.jpg')
    assert image_utils._load_image(path_file).format == 'JPEG'


def test_check_directory_files():
    path_dir = Path('tests/data/base_images')
    assert len(image_utils.check_directory_files(path_dir=path_dir, return_file=True)) == 10


def test_check_directory_files_no_return():
    path_dir = Path('tests/data/base_images')
    assert image_utils.check_directory_files(path_dir=path_dir, return_file=False) is None


def test_check_directory_files_invalid():
    path_dir = Path('tests/data/formats_images')
    with pytest.raises(Exception):
        image_utils.check_directory_files(path_dir=path_dir, return_file=False)
    with pytest.raises(Exception):
        image_utils.check_directory_files(path_dir=path_dir, return_file=True)