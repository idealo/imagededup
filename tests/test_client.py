import os

from click.testing import CliRunner
from imagededup.client.client import find_duplicates

PATH_IMAGE_DIR = 'tests/data/mixed_images'
FILENAME = 'tests/test_output.json'


def test_no_image_dir_given():
    runner = CliRunner()
    result = runner.invoke(find_duplicates, ['--image_dir', ''])
    assert result.exit_code == 2


def test_image_dir_given_but_no_method():
    runner = CliRunner()
    result = runner.invoke(find_duplicates, ['--image_dir', PATH_IMAGE_DIR])
    assert result.exit_code == 2


def test_image_dir_given_and_method():
    runner = CliRunner()
    result = runner.invoke(find_duplicates, ['--image_dir', PATH_IMAGE_DIR, '--method', 'PHash'])
    assert result.exit_code == 0


def test_image_dir_given_but_wrong_method():
    runner = CliRunner()
    result = runner.invoke(find_duplicates, ['--image_dir', PATH_IMAGE_DIR, '--method', 'LHash'])
    assert result.exit_code == 2


def test_file_is_created():
    runner = CliRunner()
    result = runner.invoke(find_duplicates,
                           ['--image_dir', PATH_IMAGE_DIR, '--method', 'PHash', '--outfile', FILENAME])
    assert result.exit_code == 0
    assert os.path.isfile(FILENAME) is True
    # cleanup
    os.remove(FILENAME)


def test_max_distance_threshold_int():
    runner = CliRunner()
    result = runner.invoke(find_duplicates,
                           ['--image_dir', PATH_IMAGE_DIR, '--method', 'PHash', '--max_distance_threshold', '20'])
    assert result.exit_code == 0


def test_max_distance_threshold_no_int():
    runner = CliRunner()
    result = runner.invoke(find_duplicates,
                           ['--image_dir', PATH_IMAGE_DIR, '--method', 'PHash', '--max_distance_threshold', '0.5'])
    assert result.exit_code == 2


def test_scores_boolean():
    runner = CliRunner()
    result = runner.invoke(find_duplicates,
                           ['--image_dir', PATH_IMAGE_DIR, '--method', 'PHash', '--scores', 'False'])
    assert result.exit_code == 0


def test_scores_no_boolean():
    runner = CliRunner()
    result = runner.invoke(find_duplicates,
                           ['--image_dir', PATH_IMAGE_DIR, '--method', 'PHash', '--scores', 'hello'])
    assert result.exit_code == 2
