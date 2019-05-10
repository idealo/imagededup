from imagededup.hashing import Hashing
import numpy as np
from pathlib import Path
from PIL import Image


def load_image(path_image):
    return Image.open(path_image)


def test_if_using_path_works():
    pass


def test_if_using_numpy_array_works():
    pass


def test_phash_resize():
    """Resize one image to (300, 300) and check that hamming distance between hashes is not too different"""
    hash = Hashing()
    hash_im_1 = hash.phash(Path('tests/data/images/ukbench00120.jpg'))
    hash_im_2 = hash.phash(Path('tests/data/images/ukbench00120_resize.jpg'))
    hamdist = Hashing.hamming_distance(hash_im_1, hash_im_2)
    print(hamdist)
    assert hamdist < 3


def test_phash_small_rotation():
    """Rotate image slightly (2 degrees) and check that hamming distance between hashes is not too different"""
    hash = Hashing()
    orig_image = load_image(Path('tests/data/images/ukbench00120.jpg'))
    hash_im_1 = hash.phash(np.array(orig_image))
    rotated_image = np.array(orig_image.rotate(1))
    hash_im_2 = hash.phash(rotated_image)
    hamdist = Hashing.hamming_distance(hash_im_1, hash_im_2)
    print(hamdist)
    assert hamdist < 3


def test_phash_distinct_images():
    """Put in distinct images and check that hamming distance between hashes is very different"""
    pass

## Write above 3 TCs for avg hash and dhash too


