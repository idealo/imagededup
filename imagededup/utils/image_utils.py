import os
import numpy as np
from PIL import Image
from pathlib import Path, PosixPath
from typing import Tuple, List

"""
? Allow acceptance of os.path in addition to already existing Path and numpy image array
Todo:
1. parallelize files validation
2. Add possibilities to ignore invalid directory images and still run hashes and cnn feat gen
3. ? save invalid images to a file
"""


def _load_image(path_image: PosixPath) -> Image:
    """
    Load image given the PosixPath path to the image.
    :param path_image: A PosixPath to the image.
    :return: A Pillow Image if image gets loaded successfully.
    """
    try:
        img = Image.open(path_image)

        if img.mode != 'RGB':
            # convert to RGBA first to avoid warning
            # we ignore alpha channel if available
            img = img.convert('RGBA').convert('RGB')
        return img
    except Exception as e:
        raise Exception(f'{type(e)} Image can not be loaded!')


def _validate_single_image(path_image: PosixPath) -> int:
    """
    Checks if a files is a valid images (check for existence and correct extension).
    :param path_image: A PosixPath to the image.
    :return: integer 1 for a successful check.
    """
    if not os.path.exists(path_image):
        raise FileNotFoundError('Ensure that the file exists at the specified path!')
    str_name = path_image.name

    if not (str_name.endswith('.jpeg') or str_name.endswith('.jpg') or str_name.endswith('.bmp') or
            str_name.endswith('.png')):
        raise TypeError('Image formats supported: .jpg, .jpeg, .bmp, .png')
    return 1  # returns 1 if validation successful


def check_directory_files(path_dir: PosixPath, return_file: bool = False) -> List:
    """Checks if all files in path_dir are valid images and return valid files if return_file set to True.
    :param path_dir: A PosixPath to the image directory.
    :param return_file: Boolean indicating if a list of valid files is to be returned.
    """

    files = [Path(i.absolute()) for i in path_dir.glob('*') if not i.name.startswith('.')]  # ignore hidden files

    invalid_image_files = []
    for i in files:
        try:
            _validate_single_image(i)
        except (FileNotFoundError, TypeError):
            invalid_image_files.append(i)
            
    if len(invalid_image_files) != 0:
        raise Exception(f'Please remove the following invalid files to run deduplication: {invalid_image_files}')
    if return_file:
        return files  # The logic reaches here only if there are no invalid files


def _image_preprocess(pillow_image: Image, resize_dims: Tuple[int, int], for_hashing: bool = True) -> np.ndarray:
    """
    Resizes and typecasts a pillow image to numpy array.

    :param pillow_image: A Pillow type image to be processed.
    :return: A numpy array of processed image.
    """
    if for_hashing:
        im_res = pillow_image.resize(resize_dims, Image.ANTIALIAS)
        im_res = im_res.convert('L')  # convert to grayscale (i.e., single channel)
    else:
        im_res = pillow_image.resize(resize_dims)

    im_arr = np.array(im_res)
    return im_arr


def convert_to_array(path_image, resize_dims: Tuple[int, int], for_hashing: bool = True) -> np.ndarray:
    """
    Accepts either path of an image or a numpy array and processes it to feed it to CNN.

    :param path_image: PosixPath to the image file or Image typecast to numpy array.
    :param resize_dims: Dimensions for resizing the image
    :param for_hashing: Boolean flag to determine whether the function is being run for hashing or CNN based approach
    :return: A processed image as numpy array
    """

    if isinstance(path_image, PosixPath):
        _validate_single_image(path_image)
        im = _load_image(path_image=path_image)
    elif isinstance(path_image, np.ndarray):
        im = path_image.astype('uint8')  # fromarray can't take float32/64
        im = Image.fromarray(im)
    else:
        raise TypeError('Check Input Format! Input should be either a Path Variable or a numpy array!')
    im_arr = _image_preprocess(im, resize_dims, for_hashing)
    return im_arr
