import os
import numpy as np
from PIL import Image
from pathlib import Path, PosixPath
from typing import Tuple

"""
? Allow acceptance of os.path in addition to already existing Path and numpy image array
Todo:
1. parallelize files validation
2. add possibilities to ignore invalid directory images and still run hashes and cnn feat gen
"""


def load_image(path: PosixPath) -> Image:
    img = Image.open(path)

    if img.mode != 'RGB':
        # convert to RGBA first to avoid warning
        # we ignore alpha channel if available
        img = img.convert('RGBA').convert('RGB')
    return img


def load_valid_image(path_image: PosixPath, load=False):
    """
    Checks that file exists and is has a valid extension. If valid, loads and returns the image
    :param path_image: PosixPath of the image
    :param load: Flag to indicate if the valid image is to be loaded
    :return: loaded image if the image is valid, else corresponding exception raised
    """
    if not os.path.exists(path_image):
        raise FileNotFoundError('Ensure that the file exists at the specified path!')
    str_name = path_image.name

    if not (str_name.endswith('.jpeg') or str_name.endswith('.jpg') or str_name.endswith('.bmp') or
            str_name.endswith('.png')):
        raise TypeError('Image formats supported: .jpg, .jpeg, .bmp, .png')
    if load:
        try:
            img = load_image(path_image)
            return img
        except Exception as e:
            print(f'{e}: Image can not be loaded!')
            raise e
    else:
        return 1


def check_directory_files(path_dir: PosixPath):
    """Checks if all files in path_dir are valid images.
    """

    files = [Path(i.absolute()) for i in path_dir.glob('*') if not i.name.startswith('.')]  # ignore hidden files

    invalid_image_files = []
    for i in files:
        try:
            load_valid_image(i, load=False)
        except (FileNotFoundError, TypeError):
            invalid_image_files.append(i)
            
    if len(invalid_image_files) != 0:
        raise Exception(f'Please remove the following invalid files to run deduplication: {invalid_image_files}')


def _image_preprocess(pillow_image: Image, resize_dims: Tuple[int, int], hashmethod: bool = True) -> np.ndarray:
    """
    Resizes and typecasts a pillow image to numpy array.

    :param pillow_image: A Pillow type image to be processed.
    :return: A numpy array of processed image.
    """
    if hashmethod:
        im_res = pillow_image.resize(resize_dims, Image.ANTIALIAS)
        im_res = im_res.convert('L')  # convert to grayscale (i.e., single channel)
    else:
        im_res = pillow_image.resize(resize_dims)

    im_arr = np.array(im_res)
    return im_arr


def convert_to_array(path_image, resize_dims: Tuple[int, int], hashmethod: bool = True) -> np.ndarray:
    """
    Accepts either path of an image or a numpy array and processes it to feed it to CNN.

    :param path_image: PosixPath to the image file or Image typecast to numpy array.
    :return: A processed image as numpy array
    """

    if isinstance(path_image, PosixPath):
        # im = Image.open(path_image)
        im = load_valid_image(path_image=path_image, load=True)
    elif isinstance(path_image, np.ndarray):
        im = path_image.astype('uint8')  # fromarray can't take float32/64
        im = Image.fromarray(im)
    else:
        raise TypeError('Check Input Format! Input should be either a Path Variable or a numpy array!')
    im_arr = _image_preprocess(im, resize_dims, hashmethod)
    return im_arr
