from imagededup.utils.logger import return_logger
import os
import numpy as np
from PIL import Image
from pathlib import Path, PosixPath
from typing import Tuple, List, Optional


"""
? Allow acceptance of os.path in addition to already existing Path and numpy image array
Todo:
1. parallelize files validation
2. Add possibilities to ignore invalid directory images and still run hashes and cnn feat gen
3. ? save invalid images to a file
"""


IMG_FORMATS = ['JPEG', 'PNG', 'BMP']
logger = return_logger(__name__, os.getcwd())


def preprocess_image(image, target_size=None, grayscale: bool = False):
    if isinstance(image, np.ndarray):
        image = image.astype('uint8')
        image_pil = Image.fromarray(image)

    elif isinstance(image, Image.Image):
        image_pil = image

    if target_size:
        image_pil = image_pil.resize(target_size, Image.ANTIALIAS)

    if grayscale:
        image_pil = image_pil.convert('L')

    return image_pil


def load_image(image_file: Path, target_size=None, grayscale: bool = False,
               img_formats: List[str] = IMG_FORMATS) -> Image:
    try:
        img = Image.open(image_file)
        f = img.format  # store format after opening as it gets lost after conversion

        # validate image format
        if img.format not in img_formats:
            logger.warning(f'Invalid image format {img.format}!')
            return None

        else:
            if img.mode != 'RGB':
                # convert to RGBA first to avoid warning
                # we ignore alpha channel if available
                img = img.convert('RGBA').convert('RGB')

            img = preprocess_image(img, target_size=target_size, grayscale=grayscale)

            return np.array(img).astype('uint8')

    except Exception as e:
        logger.warning(f'Invalid image file {image_file}:\n{e}')
        return None


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
    Accepts either path of an image or a numpy array and processes it to feed it to CNN or hashing methods.

    :param path_image: PosixPath to the image file or Image typecast to numpy array.
    :param resize_dims: Dimensions for resizing the image
    :param for_hashing: Boolean flag to determine whether the function is being run for hashing or CNN based approach
    :return: A processed image as numpy array
    """

    if isinstance(path_image, PosixPath):
        # _validate_single_image(path_image)
        im = load_image(image_file=path_image)
    elif isinstance(path_image, np.ndarray):
        im = path_image.astype('uint8')  # fromarray can't take float32/64
        im = Image.fromarray(im)
    else:
        raise TypeError('Check Input Format! Input should be either a Path Variable or a numpy array!')
    im_arr = _image_preprocess(im, resize_dims, for_hashing)
    return im_arr
