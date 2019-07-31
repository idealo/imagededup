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


def load_image(path: Path, target_size=None, grayscale: bool = False) -> Image:
    img = Image.open(path)
    f = img.format  # store format after opening as it gets lost after conversion

    if img.mode != 'RGB':
        # convert to RGBA first to avoid warning
        # we ignore alpha channel if available
        img = img.convert('RGBA').convert('RGB')

    if target_size:
        img = img.resize(target_size)

    if grayscale:
        img = img.convert('L')

    img.format = f  # reassign format for later validation checks

    return img


def validate_image(
    file_name: Path, img_formats: List[str] = IMG_FORMATS
) -> Tuple[bool, Optional[Exception]]:
    """
    Checks whether File is valid image file:
        - file exists
        - file is readable
        - file is an image
    Args:
        file_name: Absolute path of file.
     Returns:
        True if file is valid image file.
        False else.
    """

    valid_image = False
    error = None

    try:
        img = load_image(file_name)

        if img.format in img_formats:
            img.load()  # Pillow uses lazy loading, so need to explicitly load

            valid_image = True

        else:
            error = f'Image format {img.format} not in supported formats {img_formats}'

    except Exception as e:
        error = e

    return valid_image, error



def validate_images(image_dir: PosixPath) -> List:
    """Checks if all files in path_dir are valid images and return valid files if return_file set to True.
    :param path_dir: A PosixPath to the image directory.
    :param return_file: Boolean indicating if a list of valid files is to be returned.
    """

    files = [Path(i.absolute()) for i in image_dir.glob('*') if not i.name.startswith('.')]  # ignore hidden files

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
    Accepts either path of an image or a numpy array and processes it to feed it to CNN or hashing methods.

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
