from pathlib import PurePath
from typing import List, Union, Tuple

import numpy as np
from PIL import Image

from imagededup.utils.logger import return_logger


IMG_FORMATS = ['JPEG', 'PNG', 'BMP', 'MPO', 'PPM', 'TIFF', 'GIF']
logger = return_logger(__name__)


def _image_array_reshaper(image_arr):
    if len(image_arr.shape) == 3:
        return image_arr
    elif len(image_arr.shape) == 2:
        image_arr = np.tile(image_arr[..., np.newaxis], (1, 1, 3))
        return image_arr
    else:
        raise ValueError('Expected number of image array dimensions are 3 for rgb image and 2 for grayscale image!')


def preprocess_image(
    image, target_size: Tuple[int, int] = None, grayscale: bool = False
) -> np.ndarray:
    """
    Take as input an image as numpy array or Pillow format. Returns an array version of optionally resized and grayed
    image.

    Args:
        image: numpy array or a pillow image.
        target_size: Size to resize the input image to.
        grayscale: A boolean indicating whether to grayscale the image.

    Returns:
        A numpy array of the processed image.
    """
    print(f'inside preprocess_image, image shape: {image.size}')
    if isinstance(image, np.ndarray):
        image = image.astype('uint8')
        image = _image_array_reshaper(image)
        image_pil = Image.fromarray(image)

    elif isinstance(image, Image.Image):
        image_pil = image
    else:
        raise ValueError('Input is expected to be a numpy array or a pillow object!')

    if target_size:
        image_pil = image_pil.resize(target_size, Image.ANTIALIAS)
        print(f'inside preprocess_image, after resizing shape: {image_pil.size}')

    if grayscale:
        image_pil = image_pil.convert('L')
        print(f'inside preprocess_image, after grayscale shape: {image_pil.size}')

    return np.array(image_pil).astype('uint8')


def load_image(
    image_file: Union[PurePath, str],
    target_size: Tuple[int, int] = None,
    grayscale: bool = False,
    img_formats: List[str] = IMG_FORMATS,
) -> np.ndarray:
    """
    Load an image given its path. Returns an array version of optionally resized and grayed image. Only allows images
    of types described by img_formats argument.

    Args:
        image_file: Path to the image file.
        target_size: Size to resize the input image to.
        grayscale: A boolean indicating whether to grayscale the image.
        img_formats: List of allowed image formats that can be loaded.
    """
    try:
        img = Image.open(image_file)
        print(f'inside load_image, image shape: {img.size}')

        # validate image format
        if img.format not in img_formats:
            logger.warning(f'Invalid image format {img.format}!')
            return None

        else:
            if img.mode != 'RGB':
                print(f'inside load_image, not rgb image shape: {img.size}')
                # convert to RGBA first to avoid warning
                # we ignore alpha channel if available
                img = img.convert('RGBA').convert('RGB')
                print(f'converted load_image, image shape: {img.size}')

            img = preprocess_image(img, target_size=target_size, grayscale=grayscale)

            return img

    except Exception as e:
        logger.warning(f'Invalid image file {image_file}:\n{e}')
        return None
