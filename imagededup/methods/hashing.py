import os
from pathlib import PosixPath, Path
from typing import Dict, List, Optional

import pywt
import numpy as np
from scipy.fftpack import dct


from imagededup.handlers.search.retrieval import HashEval
from imagededup.utils.general_utils import get_files_to_remove, save_json, parallelise
from imagededup.utils.image_utils import load_image, preprocess_image


"""
TODO:
Wavelet hash: Zero the LL coeff, reconstruct image, then get wavelet transform
Wavelet hash: Allow possibility of different wavelet functions

"""


class Hashing:
    def __init__(self) -> None:
        self.target_size = (8, 8)  # resizing to dims

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> float:
        """
        Calculate the hamming distance between two hashes. If length of hashes is not 64 bits, then pads the length
        to be 64 for each hash and then calculates the hamming distance.

        Args:
            hash1: hash string
            hash2: hash string

        Returns:
            Hamming distance between the two hashes.
        """
        hash1_bin = bin(int(hash1, 16))[2:].zfill(
            64
        )  # zfill ensures that len of hash is 64 and pads MSB if it is < A
        hash2_bin = bin(int(hash2, 16))[2:].zfill(64)
        return np.sum([i != j for i, j in zip(hash1_bin, hash2_bin)])

    @staticmethod
    def _array_to_hash(hash_mat: np.ndarray) -> str:
        """
        Convert a matrix of binary numerals to 64 character hash.

        Args:
            hash_mat: A numpy array consisting of 0/1 values.

        Returns:
            An hexadecimal hash string.
        """
        return ''.join('%0.2x' % x for x in np.packbits(hash_mat))

    def encode_image(
        self, image_file=None, image_array: Optional[np.ndarray] = None
    ) -> str:
        """
        Generate hash for a single image.

        Args:
            image_file: Path to the image file.
            image_array: Image typecast to numpy array.

        Returns:
            A 64 character string hash for the image.

        Example usage:
        ```
        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        hash = myencoder.encode_image(image_file='path/to/image.jpg')
        OR
        hash = myencoder.encode_image(image_array=<numpy array of image>)
        ```
        """
        try:
            if image_file and os.path.exists(image_file):
                image_file = Path(image_file)
                image_pp = load_image(
                    image_file=image_file, target_size=self.target_size, grayscale=True
                )

            elif isinstance(image_array, np.ndarray):
                image_pp = preprocess_image(
                    image=image_array, target_size=self.target_size, grayscale=True
                )
            else:
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError('Please provide either image file path or image array!')

        return self._hash_func(image_pp) if isinstance(image_pp, np.ndarray) else None

    def encode_images(self, image_dir=None):
        """
        Generate hashes for all images in a given directory of images.

        Args:
            image_dir: Path to the image directory.

        Returns:
            A dictionary that contains a mapping of filenames and corresponding 64 character hash string.

        Example usage:
        ```
        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        mapping = myencoder.encode_images('path/to/directory')

        'mapping' contains: {'Image1.jpg': 'hash_string1', 'Image2.jpg': 'hash_string2', ...}
        ```
        """
        if not os.path.isdir(image_dir):
            raise ValueError('Please provide a valid directory path!')

        image_dir = Path(image_dir)

        files = [
            i.absolute() for i in image_dir.glob('*') if not i.name.startswith('.')
        ]  # ignore hidden files

        print(f'Start: Calculating hashes...')

        hashes = parallelise(self.encode_image, files)
        hash_initial_dict = dict(zip([f.name for f in files], hashes))
        hash_dict = {
            k: v for k, v in hash_initial_dict.items() if v
        }  # To ignore None (returned if some probelm with image file)

        print(f'End: Calculating hashes!')
        return hash_dict

    def _hash_algo(self, image_array: np.ndarray):
        pass

    def _hash_func(self, image_array: np.ndarray):
        hash_mat = self._hash_algo(image_array)
        return self._array_to_hash(hash_mat)

    # search part

    @staticmethod
    def _check_hamming_distance_bounds(thresh: int) -> None:
        """
        Check if provided threshold is valid. Raises TypeError if wrong threshold variable type is passed or a
        ValueError if an out of range value is supplied.

        Args:
            thresh: Threshold value (must be int between 0 and 64)

        Raises:
            TypeError: If wrong variable type is provided.
            ValueError: If invalid value is provided.
        """
        if not isinstance(thresh, int):
            raise TypeError('Threshold must be an int between 0 and 64')
        elif thresh < 0 or thresh > 64:
            raise ValueError('Threshold must be an int between 0 and 64')
        else:
            return None

    def _find_duplicates_dict(
        self,
        encoding_map: Dict[str, str],
        max_distance_threshold: int = 10,
        scores: bool = False,
        outfile: Optional[str] = None,
    ) -> Dict:
        """
        Take in dictionary {filename: encoded image}, detects duplicates below the given hamming distance threshold
        and returns a dictionary containing key as filename and value as a list of duplicate filenames. Optionally,
        the hamming distances could be returned instead of just duplicate filenames for each query file.

        Args:
            encoding_map: Dictionary with keys as file names and values as encoded images (hashes).
            max_distance_threshold: Hamming distance between two images below which retrieved duplicates are valid.
            scores: Boolean indicating whether hamming distance scores are to be returned along with retrieved
            duplicates.

        Returns:
            if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
            score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}
            if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg',
            'image1_duplicate2.jpg'], 'image2.jpg':['image1_duplicate1.jpg',..], ..}
        """
        print('Start: Evaluating hamming distances for getting duplicates')

        result_set = HashEval(
            test=encoding_map,
            queries=encoding_map,
            distance_function=self.hamming_distance,
            threshold=max_distance_threshold,
            search_method='bktree',
        )

        print('End: Evaluating hamming distances for getting duplicates')

        self.results = result_set.retrieve_results(scores=scores)
        if outfile:
            save_json(self.results, outfile)
        return self.results

    def _find_duplicates_dir(
        self,
        image_dir: PosixPath,
        max_distance_threshold: int = 10,
        scores: bool = False,
        outfile: Optional[str] = None,
    ) -> Dict:
        """
        Take in path of the directory in which duplicates are to be detected below the given hamming distance
        threshold. Returns dictionary containing key as filename and value as a list of duplicate file names.
        Optionally, the hamming distances could be returned instead of just duplicate filenames for each query file.

        Args:
            image_dir: Path to the directory containing all the images.
            max_distance_threshold: Hamming distance between two images below which retrieved duplicates are valid.
            scores: Boolean indicating whether Hamming distances are to be returned along with retrieved duplicates.
            outfile: Name of the file the results should be written to.

        Returns:
            if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
            score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}
            if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg',
            'image1_duplicate2.jpg'], 'image2.jpg':['image1_duplicate1.jpg',..], ..}
        """
        encoding_map = self.encode_images(image_dir)
        results = self._find_duplicates_dict(
            encoding_map=encoding_map,
            max_distance_threshold=max_distance_threshold,
            scores=scores,
            outfile=outfile,
        )
        return results

    def find_duplicates(
        self,
        image_dir: PosixPath = None,
        encoding_map: Dict[str, str] = None,
        max_distance_threshold: int = 10,
        scores: bool = False,
        outfile: Optional[str] = None,
    ) -> Dict:
        """
        Find duplicates for each file. Takes in path of the directory or encoding dictionary in which duplicates are to
        be detected below the given hamming distance threshold. Returns dictionary containing key as filename and value
        as a list of duplicate file names. Optionally, the below the given hamming distance could be returned instead of
        just duplicate filenames for each query file.

        Args:
            image_dir: Path to the directory containing all the images or dictionary with keys as file names
            and values as hash strings for the key image file.
            encoding_map: A dictionary containing mapping of filenames and corresponding hashes.
            max_distance_threshold: Hamming distance between two images below which retrieved duplicates are valid.
            (must be an int between 0 and 64)
            scores: Boolean indicating whether Hamming distances are to be returned along with retrieved duplicates.
            outfile: Name of the file to save the results.

        Returns:
            if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
            score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}
            if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg',
            'image1_duplicate2.jpg'], 'image2.jpg':['image1_duplicate1.jpg',..], ..}

            Example usage:
        ```
        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        duplicates = myencoder.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True,
        outfile='results.json')

        OR

        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        duplicates = myencoder.find_duplicates(encoding_map=<mapping filename to hashes>,
        max_distance_threshold=15, scores=True, outfile='results.json')
        ```
        """
        self._check_hamming_distance_bounds(thresh=max_distance_threshold)
        if image_dir:
            result = self._find_duplicates_dir(
                image_dir=image_dir,
                max_distance_threshold=max_distance_threshold,
                scores=scores,
                outfile=outfile,
            )
        elif encoding_map:
            result = self._find_duplicates_dict(
                encoding_map=encoding_map,
                max_distance_threshold=max_distance_threshold,
                scores=scores,
                outfile=outfile,
            )
        else:
            raise ValueError('Provide either an image directory or encodings!')
        return result

    def find_duplicates_to_remove(
        self,
        image_dir: PosixPath = None,
        encoding_map: Dict[str, str] = None,
        max_distance_threshold: int = 10,
        outfile: Optional[str] = None,
    ) -> List:
        """
        Give out a list of image file names to remove based on the hamming distance threshold threshold. Does not
        remove the mentioned files.

        Args:
            image_dir: Path to the directory containing all the images or dictionary with keys as file names
            and values as hash strings for the key image file.
            encoding_map: A dictionary containing mapping of filenames and corresponding hashes.
            max_distance_threshold: Hamming distance between two images below which retrieved duplicates are valid.
            (must be an int between 0 and 64)
            outfile: Name of the file to save the results.

        Returns:
            List of image file names that should be removed.

        Example usage:
        ```
        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        list_of_files_to_remove = myencoder.find_duplicates_to_remove(image_dir='path/to/images/directory'),
        max_distance_threshold=15)

        OR

        from imagededup.methods import <hash-method>
        myencoder = <hash-method>()
        duplicates = myencoder.find_duplicates(encoding_map=<mapping filename to hashes>,
        max_distance_threshold=15, outfile='results.json')
        ```
        """
        result = self.find_duplicates(
            image_dir=image_dir,
            encoding_map=encoding_map,
            max_distance_threshold=max_distance_threshold,
            scores=False,
        )
        files_to_remove = get_files_to_remove(result)
        if outfile:
            save_json(files_to_remove, outfile)
        return files_to_remove


class PHash(Hashing):
    """
    Find duplicates using perceptual hashing algorithm and/or generate perceptual hashes given a single image or a
    directory of images. The module can be used for 2 purposes: Feature generation and duplicate detection.

    Feature generation:

    To generate perceptual hashes. The generated hashes can be used at a later time for deduplication. There are two
    possibilities to get hashes:
    1. At a single image level: Using the method 'encode_image', the perceptual hash for a single image can be
    obtained.
    Example usage:
    ```
    from imagededup.methods import PHash
    myencoder = PHash()
    hash = myencoder.encode_image('path/to/image.jpg')
    ```
    2. At a directory level: In case perceptual hash for several images needs to be generated, the images can be
    placed in a directory and hashes for all of the images can be obtained using the 'encode_images' method.
    Example usage:
    ```
    from imagededup.methods import PHash
    myencoder = PHash()
    hashes = myencoder.encode_images('path/to/directory')
        ```
    Duplicate detection:

    Find duplicates either using the hashes generated previously using 'encode_images' or using a Path to the
    directory that contains the image dataset that needs to be deduplicated. There are 2 inputs that can be provided to
    the find_duplicates function:
    1. Dictionary generated using 'encode_images' function above.
    Example usage:
    ```
    from imagededup.methods import PHash
    myencoder = PHash()
    duplicates = myencoder.find_duplicates(encoding_map, max_distance_threshold=15, scores=True)
    ```
    2. Using the Path of the directory where all images are present.
    Example usage:
    ```
    from imagededup.methods import PHash
    myencoder = PHash()
    duplicates = myencoder.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True)
    ```
    If a list of file names to remove are desired, then the function find_duplicates_to_remove can be used with either
    the path to the image directory as input or the dictionary with features. A threshold for maximum hamming distance
    should be considered.

    Example usage:
        ```
        from imagededup.methods import PHash
        myencoder = PHash()
        files_to_remove = myencoder.find_duplicates_to_remove(image_dir='path/to/images/directory',
        max_distance_threshold=15)
        ```
    """

    def __init__(self) -> None:
        super().__init__()
        self.__coefficient_extract = (8, 8)
        self.target_size = (32, 32)

    def _hash_algo(self, image_array):
        """
        Get perceptual hash of the input image.

        Args:
            image_array: numpy array that corresponds to the image.

        Returns:
            A string representing the perceptual hash of the image.

        Implementation reference: http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
        """
        dct_coef = dct(dct(image_array, axis=0), axis=1)

        # retain top left 8 by 8 dct coefficients
        dct_reduced_coef = dct_coef[
            : self.__coefficient_extract[0], : self.__coefficient_extract[1]
        ]

        # median of coefficients excluding the DC term (0th term)
        # mean_coef_val = np.mean(np.ndarray.flatten(dct_reduced_coef)[1:])
        median_coef_val = np.median(np.ndarray.flatten(dct_reduced_coef)[1:])

        # return mask of all coefficients greater than mean of coefficients
        hash_mat = dct_reduced_coef >= median_coef_val
        return hash_mat


class AHash(Hashing):
    """
    Find duplicates using average hashing algorithm and/or generates average hashes given a single image or a
    directory of images. The module can be used for 2 purposes: Feature generation and duplicate detection.

    Feature generation:
    To generate average hashes. The generated hashes can be used at a later time for deduplication. There are two
    possibilities to get hashes:
    1. At a single image level: Using the method 'encode_image', the average hash for a single image can be
    obtained.
    Example usage:
    ```
    from imagededup.methods import AHash
    myencoder = AHash()
    hash = myencoder.encode_image('path/to/image.jpg')
    ```
    2. At a directory level: In case average hash for several images need to be generated, the images can be
    placed in a directory and hashes for all of the images can be obtained using the 'encode_images' method.
    Example usage:
    ```
    from imagededup.methods import AHash
    myencoder = AHash()
    hashes = myencoder.encode_images('path/to/directory')
    ```

    Duplicate detection:

    Find duplicates either using the hashes generated previously using 'encode_images' or using a Path to the
    directory that contains the image dataset that needs to be deduplicated. There are 2 inputs that can be provided to
    the find_duplicates function:
    1. Dictionary generated using 'encode_images' function above.
    Example usage:
    ```
    from imagededup.methods import AHash
    myencoder = AHash()
    duplicates = myencoder.find_duplicates(encoding_map, max_distance_threshold=15, scores=True)
    ```
    2. Using the Path of the directory where all images are present.
    Example usage:
    ```
    from imagededup.methods import AHash
    myencoder = AHash()
    duplicates = myencoder.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True)
    ```
    If a list of file names to remove are desired, then the function find_duplicates_to_remove can be used with either
    the path to the image directory as input or the dictionary with features. A threshold for maximum hamming distance
    should be considered.

    Example usage:
        ```
        from imagededup.methods import AHash
        myencoder = AHash()
        files_to_remove = myencoder.find_duplicates_to_remove(image_dir='path/to/images/directory',
        max_distance_threshold=15)
        ```
    """

    def __init__(self) -> None:
        super().__init__()
        self.target_size = (8, 8)

    def _hash_algo(self, image_array: np.ndarray):
        """
        Get average hash of the input image.

        Args:
            image_array: numpy array that corresponds to the image.

        Returns:
            A string representing the average hash of the image.
        """
        avg_val = np.mean(image_array)
        hash_mat = image_array >= avg_val
        return hash_mat


class DHash(Hashing):
    """
    Find duplicates using difference hashing algorithm and/or generates difference hashes given a single image or a
    directory of images. The module can be used for 2 purposes: Feature generation and duplicate detection.

    Feature generation:
    To generate difference hashes. The generated hashes can be used at a later time for deduplication. There are two
    possibilities to get hashes:
    1. At a single image level: Using the method 'encode_image', the difference hash for a single image can be
    obtained.
    Example usage:
    ```
    from imagededup.methods import DHash
    myencoder = DHash()
    hash = myencoder.encode_image('path/to/image.jpg')
    ```
    2. At a directory level: In case difference hash for several images need to be generated, the images can be
    placed in a directory and hashes for all of the images can be obtained using the 'encode_images' method.
    Example usage:
    ```
    from imagededup.methods import DHash
    myencoder = DHash()
    hashes = myencoder.encode_images('path/to/directory')
    ```

    Duplicate detection:

    Find duplicates either using the hashes generated previously using 'encode_images' or using a Path to the
    directory that contains the image dataset that needs to be deduplicated. There are 2 inputs that can be provided to
    the find_duplicates function:
    1. Dictionary generated using 'encode_images' function above.
    Example usage:
    ```
    from imagededup.methods import DHash
    myencoder = DHash()
    duplicates = myencoder.find_duplicates(encoding_map, max_distance_threshold=15, scores=True)
    ```
    2. Using the Path of the directory where all images are present.
    Example usage:
    ```
    from imagededup.methods import DHash
    myencoder = DHash()
    duplicates = myencoder.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True)
    ```
    If a list of file names to remove are desired, then the function find_duplicates_to_remove can be used with either
    the path to the image directory as input or the dictionary with features. A threshold for maximum hamming distance
    should be considered.

    Example usage:
        ```
        from imagededup.methods import DHash
        myencoder = DHash()
        files_to_remove = myencoder.find_duplicates_to_remove(image_dir='path/to/images/directory',
        max_distance_threshold=15)
        ```
    """

    def __init__(self) -> None:
        super().__init__()
        self.target_size = (9, 8)

    def _hash_algo(self, image_array):
        """
        Get difference hash of the input image.

        Args:
            image_array: numpy array that corresponds to the image.

        Returns:
            A string representing the difference hash of the image.

        Implementation reference: http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
        """
        # Calculates difference between consecutive columns and return mask
        hash_mat = image_array[:, 1:] > image_array[:, :-1]
        return hash_mat


class WHash(Hashing):
    """
    Find duplicates using wavelet hashing algorithm and/or generates wavelet hashes given a single image or a
    directory of images. The module can be used for 2 purposes: Feature generation and duplicate detection.

    Feature generation:
    To generate wavelet hashes. The generated hashes can be used at a later time for deduplication. There are two
    possibilities to get hashes:
    1. At a single image level: Using the method 'encode_image', the wavelet hash for a single image can be
    obtained.
    Example usage:
    ```
    from imagededup.methods import WHash
    myencoder = WHash()
    hash = myencoder.encode_image('path/to/image.jpg')
    ```
    2. At a directory level: In case wavelet hash for several images need to be generated, the images can be
    placed in a directory and hashes for all of the images can be obtained using the 'encode_images' method.
    Example usage:
    ```
    from imagededup.methods import WHash
    myencoder = WHash()
    hashes = myencoder.encode_images('path/to/directory')
    ```

    Duplicate detection:

    Find duplicates either using the hashes generated previously using 'encode_images' or using a Path to the
    directory that contains the image dataset that needs to be deduplicated. There are 2 inputs that can be provided to
    the find_duplicates function:
    1. Dictionary generated using 'encode_images' function above.
    Example usage:
    ```
    from imagededup.methods import WHash
    myencoder = WHash()
    duplicates = myencoder.find_duplicates(encoding_map, max_distance_threshold=15, scores=True)
    ```
    2. Using the Path of the directory where all images are present.
    Example usage:
    ```
    from imagededup.methods import WHash
    myencoder = WHash()
    duplicates = myencoder.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True)
    ```
    If a list of file names to remove are desired, then the function find_duplicates_to_remove can be used with either
    the path to the image directory as input or the dictionary with features. A threshold for maximum hamming distance
    should be considered.

    Example usage:
        ```
        from imagededup.methods import WHash
        myencoder = WHash()
        files_to_remove = myencoder.find_duplicates_to_remove(image_dir='path/to/images/directory',
        max_distance_threshold=15)
        ```
    """

    def __init__(self) -> None:
        super().__init__()
        self.target_size = (256, 256)
        self.__wavelet_func = 'haar'

    def _hash_algo(self, image_array):
        """
        Get wavelet hash of the input image.

        Args:
            image_array: numpy array that corresponds to the image.

        Returns:
            A string representing the wavelet hash of the image.
        """
        # decomposition level set to 5 to get 8 by 8 hash matrix
        image_array = image_array / 255
        coeffs = pywt.wavedec2(data=image_array, wavelet=self.__wavelet_func, level=5)
        LL_coeff = coeffs[0]

        # median of LL coefficients
        median_coef_val = np.median(np.ndarray.flatten(LL_coeff))

        # return mask of all coefficients greater than mean of coefficients
        hash_mat = LL_coeff >= median_coef_val
        return hash_mat
