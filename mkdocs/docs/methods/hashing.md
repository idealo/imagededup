## class Hashing
Find duplicates using hashing algorithms and/or generate hashes given a single image or a directory of images.

The module can be used for 2 purposes: Feature generation and duplicate detection.

- Feature generation: To generate hashes using specific hashing method. The generated hashes can be used at a later time for deduplication. Using the method 'encode_image' from the specific hashing method object, the hash for a single image can be obtained while the 'encode_images' method can be used to get hashes for all images in a directory.

- Duplicate detection: Find duplicates either using the feature mapping generated previously using 'encode_images' or using a Path to the directory that contains the images that need to be deduplicated. 'find_duplciates' and 'find_duplicates_to_remove' methods are provided to accomplish these tasks.
### \_\_init\_\_
```python
def __init__()
```

### hamming\_distance
```python
def hamming_distance(hash1, hash2)
```
Calculate the hamming distance between two hashes. If length of hashes is not 64 bits, then pads the length to be 64 for each hash and then calculates the hamming distance.


##### Args
* **hash1**: hash string

* **hash2**: hash string

##### Returns
* **hamming_distance**: Hamming distance between the two hashes.


### encode\_image
```python
def encode_image(image_file, image_array)
```
Generate hash for a single image.


##### Args
* **image_file**: Path to the image file.

* **image_array**: Image typecast to numpy array.

##### Returns
* **hash**: A 16 character hexadecimal string hash for the image.

##### Example usage
```python

from imagededup.methods import <hash-method>
myencoder = <hash-method>()
myhash = myencoder.encode_image(image_file='path/to/image.jpg')
OR
myhash = myencoder.encode_image(image_array=<numpy array of image>)

```

### encode\_images
```python
def encode_images(image_dir)
```
Generate hashes for all images in a given directory of images.


##### Args
* **image_dir**: Path to the image directory.

##### Returns
* **dictionary**:  A dictionary that contains a mapping of filenames and corresponding 64 character hash string
            such as {'Image1.jpg': 'hash_string1', 'Image2.jpg': 'hash_string2', ...}

##### Example usage
```python

from imagededup.methods import <hash-method>
myencoder = <hash-method>()
mapping = myencoder.encode_images('path/to/directory')

```

### find\_duplicates
```python
def find_duplicates(image_dir, encoding_map, max_distance_threshold, scores, outfile)
```
Find duplicates for each file. Takes in path of the directory or encoding dictionary in which duplicates are to be detected. All images with hamming distance less than or equal to the max_distance_threshold are regarded as duplicates. Returns dictionary containing key as filename and value as a list of duplicate file names. Optionally, the below the given hamming distance could be returned instead of just duplicate filenames for each query file.


##### Args
* **image_dir**: Path to the directory containing all the images or dictionary with keys as file names
           and values as hash strings for the key image file.

* **encoding_map**: A dictionary containing mapping of filenames and corresponding hashes.

* **max_distance_threshold**: Hamming distance between two images below which retrieved duplicates are valid.
                        (must be an int between 0 and 64)

* **scores**: Boolean indicating whether Hamming distances are to be returned along with retrieved duplicates.

* **outfile**: Name of the file to save the results.

##### Returns
* **duplicates dictionary**:  if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
            score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}. if scores is False, then a
            dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg', 'image1_duplicate2.jpg'],
            'image2.jpg':['image1_duplicate1.jpg',..], ..}

##### Example usage
```python

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

### find\_duplicates\_to\_remove
```python
def find_duplicates_to_remove(image_dir, encoding_map, max_distance_threshold, outfile)
```
Give out a list of image file names to remove based on the hamming distance threshold threshold. Does not remove the mentioned files.


##### Args
* **image_dir**: Path to the directory containing all the images or dictionary with keys as file names
           and values as hash strings for the key image file.

* **encoding_map**: A dictionary containing mapping of filenames and corresponding hashes.

* **max_distance_threshold**: Hamming distance between two images below which retrieved duplicates are valid.
                        (must be an int between 0 and 64)

* **outfile**: Name of the file to save the results.

##### Returns
* **duplicates**: List of image file names that are found to be duplicate of me other file in the directory.

##### Example usage
```python

from imagededup.methods import <hash-method>
myencoder = <hash-method>()
duplicates = myencoder.find_duplicates_to_remove(image_dir='path/to/images/directory'),
max_distance_threshold=15)

OR

from imagededup.methods import <hash-method>
myencoder = <hash-method>()
duplicates = myencoder.find_duplicates(encoding_map=<mapping filename to hashes>,
max_distance_threshold=15, outfile='results.json')

```

## class PHash
Inherits from Hashing base class and implements perceptual hashing (Implementation reference: http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html).

Offers all the functionality mentioned in hashing class.
##### Example usage
```python

# Perceptual hash for images
from imagededup.methods import PHash
phasher = PHash()
perceptual_hash = phasher.encode_image(image_file = 'path/to/image.jpg')
OR
perceptual_hash = phasher.encode_image(image_array = <numpy image array>)
OR
perceptual_hashes = phasher.encode_images(image_dir = 'path/to/directory')  # for a directory of images

# Finding duplicates:
from imagededup.methods import PHash
phasher = PHash()
duplicates = phasher.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True)
OR
duplicates = phasher.find_duplicates(encoding_map=encoding_map, max_distance_threshold=15, scores=True)

# Finding duplicates to return a single list of duplicates in the image collection
from imagededup.methods import PHash
phasher = PHash()
files_to_remove = phasher.find_duplicates_to_remove(image_dir='path/to/images/directory',
max_distance_threshold=15)
OR
files_to_remove = phasher.find_duplicates_to_remove(encoding_map=encoding_map, max_distance_threshold=15)

```
### \_\_init\_\_
```python
def __init__()
```

## class AHash
Inherits from Hashing base class and implements average hashing. (Implementation reference: http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html)

Offers all the functionality mentioned in hashing class.
##### Example usage
```python

# Average hash for images
from imagededup.methods import AHash
ahasher = AHash()
average_hash = ahasher.encode_image(image_file = 'path/to/image.jpg')
OR
average_hash = ahasher.encode_image(image_array = <numpy image array>)
OR
average_hashes = ahasher.encode_images(image_dir = 'path/to/directory')  # for a directory of images

# Finding duplicates:
from imagededup.methods import AHash
ahasher = AHash()
duplicates = ahasher.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True)
OR
duplicates = ahasher.find_duplicates(encoding_map=encoding_map, max_distance_threshold=15, scores=True)

# Finding duplicates to return a single list of duplicates in the image collection
from imagededup.methods import AHash
ahasher = AHash()
files_to_remove = ahasher.find_duplicates_to_remove(image_dir='path/to/images/directory',
max_distance_threshold=15)
OR
files_to_remove = ahasher.find_duplicates_to_remove(encoding_map=encoding_map, max_distance_threshold=15)

```
### \_\_init\_\_
```python
def __init__()
```

## class DHash
Inherits from Hashing base class and implements difference hashing. (Implementation reference: http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html)

Offers all the functionality mentioned in hashing class.
##### Example usage
```python

# Difference hash for images
from imagededup.methods import DHash
dhasher = DHash()
difference_hash = dhasher.encode_image(image_file = 'path/to/image.jpg')
OR
difference_hash = dhasher.encode_image(image_array = <numpy image array>)
OR
difference_hashes = dhasher.encode_images(image_dir = 'path/to/directory')  # for a directory of images

# Finding duplicates:
from imagededup.methods import DHash
dhasher = DHash()
duplicates = dhasher.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True)
OR
duplicates = dhasher.find_duplicates(encoding_map=encoding_map, max_distance_threshold=15, scores=True)

# Finding duplicates to return a single list of duplicates in the image collection
from imagededup.methods import DHash
dhasher = DHash()
files_to_remove = dhasher.find_duplicates_to_remove(image_dir='path/to/images/directory',
max_distance_threshold=15)
OR
files_to_remove = dhasher.find_duplicates_to_remove(encoding_map=encoding_map, max_distance_threshold=15)

```
### \_\_init\_\_
```python
def __init__()
```

## class WHash
Inherits from Hashing base class and implements wavelet hashing. (Implementation reference: https://fullstackml.com/wavelet-image-hash-in-python-3504fdd282b5)

Offers all the functionality mentioned in hashing class.
##### Example usage
```python

# Wavelet hash for images
from imagededup.methods import WHash
whasher = WHash()
wavelet_hash = whasher.encode_image(image_file = 'path/to/image.jpg')
OR
wavelet_hash = whasher.encode_image(image_array = <numpy image array>)
OR
wavelet_hashes = whasher.encode_images(image_dir = 'path/to/directory')  # for a directory of images

# Finding duplicates:
from imagededup.methods import WHash
whasher = WHash()
duplicates = whasher.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True)
OR
duplicates = whasher.find_duplicates(encoding_map=encoding_map, max_distance_threshold=15, scores=True)

# Finding duplicates to return a single list of duplicates in the image collection
from imagededup.methods import WHash
whasher = WHash()
files_to_remove = whasher.find_duplicates_to_remove(image_dir='path/to/images/directory',
max_distance_threshold=15)
OR
files_to_remove = whasher.find_duplicates_to_remove(encoding_map=encoding_map, max_distance_threshold=15)

```
### \_\_init\_\_
```python
def __init__()
```

