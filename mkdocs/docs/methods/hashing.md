## class Hashing
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

##### Example usage:
```python

from imagededup.methods import <hash-method>
myencoder = <hash-method>()
hash = myencoder.encode_image(image_file='path/to/image.jpg')
OR
hash = myencoder.encode_image(image_array=<numpy array of image>)

```

### encode\_images
```python
def encode_images(image_dir)
```
Generate hashes for all images in a given directory of images.


##### Args
* **image_dir**: Path to the image directory.

##### Returns
* **dictionary**:  A dictionary that contains a mapping of filenames and corresponding 64 character hash string such as {'Image1.jpg': 'hash_string1', 'Image2.jpg': 'hash_string2', ...}

##### Example usage:
```python

from imagededup.methods import <hash-method>
myencoder = <hash-method>()
mapping = myencoder.encode_images('path/to/directory')

```

### find\_duplicates
```python
def find_duplicates(image_dir, encoding_map, max_distance_threshold, scores, outfile)
```
Find duplicates for each file. Takes in path of the directory or encoding dictionary in which duplicates are to be detected below the given hamming distance threshold. Returns dictionary containing key as filename and value as a list of duplicate file names. Optionally, the below the given hamming distance could be returned instead of just duplicate filenames for each query file.


##### Args
* **image_dir**: Path to the directory containing all the images or dictionary with keys as file names

* **encoding_map**: A dictionary containing mapping of filenames and corresponding hashes.

* **max_distance_threshold**: Hamming distance between two images below which retrieved duplicates are valid.

* **scores**: Boolean indicating whether Hamming distances are to be returned along with retrieved duplicates.

* **outfile**: Name of the file to save the results.

##### Returns
* **dictionary**:  if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg', score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}. if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg', 'image1_duplicate2.jpg'], 'image2.jpg':['image1_duplicate1.jpg',..], ..}

##### Example usage:
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

* **encoding_map**: A dictionary containing mapping of filenames and corresponding hashes.

* **max_distance_threshold**: Hamming distance between two images below which retrieved duplicates are valid.

* **outfile**: Name of the file to save the results.

##### Returns
* **duplicates list**: List of image file names that should be removed.

##### Example usage:
```python

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

## class PHash
Find duplicates using perceptual hashing algorithm and/or generate perceptual hashes given a single image or a directory of images. The module can be used for 2 purposes: Feature generation and duplicate detection.

Feature generation:

To generate perceptual hashes. The generated hashes can be used at a later time for deduplication. There are two possibilities to get hashes: 1. At a single image level: Using the method 'encode_image', the perceptual hash for a single image can be obtained.
##### Example usage:
```python

from imagededup.methods import PHash
myencoder = PHash()
hash = myencoder.encode_image('path/to/image.jpg')


2. At a directory level: In case perceptual hash for several images needs to be generated, the images can be
placed in a directory and hashes for all of the images can be obtained using the 'encode_images' method.
Example:

from imagededup.methods import PHash
myencoder = PHash()
hashes = myencoder.encode_images('path/to/directory')

Duplicate detection:

Find duplicates either using the hashes generated previously using 'encode_images' or using a Path to the
directory that contains the image dataset that needs to be deduplicated. There are 2 inputs that can be provided to
the find_duplicates function:
1. Dictionary generated using 'encode_images' function above.
Example:

from imagededup.methods import PHash
myencoder = PHash()
duplicates = myencoder.find_duplicates(encoding_map, max_distance_threshold=15, scores=True)

2. Using the Path of the directory where all images are present.
Example:

from imagededup.methods import PHash
myencoder = PHash()
duplicates = myencoder.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True)

If a list of file names to remove are desired, then the function find_duplicates_to_remove can be used with either
the path to the image directory as input or the dictionary with features. A threshold for maximum hamming distance
should be considered.

Example:

from imagededup.methods import PHash
myencoder = PHash()
files_to_remove = myencoder.find_duplicates_to_remove(image_dir='path/to/images/directory',
max_distance_threshold=15)

```
### \_\_init\_\_
```python
def __init__()
```

## class AHash
Find duplicates using average hashing algorithm and/or generates average hashes given a single image or a directory of images. The module can be used for 2 purposes: Feature generation and duplicate detection.

Feature generation: To generate average hashes. The generated hashes can be used at a later time for deduplication. There are two possibilities to get hashes: 1. At a single image level: Using the method 'encode_image', the average hash for a single image can be obtained.
##### Example usage:
```python

from imagededup.methods import AHash
myencoder = AHash()
hash = myencoder.encode_image('path/to/image.jpg')

2. At a directory level: In case average hash for several images need to be generated, the images can be
placed in a directory and hashes for all of the images can be obtained using the 'encode_images' method.
Example:

from imagededup.methods import AHash
myencoder = AHash()
hashes = myencoder.encode_images('path/to/directory')


Duplicate detection:

Find duplicates either using the hashes generated previously using 'encode_images' or using a Path to the
directory that contains the image dataset that needs to be deduplicated. There are 2 inputs that can be provided to
the find_duplicates function:
1. Dictionary generated using 'encode_images' function above.
Example:

from imagededup.methods import AHash
myencoder = AHash()
duplicates = myencoder.find_duplicates(encoding_map, max_distance_threshold=15, scores=True)

2. Using the Path of the directory where all images are present.
Example:

from imagededup.methods import AHash
myencoder = AHash()
duplicates = myencoder.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True)

If a list of file names to remove are desired, then the function find_duplicates_to_remove can be used with either
the path to the image directory as input or the dictionary with features. A threshold for maximum hamming distance
should be considered.

Example:

from imagededup.methods import AHash
myencoder = AHash()
files_to_remove = myencoder.find_duplicates_to_remove(image_dir='path/to/images/directory',
max_distance_threshold=15)

```
### \_\_init\_\_
```python
def __init__()
```

## class DHash
Find duplicates using difference hashing algorithm and/or generates difference hashes given a single image or a directory of images. The module can be used for 2 purposes: Feature generation and duplicate detection.

Feature generation: To generate difference hashes. The generated hashes can be used at a later time for deduplication. There are two possibilities to get hashes: 1. At a single image level: Using the method 'encode_image', the difference hash for a single image can be obtained.
##### Example usage:
```python

from imagededup.methods import DHash
myencoder = DHash()
hash = myencoder.encode_image('path/to/image.jpg')

2. At a directory level: In case difference hash for several images need to be generated, the images can be
placed in a directory and hashes for all of the images can be obtained using the 'encode_images' method.
Example:

from imagededup.methods import DHash
myencoder = DHash()
hashes = myencoder.encode_images('path/to/directory')


Duplicate detection:

Find duplicates either using the hashes generated previously using 'encode_images' or using a Path to the
directory that contains the image dataset that needs to be deduplicated. There are 2 inputs that can be provided to
the find_duplicates function:
1. Dictionary generated using 'encode_images' function above.
Example:

from imagededup.methods import DHash
myencoder = DHash()
duplicates = myencoder.find_duplicates(encoding_map, max_distance_threshold=15, scores=True)

2. Using the Path of the directory where all images are present.
Example:

from imagededup.methods import DHash
myencoder = DHash()
duplicates = myencoder.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True)

If a list of file names to remove are desired, then the function find_duplicates_to_remove can be used with either
the path to the image directory as input or the dictionary with features. A threshold for maximum hamming distance
should be considered.

Example:

from imagededup.methods import DHash
myencoder = DHash()
files_to_remove = myencoder.find_duplicates_to_remove(image_dir='path/to/images/directory',
max_distance_threshold=15)

```
### \_\_init\_\_
```python
def __init__()
```

## class WHash
Find duplicates using wavelet hashing algorithm and/or generates wavelet hashes given a single image or a directory of images. The module can be used for 2 purposes: Feature generation and duplicate detection.

Feature generation: To generate wavelet hashes. The generated hashes can be used at a later time for deduplication. There are two possibilities to get hashes: 1. At a single image level: Using the method 'encode_image', the wavelet hash for a single image can be obtained.
##### Example usage:
```python

from imagededup.methods import WHash
myencoder = WHash()
hash = myencoder.encode_image('path/to/image.jpg')

2. At a directory level: In case wavelet hash for several images need to be generated, the images can be
placed in a directory and hashes for all of the images can be obtained using the 'encode_images' method.
Example:

from imagededup.methods import WHash
myencoder = WHash()
hashes = myencoder.encode_images('path/to/directory')


Duplicate detection:

Find duplicates either using the hashes generated previously using 'encode_images' or using a Path to the
directory that contains the image dataset that needs to be deduplicated. There are 2 inputs that can be provided to
the find_duplicates function:
1. Dictionary generated using 'encode_images' function above.
Example:

from imagededup.methods import WHash
myencoder = WHash()
duplicates = myencoder.find_duplicates(encoding_map, max_distance_threshold=15, scores=True)

2. Using the Path of the directory where all images are present.
Example:

from imagededup.methods import WHash
myencoder = WHash()
duplicates = myencoder.find_duplicates(image_dir='path/to/directory', max_distance_threshold=15, scores=True)

If a list of file names to remove are desired, then the function find_duplicates_to_remove can be used with either
the path to the image directory as input or the dictionary with features. A threshold for maximum hamming distance
should be considered.

Example:

from imagededup.methods import WHash
myencoder = WHash()
files_to_remove = myencoder.find_duplicates_to_remove(image_dir='path/to/images/directory',
max_distance_threshold=15)

```
### \_\_init\_\_
```python
def __init__()
```

