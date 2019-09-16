# imagededup

Finding duplicates in an image dataset is a recurring task. imagededup is a python package that provides functionality 
to carry out this task effectively. The deduplication problem generally caters to 2 broad set of issues:
* Finding exact duplicates
* Finding near duplicates

Traditional methods such as hashing algorithms are particularly good at finding exact duplicates while more modern
methods involving convolutional neural networks are adept at finding near duplicates due to their ability to capture
basic contours in images.
 
This package provides functionality to address both problems. Additionally, an evaluation and experimentation framework,
 is also provided. Following details the functionality provided by the package:
* Generation of features for images using one of the following algorithms:
    - Convolutional Neural Network
    - Perceptual hashing
    - Difference hashing
    - Wavelet hashing
    - Average hashing
* Obtaining duplicates based on the features generated.
* Framework to evaluate effectiveness of deduplication  given a ground truth mapping.
* Framework to find an effective deduplication algorithm with corresponding parameters given a grid of parameters for a
 target dataset.

imagededup is compatible with Python 3.6 and is distributed under the Apache 2.0 license.

## Installation
There are two ways to install imagededup:

Install imagededup from PyPI (recommended):

`pip install imagededup`

Install imagededup from the GitHub source:

```
git clone https://github.com/idealo/image-dedup.git
cd imagededup  
python setup.py install
```  

## Getting started

#### Generate perceptual hash for one image given image path
```python
from imagededup.methods import PHash
phasher = PHash()
feature_map = phasher.encode_image(image_file='path/to/image/file')
```
#### Generate perceptual hash for one image given as a numpy array
```python
import numpy as np
from imagededup.methods import PHash
phasher = PHash()
feature_map = phasher.encode_image(image_array=np.array(image))
```

#### Generate perceptual hashes for all images in an image directory
```python
from imagededup.methods import PHash
phasher = PHash()
feature_map = phasher.encode_images(image_dir='path/to/image/directory')
```

#### Generate convolutional neural network features for all images in an image directory
```python
from imagededup.methods import CNN
cnn_encoder = CNN()
feature_map = cnn_encoder.encode_images(image_dir='path/to/image/directory')
```

#### Find duplicates given image directory
```python
from imagededup.methods import PHash
phasher = PHash()
map_filename_to_duplicates = phasher.find_duplicates(image_dir='path/to/image/directory')
```

#### Find duplicates based on the generated features
```python
from imagededup.methods import PHash
phasher = PHash()
feature_map = phasher.encode_images(image_dir='path/to/image/directory')
map_filename_to_duplicates = phasher.find_duplicates(encoding_map=feature_map)
```

#### Get a list of duplicate files given features
```python
from imagededup.methods import PHash
phasher = PHash()
duplicate_filenames = phasher.find_duplicates_to_remove(image_dir='path/to/image/directory')
```

#### Evaluate the effectiveness of deduplication if a ground truth on a minimal dataset is available
To be filled after implementing

#### Find an optimum deduplication method along with a threshold if a ground truth on a minimal dataset is available
To be filled after implementing

## Considerations

## Contribute

## Citation

## Maintainers


