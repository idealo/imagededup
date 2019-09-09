# imagededup

Finding duplicates in an image dataset is a recurring task. imagededup is a python package that provides functionality to carry out this task effectively. The offered functionality includes:
1. Generation of featured for images using one of the following algorithms:
  - Perceptual hashing
  - Difference hashing
  - Wavelet hashing
  - Average hashing
  - Convolutional Neural Network
2. Obtaining duplicates based on the features generated
3. Framework to evaluate effectiveness of deduplication  given a ground truth mapping
4. Framework to find an effective deduplication algorithm with corresponding parameters given a grid of parameters for a target dataset

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

## Usage

#### Generate hashing features for an image directory

