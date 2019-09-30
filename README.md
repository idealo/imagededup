# imagededup

Finding duplicates in an image dataset is a recurring task. imagededup is a python package that provides functionality
to carry out this task effectively. The deduplication problem generally caters to 2 broad issues:

* Finding exact duplicates

<p align="center">
  <img src="_readme_figures/103500.jpg" width="300" />
  <img src="_readme_figures/103500.jpg" width="300" />
</p>

* Finding near duplicates

<p align="center">
  <img src="../_readme_figures/103500.jpg" width="300" />
  <img src="../_readme_figures/103501.jpg" width="300" />
</p>

Traditional methods such as hashing algorithms are particularly good at finding exact duplicates while more modern 
methods involving convolutional neural networks are also adept at finding near duplicates due to their ability to 
capture basic contours in images.

This package provides functionality to address both problems. Additionally, an evaluation framework is also provided to
judge the quality of deduplication. Following details the functionality provided by the package:

- Finding duplicates in a directory using one of the following algorithms:
    - [Convolutional Neural Network](https://arxiv.org/abs/1704.04861)
    - [Perceptual hashing](http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html)
    - [Difference hashing](http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html)
    - [Wavelet hashing](https://fullstackml.com/wavelet-image-hash-in-python-3504fdd282b5)
    - [Average hashing](http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html)
- Generation of features for images using one of the above stated algorithms.
- Framework to evaluate effectiveness of deduplication  given a ground truth mapping.
- Plotting duplicates found for a given image file.

imagededup is compatible with Python 3.6 and is distributed under the Apache 2.0 license.

## Table of contents
- [Installation](#installation)
- [Finding duplicates](#finding-duplicates)
- [Feature generation](#feature-generation)
- [Evaluation of deduplication](#evaluation-of-deduplication-quality)
- [Plotting duplicates](#plotting-duplicates-of-an-image)
- [Contribute](#contribute)
- [Citation](#citation)
- [Maintainers](#maintainers)
- [License](#copyright)

## Installation
There are two ways to install imagededup:

Install imagededup from PyPI (recommended):

```
pip install imagededup
```

Install imagededup from the GitHub source:

```
git clone https://github.com/idealo/image-dedup.git
cd image-dedup  
python setup.py install
```  

## Quick start
### Finding duplicates
There are two methods available to perform deduplication:

- [find_duplicates()](#find_duplicates)
- [find_duplicates_to_remove()](#find_duplicates_to_remove)

#### find_duplicates
To deduplicate an image directory using perceptual hashing:
```python
from imagededup.methods import PHash
phasher = PHash()
duplicates = phasher.find_duplicates(image_dir='path/to/image/directory', max_distance_threshold=15)
```
Other hashing methods can be used instead of PHash: Ahash, DHash, WHash

To deduplicate an image directory using cnn:
```python
from imagededup.methods import CNN
cnn_encoder = CNN()
duplicates = cnn_encoder.find_duplicates(image_dir='path/to/image/directory', min_similarity_threshold=0.85)
```
where the returned variable *duplicates* is a dictionary with the following content:
```
{
  'image1.jpg': ['image1_duplicate1.jpg',
                'image1_duplicate2.jpg'],
  'image2.jpg': [..],
  ..
}
```
Each key in the *duplicates* dictionary corresponds to a file in the image directory passed to the *image_dir* parameter
of the *find_duplicates* function. The value is a list of all file names in the image directory that were found to be 
duplicates for the key file.

For an advanced usage, look at the user guide.

#### find_duplicates_to_remove
Returns a list of files in the image directory that are considered as duplicates. Does **NOT** remove the said files.

The api is similar to *find_duplicates* function (except the *score* attribute in *find_duplicates*). This function 
allows the return of a single list of file names in directory that are found to be duplicates.

To deduplicate an image directory using cnn:
```python
from imagededup.methods import CNN
cnn_encoder = CNN()
duplicates = cnn_encoder.find_duplicates_to_remove(image_dir='path/to/image/directory', min_similarity_threshold=0.85)
```
*duplicates* is a list containing the name of image files that are found to be 
duplicates of some file in the directory:
```
[
  'image1_duplicate1.jpg',
  'image1_duplicate2.jpg'
  ,..
]
```

For an advanced usage, look at the user guide.

### Feature generation
To only generate the hashes/cnn encodings for a given image or all images in the directory:

- [Feature generation for all images in a directory](#feature-generation-for-all-images-in-a-directory)
- [Feature generation for a single image](#feature-generation-for-a-single-image)


#### Feature generation for all images in a directory
*encode_images* function can be used here:
```python
from imagededup.methods import Dhash
dhasher = Dhash()
encodings = dhasher.encode_images(image_dir='path/to/image/directory')
```
where the returned *encodings*:
```
{
  'image1.jpg': <feature-image-1>,
  'image2.jpg': <feature-image-2>,
   ..
}
```
For hashing algorithms, the features are 64 bit hashes represented as 16 character hexadecimal strings.

For cnn, the features are numpy array with shape (1, 1024).

#### Feature generation for a single image
To generate encodings for a single image *encode_image* function can be used:
```python
from imagededup.methods import AHash
ahasher = AHash()
encoding = ahasher.encode_image(image_file='path/to/image/file')
```
where the returned variable *encoding* is either a hexadecimal string if a hashing method is used or a (1, 1024) numpy 
array if cnn is used.

### Evaluation of deduplication quality
To determine the quality of deduplication algorithm and the corresponding threshold, an evaluation framework is provided.

Given a ground truth mapping consisting of file names and a list of duplicates for each file along with a retrieved 
mapping from the deduplication algorithm for the same files, the following metrics can be obtained using the framework:

- [Mean Average Precision](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision) (MAP)
- [Mean Normalized Discounted Cumulative Gain](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) (NDCG)
- [Jaccard Index](https://en.wikipedia.org/wiki/Jaccard_index)
- Per class [Precision](https://en.wikipedia.org/wiki/Precision_and_recall) (class 0 = non-duplicate image pairs, class 1 = duplicate image pairs)
- Per class [Recall](https://en.wikipedia.org/wiki/Precision_and_recall) (class 0 = non-duplicate image pairs, class 1 = duplicate image pairs)
- Per class [f1-score](https://en.wikipedia.org/wiki/F1_score) (class 0 = non-duplicate image pairs, class 1 = duplicate image pairs)

The api for obtaining these metrics  is as below:
```python
from imagededup.evaluation import evaluate
metrics = evaluate(ground_truth_map, retrieved_map, metric='<metric-name>')
```
where the returned variable *metrics* is a dictionary containing the following content:
```
{
  'map': <map>,
  'ndcg': <mean ndcg>,
  'jaccard': <mean jaccard index>,
  'precision': <numpy array having per class precision>,
  'recall': <numpy array having per class recall>,
  'f1-score': <numpy array having per class f1-score>,
  'support': <numpy array having per class support>
}
```

### Plotting duplicates of an image
Duplicates for an image can be plotted using *plot_duplicates* method as below:
```python
from imagededup.utils import plot_duplicates
plot_duplicates(image_dir, duplicate_map, filename)
```
where *duplicate_map* is the duplciate map obtained after running [find_duplicates()](#find_duplicates) and  *filename* is the file for which duplicates are to be plotted.

The output looks as below:

![figs](_readme_figures/plot_dups.png)

## Contribute
We welcome all kinds of contributions.
See the [Contribution](CONTRIBUTING.md) guide for more details.

## Citation
Please cite Imagededup in your publications if this is useful for your research. Here is an example BibTeX entry:
```
@misc{idealods2019imagededup,
  title={Imagededup},
  author={Tanuj Jain and Christopher Lennan and Zubin John},
  year={2019},
  howpublished={\url{https://github.com/idealo/image-dedup}},
}
```

## Maintainers
* Tanuj Jain, github: [tanujjain](https://github.com/tanujjain)
* Christopher Lennan, github: [clennan](https://github.com/clennan)

## Copyright
See [LICENSE](LICENSE) for details.