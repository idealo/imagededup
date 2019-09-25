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
  <img src="_readme_figures/103500.jpg" width="300" />
  <img src="_readme_figures/103501.jpg" width="300" />
</p>

Traditional methods such as hashing algorithms are particularly good at finding exact duplicates while more modern 
methods involving convolutional neural networks are also adept at finding near duplicates due to their ability to 
capture basic contours in images.

This package provides functionality to address both problems. Additionally, an evaluation framework is also provided to
judge the quality of deduplication. Following details the functionality provided by the package:
* Finding duplicates in a directory using one of the following algorithms:
    - [Convolutional Neural Network](https://arxiv.org/abs/1704.04861)
    - [Perceptual hashing](http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html)
    - [Difference hashing](http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html)
    - [Wavelet hashing](https://fullstackml.com/wavelet-image-hash-in-python-3504fdd282b5)
    - [Average hashing](http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html)
* Generation of features for images using one of the above stated algorithms.
* Framework to evaluate effectiveness of deduplication  given a ground truth mapping.
* Plotting duplicates found for a given image file.

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

`pip install imagededup`

Install imagededup from the GitHub source:

```
git clone https://github.com/idealo/image-dedup.git
cd image-dedup  
python setup.py install
```  

## Getting started
### Finding duplicates
There are two methods available to perform deduplication:
- [find_duplicates()](#find_duplicates)
- [find_duplicates_to_remove()](#find_duplicates_to_remove)

#### find_duplicates
To deduplicate an image directory, the general api is:
```python
from imagededup.methods import <method-name>
method_object = <method-name>()
duplicates = method_object.find_duplicates(image_dir='path/to/image/directory',
                                           <threshold-parameter-value>)
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

##### Options
- *image_dir*: Optional, directory where all image files are present.

- *encoding_map*: Optional, used instead of *image_dir* attribute. Set it equal to the dictionary of file names and 
corresponding features (hashes/cnn encodings). The mentioned dictionary can be generated using the corresponding 
[*encode_images*](#feature-generation-for-all-images-in-a-directory) method.
- *scores*: Setting it to *True* returns the scores representing the hamming distance (for hashing) or cosine similarity
 (for cnn) of each of the duplicate file names from the key file. In this case, the returned 'duplicates' dictionary has
  the following content:
```
{
  'image1.jpg': [('image1_duplicate1.jpg', score),
                 ('image1_duplicate2.jpg', score)],
  'image2.jpg': [..],
  ..
}
```
Each key in the *duplicates* dictionary corresponds to a file in the image directory passed to the image_dir parameter 
of the find_duplicates function. The value is a list of all tuples representing the file names and corresponding scores 
in the image directory that were found to be duplicates for the key file.

- *outfile*: Name of file to which the returned duplicates dictionary is to be written. *None* by default.

- threshold parameter:
  * *min_similarity_threshold* for cnn method indicating the minimum amount of cosine similarity that should exist 
  between the key image and a candidate image so that the candidate image can be considered as a duplicate for the key 
  image. Should be a float between -1.0 and 1.0. Default value is 0.9.

  * *max_distance_threshold* for hashing methods indicating the maximum amount of hamming distance that can exist 
  between the key image and a candidate image so that the candidate image can be considered as a duplicate for the key 
  image. Should be an int between 0 and 64. Default value is 10.

##### Considerations

- The returned duplicates dictionary contains symmetric relationships i.e., if an image *i* is a duplicate of image *j*,
 then image *j* must also be a duplicate of image *i*. Let's say that the image directory only consists of images *i* 
 and *j*, then the duplicates dictionary would have the following content:
```
{
  'i': ['j'],
  'j': ['i']
}
```
- If an image in the image directory can't be loaded, no features are generated for the image. Hence, the image is 
disregarded for deduplication and has no entry in the returned *duplicates* dictionary.

##### Examples

To deduplicate an image directory using perceptual hashing, with a maximum allowed hamming distance of 12, scores 
returned along with duplicate filenames and the returned dictionary saved to file 'my_duplicates.json', use the 
following:
```python
from imagededup.methods import PHash
phasher = PHash()
duplicates = phasher.find_duplicates(image_dir='path/to/image/directory',
                                     max_distance_threshold=12, 
                                     scores=True, 
                                     outfile='my_duplicates.json')
```
To deduplicate an image directory using cnn, with a minimum cosine similarity of 0.85, no scores returned and the 
returned dictionary saved to file 'my_duplicates.json', use the following:

```python
from imagededup.methods import CNN
cnn_encoder = CNN()
duplicates = cnn_encoder.find_duplicates(image_dir='path/to/image/directory', 
                                         min_similarity_threshold=0.85, 
                                         scores=False, 
                                         outfile='my_duplicates.json')
```
#### find_duplicates_to_remove
Returns a list of files in the image directory that are considered as duplicates. Does **NOT** remove the said files.

The api is similar to *find_duplicates* function (except the *score* attribute in *find_duplicates*). This function 
allows the return of a single list of file names in directory that are found to be duplicates.
The general api for the method is as below:
```python
from imagededup.methods import <method-name>
method_object = <method-name>()
duplicates = method_object.find_duplicates_to_remove(image_dir='path/to/image/directory', 
                                                     <threshold-parameter-value>)
```
In this case, the returned variable *duplicates* is a list containing the name of image files that are found to be 
duplicates of some file in the directory:
```
[
  'image1_duplicate1.jpg',
  'image1_duplicate2.jpg'
  ,..
]
```

##### Options
- *image_dir*: Optional, directory where all image files are present.

- *encoding_map*: Optional, used instead of image_dir attribute. Set it equal to the dictionary of file names and 
corresponding features (hashes/cnn encodings). The mentioned dictionary can be generated using the corresponding 
[*encode_images*](#feature-generation-for-all-images-in-a-directory) method. Each key in the 'duplicates' dictionary corresponds to a file in the image directory passed to 
the image_dir parameter of the find_duplicates function. The value is a list of all tuples representing the file names 
and corresponding scores in the image directory that were found to be duplicates for the key file.

- *outfile*: Name of file to which the returned duplicates dictionary is to be written. *None* by default.

- threshold parameter:
  * *min_similarity_threshold* for cnn method indicating the minimum amount of cosine similarity that should exist 
  between the key image and a candidate image so that the candidate image can be considered as a duplicate for the key 
  image. Should be a float between -1.0 and 1.0. Default value is 0.9.

  * *max_distance_threshold* for hashing methods indicating the maximum amount of hamming distance that can exist 
  between the key image and a candidate image so that the candidate image can be considered as a duplicate for the key 
  image. Should be an int between 0 and 64. Default value is 10.

##### Considerations
- This method must be used with caution. The symmetric nature of duplicates imposes an issue of marking one image as 
duplicate and the other as original. Consider the following *duplicates* dictionary:
```
{
  '1.jpg': ['2.jpg'],
  '2.jpg': ['1.jpg', '3.jpg'],
  '3.jpg': ['2.jpg']
}
```
In this case, it is possible to remove only *2.jpg* which leaves *1.jpg* and *3.jpg* as non-duplicates of each other. 
However, it is also possible to remove both *1.jpg* and *3.jpg* leaving only *2.jpg*. The *find_duplicates_to_remove* 
method makes this decision based on the alphabetical sorting of filenames in the directory. In the above example, the 
filename *1.jpg* appears alphabetically before *2.jpg*. So, *1.jpg* would be retained, while its duplicate, *2.jpg*, 
would be marked as a duplicate. Once *2.jpg* is marked as duplicate, its own found duplicates would be disregarded. 
Thus, *1.jpg* and *3.jpg* would not be considered as duplicates. So, the final return would be:
```
['2.jpg']
```
This leaves *1.jpg* and *3.jpg* as non-duplicates in the directory.
If the user does not wish to impose this heuristic, it is advised to use [*find_duplicates*](#find_duplicates) function and use a custom 
heuristic to mark a file as duplicate.

- If an image in the image directory can't be loaded, no features are generated for the image. Hence, the image is 
disregarded for deduplication and has no entry in the returned *duplicates* dictionary.

##### Examples

To deduplicate an image directory using perceptual hashing, with a maximum allowed hamming distance of 12, and the 
returned list saved to file 'my_duplicates.json', use the following:
```python
from imagededup.methods import PHash
phasher = PHash()
duplicates = phasher.find_duplicates_to_remove(image_dir='path/to/image/directory', 
                                               max_distance_threshold=12, 
                                               outfile='my_duplicates.json')
```
To deduplicate an image directory using cnn, with a minimum cosine similarity of 0.85 and the returned list saved to 
file 'my_duplicates.json', use the following:

```python
from imagededup.methods import CNN
cnn_encoder = CNN()
duplicates = cnn_encoder.find_duplicates_to_remove(image_dir='path/to/image/directory', 
                                                   min_similarity_threshold=0.85, 
                                                   outfile='my_duplicates.json')
```

### Feature generation
It might be desirable to only generate the hashes/cnn encodings for a given image or all images in the directory instead
 of directly deduplicating using find_duplicates method. Features can be generated for a directory of images or for a single 
 image:
- [For all images in a directory](#feature-generation-for-all-images-in-a-directory)
- [For a single image](#feature-generation-for-a-single-image)


#### Feature generation for all images in a directory
To generate encodings for all images in an image directory *encode_images* function can be used. The general api for 
using *encode_images* is:
```python
from imagededup.methods import <method-name>
method_object = <method-name>()
encodings = method_object.encode_images(image_dir='path/to/image/directory')
```
where the returned variable *encodings* is a dictionary mapping image file names to corresponding encoding:
```
{
  'image1.jpg': <feature-image-1>,
  'image2.jpg': <feature-image-2>,
   ..
}
```
For hashing algorithms, the features are 64 bit hashes represented as 16 character hexadecimal strings.

For cnn, the features are numpy array with shape (1, 1024).

##### Considerations

If an image in the image directory can't be loaded, no features are generated for the image. Hence, there is no entry 
for the image in the returned encodings dictionary.

##### Examples

Generating features using Difference hash,
```python
from imagededup.methods import DHash
dhasher = DHash()
encodings = dhasher.encode_images(image_dir='path/to/image/directory')
```

#### Feature generation for a single image
To generate encodings for a single image *encode_image* function can be used. The general api for 
using *encode_image* is:
```python
from imagededup.methods import <method-name>
method_object = <method-name>()
encoding = method_object.encode_image(image_file='path/to/image/file')
```
where the returned variable *encoding* is either a hexadecimal string if a hashing method is used or a (1, 1024) numpy 
array if cnn is used.

##### Options
- image_file: Optional, path to the image file for which encodings are to be generated.
- image_array: Optional, used instead of *image_file* attribute. A numpy array representing the image.

##### Considerations

If the image can't be loaded, no features are generated for the image and *None* is returned.

##### Examples

Generating features using Difference hash,
```python
from imagededup.methods import DHash
dhasher = DHash()
encoding = dhasher.encode_image(image_file='path/to/image/file')
```

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
##### Options
- ground_truth_map:  A dictionary representing ground truth with filenames as key and a list of duplicate filenames as 
value.
- retrieved_map: A dictionary representing retrieved duplicates with filenames as key and a list of retrieved duplicate 
filenames as value.
- metric can take one of the following values:
  - 'map'
  - 'ndcg'
  - 'jaccard'
  - 'classification': Returns per class precision, recall, f1-score, support
  - 'all' (default, returns all the above metrics)


##### Considerations
- Presently, the ground truth map should be prepared manually by the user. Symmetric relations between duplicates must 
be represented in the ground truth map. If an image *i* is a duplicate for image *j*, then *j* must also be represented as a
 duplicate of *i*. Absence of symmetric relations will lead to an exception.

- Both the ground_truth_map and retrieved_map must have the same keys.

### Plotting duplicates of an image
Once a duplicate dictionary corresponding to an image directory has been obtained (using [find_duplicates](#find_duplicates)), duplicates 
for an image can be plotted using *plot_duplicates* method as below:
```python
from imagededup.utils import plot_duplicates
plot_duplicates(image_dir, duplicate_map, filename)
```
where *filename* is the file for which duplicates are to be plotted.

##### Options
- *image_dir*: Directory where all image files are present.

- *duplicate_map*: A dictionary representing retrieved duplicates with filenames as key and a list of retrieved duplicate 
filenames as value. A duplicate_map with scores can also be passed (obtained from [find_duplicates](#find_duplicates)
function with scores attribute set to True).

- *filename*: Image file name for which duplicates are to be plotted.

- *outfile*: Name of the file the plot should be saved to. *None* by default.

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