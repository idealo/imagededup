# Finding duplicates
There are two methods available to find duplicates:

- [find_duplicates()](#find_duplicates)
- [find_duplicates_to_remove()](#find_duplicates_to_remove)

## find_duplicates()
To find duplicates in an image directory, the general api is:
```python
from imagededup.methods import <method-name>
method_object = <method-name>()
duplicates = method_object.find_duplicates(image_dir='path/to/image/directory',
                                           <threshold-parameter-value>)
```

Duplicates can also be found if encodings of the images are available:
```
from imagededup.methods import <method-name>
method_object = <method-name>()
duplicates = method_object.find_duplicates(encoding_map,
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
The 'method-name' corresponds to one of the deduplication methods available and can be set to:

- PHash
- AHash
- DHash
- WHash
- CNN

#### Options
- *image_dir*: Optional, directory where all image files are present.

- *encoding_map*: Optional, used instead of *image_dir* attribute. Set it equal to the dictionary of file names and 
corresponding encodings (hashes/cnn encodings). The mentioned dictionary can be generated using the corresponding 
[*encode_images*](encoding_generation.md) method.
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
of the find_duplicates function. The value is a list of tuples representing the file names and corresponding scores 
in the image directory that were found to be duplicates of the key file.

- *outfile*: Name of file to which the returned duplicates dictionary is to be written, must be a json. *None* by default.

- threshold parameter:
    - *min_similarity_threshold* for cnn method indicating the minimum amount of cosine similarity that should exist 
  between the key image and a candidate image so that the candidate image can be considered as a duplicate of the key 
  image. Should be a float between -1.0 and 1.0. Default value is 0.9.

    - *max_distance_threshold* for hashing methods indicating the maximum amount of hamming distance that can exist 
  between the key image and a candidate image so that the candidate image can be considered as a duplicate of the key 
  image. Should be an int between 0 and 64. Default value is 10.

#### Considerations

- The returned duplicates dictionary contains symmetric relationships i.e., if an image *i* is a duplicate of image *j*,
 then image *j* must also be a duplicate of image *i*. Let's say that the image directory only consists of images *i* 
 and *j*, then the duplicates dictionary would have the following content:
```
{
  'i': ['j'],
  'j': ['i']
}
```
- If an image in the image directory can't be loaded, no encodings are generated for the image. Hence, the image is 
disregarded for deduplication and has no entry in the returned *duplicates* dictionary.

#### Examples

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
## find_duplicates_to_remove()
Returns a list of files in the image directory that are considered as duplicates. Does **NOT** remove the said files.

The api is similar to *find_duplicates* function (except the *score* attribute in *find_duplicates*). This function 
allows the return of a single list of file names in directory that are found to be duplicates.
The general api for the method is as below:
```python
from imagededup.methods import <method-name>
method_object = <method-name>()
duplicates = method_object.find_duplicates_to_remove(image_dir='path/to/image/directory', 
                                                     <threshold-parameter-value>)
OR

duplicates = method_object.find_duplicates_to_remove(encoding_map=encoding_map, 
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
The 'method-name' corresponds to one of the deduplication methods available and can be set to:

- PHash
- AHash
- DHash
- WHash
- CNN

#### Options
- *image_dir*: Optional, directory where all image files are present.

- *encoding_map*: Optional, used instead of image_dir attribute. Set it equal to the dictionary of file names and 
corresponding encodings (hashes/cnn encodings). The mentioned dictionary can be generated using the corresponding 
[*encode_images*](encoding_generation.md) method.

- *outfile*: Name of file to which the returned duplicates dictionary is to be written, must be a json. *None* by default.

- threshold parameter:
    - *min_similarity_threshold* for cnn method indicating the minimum amount of cosine similarity that should exist 
  between the key image and a candidate image so that the candidate image can be considered as a duplicate for the key 
  image. Should be a float between -1.0 and 1.0. Default value is 0.9.

    - *max_distance_threshold* for hashing methods indicating the maximum amount of hamming distance that can exist 
  between the key image and a candidate image so that the candidate image can be considered as a duplicate for the key 
  image. Should be an int between 0 and 64. Default value is 10.

#### Considerations
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
method can thus, return either of the outputs. In the above example, let's say that *1.jpg* is retained, while its 
duplicate, *2.jpg*, is marked as a duplicate. Once *2.jpg* is marked as duplicate, its own found duplicates would be 
disregarded.  Thus, *1.jpg* and *3.jpg* would not be considered as duplicates. So, the final return would be:
```
['2.jpg']
```
This leaves *1.jpg* and *3.jpg* as non-duplicates in the directory.
If the user does not wish to impose this heuristic, it is advised to use [*find_duplicates*](#find_duplicates) function 
and use a custom heuristic to mark a file as duplicate.

- If an image in the image directory can't be loaded, no encodings are generated for the image. Hence, the image is 
disregarded for deduplication and has no entry in the returned *duplicates* dictionary.

#### Examples

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
