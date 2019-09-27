## class CNN
Find duplicates using CNN and/or generate CNN features given a single image or a directory of images.

The module can be used for 2 purposes: Feature generation and duplicate detection.

- Feature generation: To propagate an image through a Convolutional Neural Network architecture and generate features. The generated features can be used at a later time for deduplication. Using the method 'encode_image', the CNN feature for a single image can be obtained while the 'encode_images' method can be used to get features for all images in a directory.

- Duplicate detection: Find duplicates either using the feature mapping generated previously using 'encode_images' or using a Path to the directory that contains the images that need to be deduplicated. 'find_duplciates' and 'find_duplicates_to_remove' methods are provided to accomplish these tasks.
### \_\_init\_\_
```python
def __init__()
```
Initialize a keras MobileNet model that is sliced at the last convolutional layer. Set the batch size for keras generators to be 64 samples. Set the input image size to (224, 224) for providing as input to MobileNet model.



### encode\_image
```python
def encode_image(image_file, image_array)
```
Generate CNN features for a single image.


##### Args
* **image_file**: Path to the image file.

* **image_array**: Image typecast to numpy array.

##### Returns
* **feature**: Features for the image in the form of numpy array.

##### Example usage
```python

from imagededup.methods import CNN
myencoder = CNN()
feature_vector = myencoder.encode_image(image_file='path/to/image.jpg')
OR
feature_vector = myencoder.encode_image(image_array=<numpy array of image>)

```

### encode\_images
```python
def encode_images(image_dir)
```
Generate CNN features for all images in a given directory of images.


##### Args
* **image_dir**: Path to the image directory.

##### Returns
* **dictionary**: Contains a mapping of filenames and corresponding numpy array of CNN features.

##### Example usage
```python

from imagededup.methods import CNN
myencoder = CNN()
feature_map = myencoder.encode_images(image_dir='path/to/image/directory')

```

### find\_duplicates
```python
def find_duplicates(image_dir, encoding_map, min_similarity_threshold, scores, outfile)
```
Find duplicates for each file. Take in path of the directory or encoding dictionary in which duplicates are to be detected above the given threshold. Return dictionary containing key as filename and value as a list of duplicate file names. Optionally, the cosine distances could be returned instead of just duplicate filenames for each query file.


##### Args
* **image_dir**: Path to the directory containing all the images or dictionary with keys as file names

* **encoding_map**: A dictionary containing mapping of filenames and corresponding CNN features.

* **min_similarity_threshold**: Threshold value (must be float between -1.0 and 1.0)

* **scores**: Boolean indicating whether similarity scores are to be returned along with retrieved duplicates.

* **outfile**: Name of the file to save the results.

##### Returns
* **dictionary**:  if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
            score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}. if scores is False, then a
            dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg', 'image1_duplicate2.jpg'],
            'image2.jpg':['image1_duplicate1.jpg',..], ..}

##### Example usage
```python

from imagededup.methods import CNN
myencoder = CNN()
duplicates = myencoder.find_duplicates(image_dir='path/to/directory', min_similarity_threshold=15, scores=True,
outfile='results.json')

OR

from imagededup.methods import CNN
myencoder = CNN()
duplicates = myencoder.find_duplicates(encoding_map=<mapping filename to cnn features>,
min_similarity_threshold=15, scores=True, outfile='results.json')

```

### find\_duplicates\_to\_remove
```python
def find_duplicates_to_remove(image_dir, encoding_map, min_similarity_threshold, outfile)
```
Give out a list of image file names to remove based on the similarity threshold. Does not remove the mentioned files.


##### Args
* **image_dir**: Path to the directory containing all the images or dictionary with keys as file names
           and values as numpy arrays which represent the CNN feature for the key image file.

* **encoding_map**: A dictionary containing mapping of filenames and corresponding CNN features.

* **min_similarity_threshold**: Threshold value (must be float between -1.0 and 1.0)

* **outfile**: Name of the file to save the results.

##### Returns
* **duplicates**: List of image file names that should be removed.

##### Example usage
```python

from imagededup.methods import CNN
myencoder = CNN()
duplicates = myencoder.find_duplicates_to_remove(image_dir='path/to/images/directory'),
min_similarity_threshold=15)

OR

from imagededup.methods import CNN
myencoder = CNN()
duplicates = myencoder.find_duplicates_to_remove(encoding_map=<mapping filename to cnn features>,
min_similarity_threshold=15, outfile='results.json')

```

