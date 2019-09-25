### preprocess\_image
```python
def preprocess_image(image, target_size, grayscale)
```
Take as input an image as numpy array or Pillow format. Returns an array version of optionally resized and grayed image.


##### Args
* **image**: numpy array or a pillow image.

* **target_size**: Size to resize the input image to.

* **grayscale**: A boolean indicating whether to grayscale the image.

##### Returns
##### Example usage:
```python
```

### load\_image
```python
def load_image(image_file, target_size, grayscale, img_formats)
```
Load an image given its path. Returns an array version of optionally resized and grayed image. Only allows images of types described by img_formats argument.


##### Args
* **image_file**: Path to the image file.

* **target_size**: Size to resize the input image to.

* **grayscale**: A boolean indicating whether to grayscale the image.

* **img_formats**: List of allowed image formats that can be loaded.

##### Example usage:
```python
```

