## class DataGenerator
Class inherits from Keras Sequence base object, allows to use multiprocessing in .fit_generator.


##### Attributes
* **image_dir**: Path of image directory.

* **batch_size**: Number of images per batch.

* **basenet_preprocess**: Basenet specific preprocessing function.

* **target_size**: Dimensions that images get resized into when loaded.

### \_\_init\_\_
```python
def __init__(image_dir, batch_size, basenet_preprocess, target_size)
```
Init DataGenerator object.



### on\_epoch\_end
```python
def on_epoch_end()
```
Method called at the end of every epoch.



### \_\_len\_\_
```python
def __len__()
```
Number of batches in the Sequence.



### \_\_getitem\_\_
```python
def __getitem__(index)
```
Get batch at position `index`.



