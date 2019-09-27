### plot\_duplicates
```python
def plot_duplicates(image_dir, duplicate_map, filename, outfile)
```
Given filename for an image, plot duplicates along with the original image using the duplicate map obtained using find_duplicates method.


##### Args
* **image_dir**: image directory where all files in duplicate_map are present.

* **duplicate_map**: mapping of filename to found duplicates (could be with or without scores).

* **filename**: Name of the file for which duplicates are to be plotted, must be a key in the duplicate_map

* **outfile**: Name of the file to save the plot.


