# Plotting duplicates of an image

Once a duplicate dictionary corresponding to an image directory has been obtained (using [find_duplicates](#find_duplicates)), duplicates 
for an image can be plotted using *plot_duplicates* method as below:
```python
from imagededup.utils import plot_duplicates
plot_duplicates(image_dir, filename)
OR
plot_duplicates(duplicate_map, filename)
```
where *filename* is the file for which duplicates are to be plotted.

#### Options
- *image_dir*: Directory where all image files are present.

- *duplicate_map*: A dictionary representing retrieved duplicates with filenames as key and a list of retrieved duplicate 
filenames as value. A duplicate_map with scores can also be passed (obtained from [find_duplicates](#find_duplicates)
function with scores attribute set to True).

- *filename*: Image file name for which duplicates are to be plotted.

- *outfile*: Name of the file the plot should be saved to. *None* by default.

The output looks as below:

![figs](../_readme_figures/plot_dups.png)