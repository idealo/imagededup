# Cats and Dogs Example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idealo/imageatm/blob/master/examples/imageatm_cats_and_dogs.ipynb)

### Install imageatm via PyPi
```python
pip install imageatm
```

### Download the cats and dogs dataset
```bash
wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
    -O cats_and_dogs_filtered.zip
```

### Unzip dataset and create working directory
```bash
unzip cats_and_dogs_filtered.zip
mkdir -p cats_and_dogs/train
mv cats_and_dogs_filtered/train/cats/* cats_and_dogs/train
mv cats_and_dogs_filtered/train/dogs/* cats_and_dogs/train
```

### Create the sample file
```python
import os
import json

filenames = os.listdir('cats_and_dogs/train')
sample_json = []
for i in filenames:
    sample_json.append(
        {
        'image_id': i,
        'label': 'Cat' if 'cat' in i else 'Dog'
        }
        )

with open('data.json', 'w') as outfile:
    json.dump(sample_json, outfile, indent=4, sort_keys=True)
```

### Run the data preparation with resizing
```python
from imageatm.components import DataPrep

dp = DataPrep(
    image_dir='cats_and_dogs/train',
    samples_file='data.json',
    job_dir='cats_and_dogs'
)

dp.run(resize=True)
```

### Initialize the Training class and run it
```python
from imageatm.components import Training

trainer = Training(dp.image_dir, dp.job_dir, epochs_train_dense=5, epochs_train_all=5)

trainer.run()
```

### Evaluate the best model
```python
from imageatm.components import Evaluation

e = Evaluation(image_dir=dp.image_dir, job_dir=dp.job_dir)

e.run()
```

### Visualize CAM analysis on the correct and wrong examples
```python
c, w = e.get_correct_wrong_examples(label=1)

e.visualize_images(w, show_heatmap=True)

e.visualize_images(c, show_heatmap=True)
```