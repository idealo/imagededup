# CIFAR10 deduplication example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idealo/imagededup/blob/master/examples/CIFAR10_duplicates.ipynb)

### Install imagededup via PyPI
```
!pip install imagededup
```

### Download CIFAR10 dataset and untar
```
!wget http://pjreddie.com/media/files/cifar.tgz
!tar xzf cifar.tgz
```

### Create working directory and move all images into this directory
```
image_dir = 'cifar10_images'
!mkdir $image_dir
!cp -r '/content/cifar/train/.' $image_dir
!cp -r '/content/cifar/test/.' $image_dir
```

### Find duplicates in the entire dataset with CNN
```python
from imagededup.methods import CNN

cnn = CNN()
encodings = cnn.encode_images(image_dir=image_dir)
duplicates = cnn.find_duplicates(encoding_map=encodings)
```


### Do some imports for plotting
```python
from pathlib import Path
from imagededup.utils import plot_duplicates
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 10)
```

### Find and plot duplicates in the test set with CNN
```python
# test images are stored under '/content/cifar/test'
filenames_test = set([i.name for i in Path('/content/cifar/test').glob('*.png')])

duplicates_test = {}
for k, v in duplicates.items():
  if k in filenames_test:
    tmp = [i for i in v if i in filenames_test]
    duplicates_test[k] = tmp
    
# sort in descending order of duplicates
duplicates_test = {k: v for k, v in sorted(duplicates_test.items(), key=lambda x: len(x[1]), reverse=True)}

# plot duplicates found for some file
plot_duplicates(image_dir=image_dir, duplicate_map=duplicates_test, filename=list(duplicates_test.keys())[0])
```

### Find and plot duplicates in the train set with CNN
```python
# train images are stored under '/content/cifar/train'
filenames_train = set([i.name for i in Path('/content/cifar/train').glob('*.png')])

duplicates_train = {}
for k, v in duplicates.items():
  if k in filenames_train:
    tmp = [i for i in v if i in filenames_train]
    duplicates_train[k] = tmp
    

# sort in descending order of duplicates
duplicates_train = {k: v for k, v in sorted(duplicates_train.items(), key=lambda x: len(x[1]), reverse=True)}

# plot duplicates found for some file
plot_duplicates(image_dir=image_dir, duplicate_map=duplicates_train, filename=list(duplicates_train.keys())[0])
```

### Examples from test set with duplicates in train set
```python
# keep only filenames that are in test set have duplicates in train set
duplicates_test_train = {}
for k, v in duplicates.items():
    if k in filenames_test:
        tmp = [i for i in v if i in filenames_train]
        duplicates_test_train[k] = tmp
    
# sort in descending order of duplicates
duplicates_test_train = {k: v for k, v in sorted(duplicates_test_train.items(), key=lambda x: len(x[1]), reverse=True)}

# plot duplicates found for some file
plot_duplicates(image_dir=image_dir, duplicate_map=duplicates_test_train, filename=list(duplicates_test_train.keys())[0])
```