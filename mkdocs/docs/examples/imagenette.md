# Imagenette Example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idealo/imageatm/blob/master/examples/imageatm_imagenette.ipynb)

### Install imageatm via PyPi
```python
pip install imageatm
```

### Download the [Imagenette dataset (320px)](https://github.com/fastai/imagenette) and ImageNet mapping
```bash
wget --no-check-certificate \
    https://s3.amazonaws.com/fast-ai-imageclas/imagenette-320.tgz
    
wget --no-check-certificate \
  https://raw.githubusercontent.com/ozendelait/wordnet-to-json/master/mapping_imagenet.json
```

### Untar the dataset
```bash
tar -xzf imagenette-320.tgz
```

### Create mapping for Imagenette classes and prepare the data.json
```python
import os
import json

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

mapping = load_json('mapping_imagenet.json')

mapping_synset_txt = {}
for i, j in enumerate(mapping):
  mapping_synset_txt[j['v3p0']] = j['label'].split(',')[0]
  
classes = os.listdir('imagenette-320/train')
sample_json = []
for c in classes:
  filenames = os.listdir('imagenette-320/train/{}'.format(c))
  for i in filenames:
      sample_json.append(
          {
          'image_id': i,
          'label': mapping_synset_txt[c]
          }
          )
          
with open('data.json', 'w') as outfile:
    json.dump(sample_json, outfile, indent=4, sort_keys=True)
```

### Prepare our image directory
```python
IMAGE_DIR ='images'

if not os.path.exists(IMAGE_DIR):
  os.makedirs(IMAGE_DIR)

classes = os.listdir('imagenette-320/train')
for c in classes:
  cmd = 'cp -r {}. {}'.format(os.path.join('imagenette-320/train', c) + '/', os.path.join(IMAGE_DIR))
  os.system(cmd)
```

### Run the data preparation
```python
from imageatm.components import DataPrep

dp = DataPrep(
    image_dir = 'images',
    samples_file = 'data.json',
    job_dir = 'imagenette'
)

dp.run(resize=False)
```

### Initialize the Training class and run it
```python
from imageatm.components import Training

trainer = Training(
     dp.image_dir, dp.job_dir, epochs_train_dense=5, epochs_train_all=5, batch_size=64,
)

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