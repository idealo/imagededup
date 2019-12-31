# Benchmarks

To gauge an idea of the speed and accuracy of the implemented algorithms, a benchmark
has been provided on the [UKBench dataset](https://archive.org/details/ukbench) and 
some variations derived from it.


## Datasets

3 datasets that have been used:

1. Near duplicate dataset ([UKBench dataset](https://archive.org/details/ukbench)): This dataset has near-duplicates
 that are arranged in groups of 4. There are a total of 2550 such groups amounting to a total
  of 10200 RGB images. The size of each image is 640 x 480 with `jpg` extension.

2. Transformed dataset derived from UKBench dataset: An image from different groups of the UKBench
 dataset was taken and the following 5 transformations were applied to the original image:
 
    - Random crop preserving the original aspect ratio (new size - 560 x 420)
    - Horizontal flip
    - Vertical flip
    - 25 degree rotation
    - Resizing with change in aspect ratio (new aspect ratio - 1:1)
    
    Thus, each group has a total of 6 images (original +  transformed). A total of 1800 such groups
    were created totalling 10800 images in the dataset.

3. Exact duplicate dataset: An image from each of the 2550 image groups of the UKBench dataset was
 taken and an exact duplicate was created. The number of images totalled 5100.
 
## Metrics
The metrics used here are classification metrics as explained in the 
[documentation](https://idealo.github.io/imagededup/user_guide/evaluating_performance/).

class-0 refers to non-duplicate image pairs.
class-1 refers to duplicate image pairs.


## Timings
The times are reported in seconds and comprise the time taken to generate encodings and find duplicates.
The time taken to perform the evaluation task is *NOT* reported.

## Results
### Near Duplicate dataset
| *Method* | *Threshold* | *Time (s)* | *class-0 precision* |*class-1 precision*  |  *class-0 recall*| *class-1 recall*|
|--------|-----------|----------|-------------------|-------------------|----------------|----------------|
| dhash  | 0         | 35.57    | 0.99970585        | 0                 | 1              | 0              |
| dhash  | 10        | 35.81    | 0.99971922        | 0.01867083        | 0.9992864      | 0.04614379     |
| dhash  | 32        | 106.67   | 0.998             | 0.00038           | 0.32576433     | 0.88359477     |
| phash  | 0         | 40.073   | 0.99970589        | 1                 | 1              | 0.00013        |
| phash  | 10        | 39.056   | 0.9997105         | 0.49896           | 0.99999533     | 0.01581699     |
| phash  | 32        | 98.835   | 0.998             | 0.0008            | 0.3434116      | 0.85588        |
| ahash  | 0         | 36.171   | 0.999706          | 0.2828            | 0.9999         | 0.0018         |
| ahash  | 10        | 36.56    | 0.9997614         | 0.0127            | 0.9956         | 0.1926         |
| ahash  | 32        | 97.17    | 0.999             | 0.0004            | 0.447443       | 0.93163        |
| whash  | 0         | 51.71    | 0.99970           | 0.1117            | 0.9999         | 0.0025         |
| whash  | 10        | 51.94    | 0.99976           | 0.00868           | 0.99334        | 0.1981         |
| whash  | 32        | 112.56   | 0.999             | 0.0004            | 0.41623        | 0.93300        |
| cnn    | 0.5       | 379.68   | 0.999999          | 0.002             | 0.8566         | 0.9995         |
| cnn    | 0.9       | 377.157  | 0.99974           | 0.99488           | 0.9999         | 0.1271         |
| cnn    | 1.0       | 379.57   | 0.99970           | 0.                | 1.0            | 0.0            |


### Transformed dataset
| *Method* | *Threshold* | *Time (s)* | *class-0 precision* | *class-1 precision* | *class-0 recall* | *class-1 recall* |
|--------|-----------|----------|-------------------|-------------------|----------------|----------------|
| dhash  | 0         | 25.36    | 0.99955           | 1                 | 1              | 0.04           |
| dhash  | 10        | 25.309   | 0.9995            | 0.13760           | 0.99965        | 0.1174         |
| dhash  | 32        | 108.96   | 0.99              | 0                 | 0.3362         | 0.8721         |
| phash  | 0         | 28.069   | 0.9995613         | 1                 | 1              | 0.05           |
| phash  | 10        | 28.075   | 0.9995            | 0.341             | 0.9999         | 0.079074       |
| phash  | 32        | 107.079  | 0.99              | 0.003             | 0.32814        | 0.84762        |
| ahash  | 0         | 25.27    | 0.99956           | 0.961             | 0.99999        | 0.05777778     |
| ahash  | 10        | 25.389   | 0.99963605        | 0.03504           | 0.99724        | 0.216185       |
| ahash  | 32        | 93.084   | 0.99              | 0                 | 0.4405         | 0.8485         |
| whash  | 0         | 40.39    | 0.99956528        | 0.91717           | 0.9999         | 0.06111        |
| whash  | 10        | 41.26    | 0.99962           | 0.02319           | 0.996032       | 0.2034         |
| whash  | 32        | 109.63   | 0.99              | 0                 | 0.4102         | 0.853          |
| cnn    | 0.5       | 397.38   | 0.9999            | 0.00312           | 0.8523         | 0.99996        |
| cnn    | 0.9       | 392.09   | 0.99971           | 0.9997            | 0.99           | 0.38392        |
| cnn    | 1.0       | 396.25   | 0.99              | 0                 | 1              | 0              |


### Exact duplicates dataset

| *Method* | *Threshold* | *Time (s)* | *class-0 precision* | *class-1 precision* | *class-0 recall* | *class-1 recall* |
|--------|-----------|----------|-------------------|-------------------|----------------|----------------|
| dhash  | 0         | 18.38    | 1                 | 1                 | 1              | 1              |
| dhash  | 10        | 18.41    | 1                 | 0.2233            | 0.9998         | 1              |
| dhash  | 32        | 34.602   | 1                 | 0                 | 0.327          | 1              |
| phash  | 0         | 19.78    | 1                 | 1                 | 1              | 1              |
| phash  | 10        | 20.012   | 1                 | 0.98              | 0.999          | 1              |
| phash  | 32        | 34.054   | 1                 | 0                 | 0.344          | 1              |
| ahash  | 0         | 18.18    | 1                 | 0.998             | 0.999          | 1              |
| ahash  | 10        | 18.228   | 1                 | 0.0440            | 0.995          | 1              |
| ahash  | 32        | 31.961   | 1                 | 0.0003            | 0.448          | 1              |
| whash  | 0         | 26.097   | 1                 | 0.98              | 0.999          | 1              |
| whash  | 10        | 26.056   | 1                 | 0.029             | 0.993          | 1              |
| whash  | 32        | 39.408   | 1                 | 0                 | 0.4167         | 1              |
| cnn    | 0.5       | 192.05   | 1                 | 0.0014            | 0.86           | 1              |
| cnn    | 0.9       | 191.024  | 1                 | 1                 | 1              | 1              |
| cnn    | 1.0       | 194.27   | 0.999917          | 1                 | 1              | 0.58           |