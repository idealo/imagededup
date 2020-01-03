# Benchmarks

To gauge an idea of the speed and accuracy of the implemented algorithms, a benchmark
has been provided on the [UKBench dataset](https://archive.org/details/ukbench) (zip file titled 'UKBench image collection'
having size ~1.5G) and some variations derived from it.


## Datasets

3 datasets that have been used:

1. Near duplicate dataset ([UKBench dataset](https://archive.org/details/ukbench)): This dataset has near duplicates
 that are arranged in groups of 4. There are a total of 2550 such groups amounting to a total
  of 10200 RGB images. The size of each image is 640 x 480 with `jpg` extension. The image below depicts 3 example groups
  from the UKBench dataset. Each row represents a group with the corresponding 4 images from the group.

    <p align="center">
      <img src="../../img/collage_ukbench.png" width="600" />
    </p>  

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

## Environment
The benchmarks were performed on an AWS ec2 [r5.xlarge](https://aws.amazon.com/ec2/instance-types/r5/) instance having 4
 vCPUs and 32 GB memory. The instance does not have a GPU, so all the runs are done on CPUs.

## Metrics
The metrics used here are classification metrics as explained in the
[documentation](https://idealo.github.io/imagededup/user_guide/evaluating_performance/).

class-0 refers to non-duplicate image pairs.

class-1 refers to duplicate image pairs.

The reported numbers are rounded off to nearest 3 digits.

## Timings
The times are reported in seconds and comprise the time taken to generate encodings and find duplicates.
The time taken to perform the evaluation task is *NOT* reported.

## Threshold selection
For each method, 3 different thresholds have been selected.

For hashing methods, following `max_distance_threshold` values are used:

- 0: Indicates that exactly the same hash should be generated for the image pairs to be considered duplicates.
- 10: Default.
- 32: Halfway between the maximum and minimum values (0 and 64).

For cnn method, following `min_similarity_threshold` values are used:

- 1.0: Indicates that exactly the same cnn embeddings should be generated for the image pairs to be considered duplicates.
- 0.9: Default.
- 0.5: A threshold that allows large deviation between image pairs.

## Results
### Near Duplicate dataset
| *Method* | *Threshold* | *Time (s)* | *class-0 precision* |*class-1 precision*  |  *class-0 recall*| *class-1 recall*|
|--------|-----------|----------|-------------------|-------------------|----------------|----------------|
| dhash  | 0         | 35.570    | 0.999        | 0.0                 | 1.0              | 0.0              |
| dhash  | 10        | 35.810    | 0.999        | 0.018        | 0.999      | 0.0461     |
| dhash  | 32        | 106.670   | 0.998             | 0.0           | 0.326     | 0.884     |
| phash  | 0         | 40.073   | 0.999        | 1.0                 | 1.0              | 0.0        |
| phash  | 10        | 39.056   | 0.999         | 0.498           | 0.999     | 0.016     |
| phash  | 32        | 98.835   | 0.998             | 0.0            | 0.343      | 0.856        |
| ahash  | 0         | 36.171   | 0.999         | 0.282            | 0.999         | 0.002         |
| ahash  | 10        | 36.560    | 0.999         | 0.012            | 0.996         | 0.193         |
| ahash  | 32        | 97.170    | 0.999             | 0.000            | 0.448       | 0.932        |
| whash  | 0         | 51.710    | 0.999          | 0.112            | 0.999         | 0.002         |
| whash  | 10        | 51.940    | 0.999           | 0.008           | 0.993        | 0.199         |
| whash  | 32        | 112.560   | 0.999             | 0.0            | 0.416        | 0.933        |
| cnn    | 0.5       | 379.680   | 0.999          | 0.0             | 0.856         | 0.999         |
| cnn    | 0.9       | 377.157  | 0.999           | 0.995           | 0.999         | 0.127         |
| cnn    | 1.0       | 379.570   | 0.999           | 0.0                | 1.0            | 0.0            |

#### Observations
- The cnn method with a threshold between 0.5 and 0.9 would work best for finding near duplicates. This is indicated by
 the extreme values class-1 precision and recall takes for the two thresholds.
- Hashing methods do not perform well for finding near duplicates.

### Transformed dataset
| *Method* | *Threshold* | *Time (s)* | *class-0 precision* | *class-1 precision* | *class-0 recall* | *class-1 recall* |
|--------|-----------|----------|-------------------|-------------------|----------------|----------------|
| dhash  | 0         | 25.360    | 0.999           | 1.0                 | 1.0              | 0.040           |
| dhash  | 10        | 25.309   | 0.999            | 0.138           | 0.999        | 0.117         |
| dhash  | 32        | 108.960   | 0.990              | 0.0                 | 0.336         | 0.872         |
| phash  | 0         | 28.069   | 0.999         | 1.0                 | 1.0              | 0.050           |
| phash  | 10        | 28.075   | 0.999            | 0.341             | 0.999         | 0.079       |
| phash  | 32        | 107.079  | 0.990              | 0.003             | 0.328        | 0.847        |
| ahash  | 0         | 25.270    | 0.999           | 0.961             | 0.999        | 0.058     |
| ahash  | 10        | 25.389   | 0.999        | 0.035           | 0.997        | 0.216       |
| ahash  | 32        | 93.084   | 0.990              | 0.0                 | 0.441         | 0.849         |
| whash  | 0         | 40.390    | 0.999        | 0.917           | 0.999         | 0.061        |
| whash  | 10        | 41.260    | 0.999           | 0.023           | 0.996       | 0.203         |
| whash  | 32        | 109.630   | 0.990              | 0.0                 | 0.410         | 0.853          |
| cnn    | 0.5       | 397.380   | 0.999            | 0.003           | 0.852         | 0.999        |
| cnn    | 0.9       | 392.090   | 0.999           | 0.999            | 0.990           | 0.384        |
| cnn    | 1.0       | 396.250   | 0.990              | 0.0                 | 1.0              | 0.0              |

#### Observations
- The cnn method with threshold 0.9 seems to work best for finding transformed duplicates. A slightly lower
`min_similarity_threshold` value could lead to a higher class-1 recall.
- Hashing methods do not perform well for finding transformed duplicates. In reality, resized images get found easily,
but all other transformations lead to a bad performance for hashing methods.

### Exact duplicates dataset

| *Method* | *Threshold* | *Time (s)* | *class-0 precision* | *class-1 precision* | *class-0 recall* | *class-1 recall* |
|--------|-----------|----------|-------------------|-------------------|----------------|----------------|
| dhash  | 0         | 18.380    | 1.0                 | 1.0                 | 1.0              | 1.0              |
| dhash  | 10        | 18.410    | 1.0                 | 0.223            | 0.999         | 1.0              |
| dhash  | 32        | 34.602   | 1.0                 | 0.0                 | 0.327          | 1.0              |
| phash  | 0         | 19.780    | 1.0                 | 1.0                 | 1.0              | 1.0              |
| phash  | 10        | 20.012   | 1.0                 | 0.980              | 0.999          | 1.0              |
| phash  | 32        | 34.054   | 1.0                 | 0.0                 | 0.344          | 1.0              |
| ahash  | 0         | 18.180    | 1.0                 | 0.998             | 0.999          | 1.0              |
| ahash  | 10        | 18.228   | 1.0                 | 0.044            | 0.995          | 1.0              |
| ahash  | 32        | 31.961   | 1.0                 | 0.0            | 0.448          | 1.0              |
| whash  | 0         | 26.097   | 1.0                 | 0.980              | 0.999          | 1.0              |
| whash  | 10        | 26.056   | 1.0                 | 0.029             | 0.993          | 1.0              |
| whash  | 32        | 39.408   | 1.0                 | 0.0                 | 0.417         | 1.0              |
| cnn    | 0.5       | 192.050   | 1.0                 | 0.001            | 0.860           | 1.0              |
| cnn    | 0.9       | 191.024  | 1.0                 | 1.0                 | 1.0              | 1.0              |
| cnn    | 1.0       | 194.270   | 0.999          | 1.0                 | 1.0              | 0.580\*           |

\* The value is low as opposed to the expected 1.0 because of the [`cosine_similarity`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)
 function from scikit-learn (used within the package) which sometimes calculates the similarity to be slightly less than 1.0 even when the vectors are same.

#### Observations
- Difference hashing is the fastest (`max_distance_threshold` 0).
- When using hashing methods for exact duplicates, keep `max_distance_threshold` to a low value. The value of 0 is
 good, but a slightly higher value should also work fine.
- When using cnn method, keep `min_similarity_threshold` to a high value. The default value of 0.9 seems to work well.
A slightly higher value can also be used.


## Summary
- Near duplicate dataset: use cnn with an appropriate `min_similarity_threshold`.
- Transformed dataset: use cnn with `min_similarity_threshold` of around 0.9 (default).
- Exact duplicates dataset: use Difference hashing with 0 `max_distance_threshold`.
- A higher `max_distance_threshold` (i.e., hashing) leads to a higher execution time. cnn method doesn't seem much affected by
 the `min_similarity_threshold` (though a lower value would add a few seconds to the execution time as can be seen in all the
 runs above.)
 - Generally speaking, the cnn method takes longer to run as compared to hashing methods for all datasets. If a GPU is
available, cnn method should be much faster.
