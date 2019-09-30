# Evaluation of deduplication quality

To determine the quality of deduplication algorithm and the corresponding threshold, an evaluation framework is provided.

Given a ground truth mapping consisting of file names and a list of duplicates for each file along with a retrieved 
mapping from the deduplication algorithm for the same files, the following metrics can be obtained using the framework:

- [Mean Average Precision](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision) (MAP)
- [Mean Normalized Discounted Cumulative Gain](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) (NDCG)
- [Jaccard Index](https://en.wikipedia.org/wiki/Jaccard_index)
- Per class [Precision](https://en.wikipedia.org/wiki/Precision_and_recall) (class 0 = non-duplicate image pairs, class 1 = duplicate image pairs)
- Per class [Recall](https://en.wikipedia.org/wiki/Precision_and_recall) (class 0 = non-duplicate image pairs, class 1 = duplicate image pairs)
- Per class [f1-score](https://en.wikipedia.org/wiki/F1_score) (class 0 = non-duplicate image pairs, class 1 = duplicate image pairs)

The api for obtaining these metrics  is as below:
```python
from imagededup.evaluation import evaluate
metrics = evaluate(ground_truth_map, retrieved_map, metric='<metric-name>')
```
where the returned variable *metrics* is a dictionary containing the following content:
```
{
  'map': <map>,
  'ndcg': <mean ndcg>,
  'jaccard': <mean jaccard index>,
  'precision': <numpy array having per class precision>,
  'recall': <numpy array having per class recall>,
  'f1-score': <numpy array having per class f1-score>,
  'support': <numpy array having per class support>
}
```
#### Options
- ground_truth_map:  A dictionary representing ground truth with filenames as key and a list of duplicate filenames as 
value.
- retrieved_map: A dictionary representing retrieved duplicates with filenames as key and a list of retrieved duplicate 
filenames as value.
- metric: Can take one of the following values:
    - 'map'
    - 'ndcg'
    - 'jaccard'
    - 'classification': Returns per class precision, recall, f1-score, support
    - 'all' (default, returns all the above metrics)


#### Considerations
- Presently, the ground truth map should be prepared manually by the user. Symmetric relations between duplicates must 
be represented in the ground truth map. If an image *i* is a duplicate for image *j*, then *j* must also be represented as a
 duplicate of *i*. Absence of symmetric relations will lead to an exception.

- Both the ground_truth_map and retrieved_map must have the same keys.
