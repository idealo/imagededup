### evaluate
```python
def evaluate(ground_truth_map, retrieved_map, metric)
```
Given a ground truth map and a duplicate map retrieved from a deduplication algorithm, get metrics to evaluate the effectiveness of the applied deduplication algorithm.


##### Args
* **ground_truth_map**: A dictionary representing ground truth with filenames as key and a list of duplicate filenames

* **retrieved_map**: A dictionary representing retrieved duplicates with filenames as key and a list of retrieved

* **metric**:  Name of metric to be evaluated and returned. Accepted values are: 'map', 'ndcg', 'jaccard',

##### Returns
* **dictionary**: A dictionary with metric name as key and corresponding calculated metric as the value.


