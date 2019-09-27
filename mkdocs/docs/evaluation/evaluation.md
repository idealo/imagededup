### evaluate
```python
def evaluate(ground_truth_map, retrieved_map, metric)
```
Given a ground truth map and a duplicate map retrieved from a deduplication algorithm, get metrics to evaluate the effectiveness of the applied deduplication algorithm.


##### Args
* **ground_truth_map**: A dictionary representing ground truth with filenames as key and a list of duplicate filenames
                  as value.

* **retrieved_map**: A dictionary representing retrieved duplicates with filenames as key and a list of retrieved
               duplicate filenames as value.

* **metric**:  Name of metric to be evaluated and returned. Accepted values are: 'map', 'ndcg', 'jaccard',
        'classification', 'all' where 'all' returns every metric. 'map', 'ndcg' and 'jaccard' return a single
        number denoting the corresponding information retrieval metric.

##### Returns
* **dictionary**: A dictionary with metric name as key and corresponding calculated metric as the value.
            'classification' metrics include 'precision', 'recall' and 'f1-score' which are returned in the form
             of individual entries in the returned dictionary. The value for each of the classification metric
             is a numpy array with first entry as the score for non-duplicate file pairs(class-0) and second
             entry as the score for duplicate file pairs (class-1). Additionally, a support is also returned as
             another key with first entry denoting number of non-duplicate file pairs and second entry having
             duplicate file pairs.


