### avg\_prec
```python
def avg_prec(correct_duplicates, retrieved_duplicates)
```
Get average precision(AP) for a single query given correct and retrieved file names.


##### Args
* **correct_duplicates**: List of correct duplicates i.e., ground truth)

* **retrieved_duplicates**: List of retrieved duplicates for one single query

##### Returns
##### Example usage:
```python
```

### ndcg
```python
def ndcg(correct_duplicates, retrieved_duplicates)
```
Get Normalized discounted cumulative gain(NDCG) for a single query given correct and retrieved file names.


##### Args
* **correct_duplicates**: List of correct duplicates i.e., ground truth)

* **retrieved_duplicates**: List of retrieved duplicates for one single query

##### Returns
##### Example usage:
```python
```

### jaccard\_similarity
```python
def jaccard_similarity(correct_duplicates, retrieved_duplicates)
```
Get jaccard similarity for a single query given correct and retrieved file names.


##### Args
* **correct_duplicates**: List of correct duplicates i.e., ground truth)

* **retrieved_duplicates**: List of retrieved duplicates for one single query

##### Returns
##### Example usage:
```python
```

### mean\_metric
```python
def mean_metric(ground_truth, retrieved, metric)
```
Get mean of specified metric.


##### Args
* **metric_func**: metric function on which mean is to be calculated across all queries

##### Returns
##### Example usage:
```python
```

### get\_all\_metrics
```python
def get_all_metrics(ground_truth, retrieved)
```
Get mean of all information retrieval metrics across all queries.


##### Args
* **ground_truth**: A dictionary representing ground truth with filenames as key and a list of duplicate filenames

* **retrieved**: A dictionary representing retrieved duplicates with filenames as key and a list of retrieved

##### Returns
##### Example usage:
```python
```

