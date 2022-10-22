### classification\_metrics
```python
def classification_metrics(ground_truth, retrieved)
```
Given ground truth dictionary and retrieved dictionary, return per class precision, recall and f1 score. Class 1 is assigned to duplicate file pairs while class 0 is for non-duplicate file pairs.


##### Args
* **ground_truth**: A dictionary representing ground truth with filenames as key and a list of duplicate filenames

* **retrieved**: A dictionary representing retrieved duplicates with filenames as key and a list of retrieved

##### Returns

