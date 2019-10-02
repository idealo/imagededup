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
be represented in the ground truth map. If an image *i* is a duplicate of image *j*, then *j* must also be represented 
as a duplicate of *i*. Absence of symmetric relations will lead to an exception.

- Both the ground_truth_map and retrieved_map must have the same keys.

- There is a difference between the way information retrieval metrics(map, ndcg, jaccard index) and classification 
metrics(precision, recall, f1-score) treat the symmetric relationships in duplicates. Consider the following 
ground_truth_map and retrieved_map:

ground_truth_map:
```
{
  '1.jpg': ['2.jpg', '4.jpg'],
  '2.jpg': ['1.jpg'],
  '3.jpg': [],
  '4.jpg': ['1.jpg']
}
```

retrieved_map:
```
{
  '1.jpg': ['2.jpg'],
  '2.jpg': ['1.jpg'],
  '3.jpg': [],
  '4.jpg': []
}
```
From the above, it can be seen that images *'1.jpg'* and *'4.jpg'* are not found to be duplicates of each other by the 
deduplication algorithm.

For calculating information retrieval metrics, each key in the maps is considered as an independent 'query'. 
In the ground truth, *'4.jpg'* is a duplicate of the key *'1.jpg'*. When it is not retrieved, it is considered a miss for 
query *'1.jpg'*.  Similarly, *'1.jpg'* is a duplicate of the key *'4.jpg'* in the ground truth. When this is not retrieved, 
it is considered a miss for query *'4.jpg'*.  Thus, the missing relationship is accounted for twice instead of just once.

Classification metrics, on the other hand, consider the relationships only once by forming unique pairs of images and 
labelling each pair as a 0 (non-duplicate image pair) and 1 (duplicate image pair). 

Using the ground_truth_map, the ground truth pairs with the corresponding labels are:

| Image Pair        | Label           
| ------------- |:-------------:
| ('1.jpg', '2.jpg')     | 1 
| ('1.jpg', '3.jpg')      | 0     
| ('1.jpg', '4.jpg') | 1
| ('2.jpg', '3.jpg')     | 0
| ('2.jpg', '4.jpg')      | 0      
| ('3.jpg', '4.jpg')| 0   


Similarly, using retrieved_map, the retrieved pairs are generated:

| Image Pair        | Label           
| ------------- |:-------------:
| ('1.jpg', '2.jpg')     | 1 
| ('1.jpg', '3.jpg')      | 0     
| ('1.jpg', '4.jpg') | 0
| ('2.jpg', '3.jpg')     | 0
| ('2.jpg', '4.jpg')      | 0      
| ('3.jpg', '4.jpg')| 0  

These two sets of pairs are then used to calculate metrics such as precision/recall/f1-score. It can be seen that the 
missing relationship between pair *('1jpg', '4.jpg')* is accounted for only once.