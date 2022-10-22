### cosine\_similarity\_chunk
```python
def cosine_similarity_chunk(t)
```

### get\_cosine\_similarity
```python
def get_cosine_similarity(X, verbose, chunk_size, threshold)
```

## class HashEval
### \_\_init\_\_
```python
def __init__(test, queries, distance_function, verbose, threshold, search_method)
```
Initialize a HashEval object which offers an interface to control hashing and search methods for desired dataset. Compute a map of duplicate images in the document space given certain input control parameters.



### retrieve\_results
```python
def retrieve_results(scores)
```
Return results with or without scores.


##### Args
* **scores**: Boolean indicating whether results are to eb returned with or without scores.

##### Returns
* **if scores is True, then a dictionary of the form {'image1.jpg'**: [('image1_duplicate1.jpg',

* **score), ('image1_duplicate2.jpg', score)], 'image2.jpg'**: [] ..}

* **if scores is False, then a dictionary of the form {'image1.jpg'**: ['image1_duplicate1.jpg',

* **'image1_duplicate2.jpg'], 'image2.jpg'**: ['image1_duplicate1.jpg',..], ..}


