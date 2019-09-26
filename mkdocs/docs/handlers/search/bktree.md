## class BkTreeNode
Class to contain the attributes of a single node in the BKTree.


### \_\_init\_\_
```python
def __init__(node_name, node_value, parent_name)
```

## class BKTree
Class to construct and perform search using a BKTree.


### \_\_init\_\_
```python
def __init__(hash_dict, distance_function)
```
Initialize a root for the BKTree and triggers the tree construction using the dictionary for mapping file names and corresponding hashes.


##### Args
* **hash_dict**:  Dictionary mapping file names to corresponding hash strings {filename: hash}

* **distance_function**: A function for calculating distance between the hashes.


### construct\_tree
```python
def construct_tree()
```
Construct the BKTree.



### search
```python
def search(query, tol)
```
Function to search the bktree given a hash of the query image.


##### Args
* **query**: hash string for which BKTree needs to be searched.

* **tol**: distance upto which duplicate is valid.

##### Returns
* **List of tuples of the form [(valid_retrieval_filename1**:  distance), (valid_retrieval_filename2: distance)]


