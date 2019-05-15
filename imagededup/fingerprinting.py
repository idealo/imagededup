from types import FunctionType
import os
import random
from PIL import Image
from copy import deepcopy
from numpy import array

class Dataset:
    """
    Class wrapper to instantiate a Dataset object composing of a subset of test images
    and a smaller fraction of images that are used as queries to test search and retrieval.
    Object contains hashed image fingerprints as well, however, hashing method is set by user.
    """
    def __init__(self, path_to_queries: str, path_to_test: str) -> None:
        self.query_docs = self.load_image_arrays(path_to_queries)
        self.test_docs = self.load_image_arrays(path_to_test)


    @staticmethod
    def load_image_arrays(path: str) -> dict: 
        _docs = {doc: os.path.join(path, doc) for doc in os.listdir(path) if doc.endswith('.jpg')}
        return {doc: array(Image.open(_docs[doc])) for doc in _docs} # Save each image as a Numpy Array


class HashedDataset(Dataset):
    def __init__(self, hashing_function: FunctionType, *args, **kwargs) -> None:
        super(HashedDataset, self).__init__(*args, **kwargs)
        self.hasher = hashing_function
        self.fingerprint()
        self.doc2hash = deepcopy(self.test_hashes)
        self.doc2hash.update(self.query_hashes)
        self.hash2doc = {self.doc2hash[doc]: doc for doc in self.doc2hash}

    
    def fingerprint(self) -> None:
        self.test_hashes = {doc: str(self.hasher(self.test_docs[doc])) for doc in self.test_docs}
        self.query_hashes = {doc: str(self.hasher(self.query_docs[doc])) for doc in self.query_docs}
