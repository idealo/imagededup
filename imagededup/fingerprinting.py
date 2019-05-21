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
        #print(path_to_queries, path_to_test)
        self.query_docs = self.load_image_set(path_to_queries)
        self.test_docs = self.load_image_set(path_to_test)

    @staticmethod
    def load_image_set(path: str) -> dict: 
        return {doc: os.path.join(path, doc) for doc in os.listdir(path) if doc.endswith('.jpg')}


class HashedDataset(Dataset):
    def __init__(self, hashing_function: FunctionType, *args, **kwargs) -> None:
        super(HashedDataset, self).__init__(*args, **kwargs)
        self.hasher = hashing_function
        # self.test_hashes = {doc: str(self.hasher(Image.open(self.test_docs[doc]))) for doc in self.test_docs}
        # self.query_hashes = {doc: str(self.hasher(Image.open(self.query_docs[doc]))) for doc in self.query_docs}
        self.fingerprint()
        self.doc2hash = deepcopy(self.test_hashes)
        #self.doc2hash.update(self.query_hashes)
        self.hash2doc = {self.doc2hash[doc]: doc for doc in self.doc2hash}

    
    def fingerprint(self) -> None:
        self.test_hashes = {doc: str(self.hasher(Image.open(self.test_docs[doc]))) for doc in self.test_docs}
        self.query_hashes = {doc: str(self.hasher(Image.open(self.query_docs[doc]))) for doc in self.query_docs}
        
        
    def get_hashes(self) -> dict:
        return self.doc2hash


    def get_query_hashes(self) -> dict:
        return self.query_hashes


    def get_test_hashes(self) -> dict:
        return self.test_hashes