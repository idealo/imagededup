import sys
sys.path.append('/Users/zubin.john/forge/image-dedup/')
from imagededup.fingerprinting import HashedDataset
from imagededup.hashing import Hashing
from imagededup.retrieval import ResultSet
from imagededup.evaluation import EvalPerformance
from pickle import load as pickle_loader

if __name__ == '__main__':
    d_hasher = Hashing().dhash ## Instantiate a hashing function to be used for fingerprinting
    # Instantiate a HashedDataset object for each method that needs benchmarking
    
    print('Stage: Dataset fingerprinting ')
    
    dobj = HashedDataset(
        d_hasher,
        '/Users/zubin.john/forge/image-dedup/Transformed_dataset/Query/',
        '/Users/zubin.john/forge/image-dedup/Transformed_dataset/Retrieval/'
    )

    hashes = dobj.get_hashes()
    queries = dobj.get_query_hashes()

    result = ResultSet('imageset', hashes, queries).retrieve_results()
    with open('/Users/zubin.john/forge/image-dedup/Transformed_dataset/ground_truth_transformed.pkl', 'rb') as rb:
        correct_dict = pickle_loader(rb)
    
    # Return results
    print(f'Metrics:\n{EvalPerformance(correct_dict, result).get_all_metrics()}')