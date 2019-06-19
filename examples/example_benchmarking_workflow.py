from imagededup.hashing import Hashing, HashedDataset
from imagededup.retrieval import ResultSet
from imagededup.evaluation import EvalPerformance
from pickle import load as pickle_loader
from datetime import datetime

"""
    SAMPLE OUTPUT
    -----------------------------------------------------------
    Stage 1: FINGEPRINTING completed in 183.915085 seconds
    Stage 2: SEARCH & RETRIEVAL completed in 253.477759 seconds
    Stage 3: EVALUATION completed in 1.1e-05 seconds

    MAP  0.21038649237472765
    NDCG  0.9794489685026253
    Jaccard  0.20671185711242465
    -----------------------------------------------------------
"""

QUERY_PATH = '/Users/zubin.john/forge/image-dedup/Transformed_dataset/Query/'
TEST_PATH = '/Users/zubin.john/forge/image-dedup/Transformed_dataset/Retrieval/'
GOLD_PATH = '/Users/zubin.john/forge/image-dedup/Transformed_dataset/ground_truth_transformed.pkl'

if __name__ == '__main__':
    hasher = Hashing()  # Instantiate a hashing function to be used for fingerprinting
    start = datetime.utcnow()
    ds = HashedDataset(hasher.dhash, QUERY_PATH, TEST_PATH)
    end = datetime.utcnow()

    print(f'Stage 1: FINGEPRINTING completed in {(end-start).total_seconds()} seconds')
    hashes = ds.get_test_hashes()
    queries = ds.get_query_hashes()
    doc_mapper = ds.get_docmap()
    start = datetime.utcnow()
    rs = ResultSet(hashes, queries, hasher.hamming_distance)
    end = datetime.utcnow()

    print(f'Stage 2: SEARCH & RETRIEVAL completed in {(end-start).total_seconds()} seconds')
    returned_dict = rs.retrieve_results()
    returned_dict = {doc_mapper[row]: returned_dict[row] for row in returned_dict}
    with open(GOLD_PATH, 'rb') as buf:
        correct_dict = pickle_loader(buf)
    start = datetime.utcnow()
    evaluator = EvalPerformance(correct_dict, returned_dict)
    end = datetime.utcnow()

    print(f'Stage 3: EVALUATION completed in {(end-start).total_seconds()} seconds\n\n')
    for metric, reading in evaluator.get_all_metrics().items():
        print(f'{metric}\t{reading}')
