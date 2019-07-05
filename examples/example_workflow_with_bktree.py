from imagededup.hashing import Hashing, HashedDataset
from imagededup.retrieve.retrieval import HashEval
from imagededup.evaluation import EvalPerformance
from pickle import load as pickle_loader
from datetime import datetime

QUERY_PATH = '/Users/tanuj.jain/Documents/dedup-data/Transformed_dataset/Query'
TEST_PATH = '/Users/tanuj.jain/Documents/dedup-data/Transformed_dataset/Retrieval'
GOLD_PATH = '/Users/tanuj.jain/Documents/dedup-data/Transformed_dataset/ground_truth_transformed.pkl'

if __name__ == '__main__':
    hasher = Hashing()
    start = datetime.utcnow()
    ds = HashedDataset(hasher.phash, QUERY_PATH, TEST_PATH)
    end = datetime.utcnow()
    print(f'Stage 1: FINGEPRINTING completed in {(end - start).total_seconds()} seconds')

    hashes = ds.get_test_hashes()
    queries = ds.get_query_hashes()
    doc_mapper = ds.get_docmap()
    start = datetime.utcnow()
    rs = HashEval(hashes, queries, hasher.hamming_distance)
    end = datetime.utcnow()

    print(f'Stage 2: SEARCH & RETRIEVAL completed in {(end - start).total_seconds()} seconds')
    returned_dict = rs.retrieve_results()
    returned_dict = {doc_mapper[row]: returned_dict[row] for row in returned_dict}
    with open(GOLD_PATH, 'rb') as buf:
        correct_dict = pickle_loader(buf)
    start = datetime.utcnow()
    evaluator = EvalPerformance(correct_dict, returned_dict)
    end = datetime.utcnow()

    print(f'Stage 3: EVALUATION completed in {(end - start).total_seconds()} seconds\n\n')
    for metric, reading in evaluator.get_all_metrics().items():
        print(f'{metric}\t{reading}')

