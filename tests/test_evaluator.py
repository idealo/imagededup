from imagededup.evaluation.evaluation import Evaluate
from pathlib import Path
import pickle

QUERY_PATH = Path('/Users/tanuj.jain/Documents/dedup-data/Transformed_dataset/Query')
TEST_PATH = Path('/Users/tanuj.jain/Documents/dedup-data/Transformed_dataset/Retrieval')
GOLD_PATH = Path('/Users/tanuj.jain/Documents/dedup-data/Transformed_dataset/ground_truth_transformed.pkl')
with open(GOLD_PATH, 'rb') as f:
    ground_truth = pickle.load(f)

def initializer():

    pass

def test_correct_initialization():
    myEval = Evaluate(query_dir_dict=QUERY_PATH, test_dir_dict=TEST_PATH, ground_truth_dict=ground_truth,
                      method='hashing', method_args=hash_method_args, save_filename=None)
    pass


def test_incorrect_initialization_method_throws_valueerror():
    pass


def test__get_features_throws_valueerror():
    pass


def test__show_metrics_args():
    pass

def test_

