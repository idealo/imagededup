from imagededup.hashing import Hashing
from imagededup.cnn import CNN
from imagededup.retrieve.retrieval import HashEval
from imagededup.retrieve.retrieval import CosEval
from imagededup.evaluate.performance import Metrics
from pathlib import Path


class Evaluate:
    def __init__(self, query_dir_dict=None, test_dir_dict=None, ground_truth_dict=None, method: str = None,
                 method_args={'hash_method': None, 'hashing_threshold': 0, 'cosine_threshold': 0.8},
                 save_filename: str = None) -> None:
        self.query_dir_dict = query_dir_dict
        self.test_dir_dict = test_dir_dict
        self.method_args = method_args
        self.save_filename = save_filename
        self.correct_dict = ground_truth_dict

        if method == 'hashing':
            self._hashing_performance()
        elif method == 'cnn':
            self._cnn_performance()
        else:
            raise ValueError('Valid methods are: \'hashing\' and \'cnn\'')

    def _get_features(self, feature_generator):
        if isinstance(self.query_dir_dict, dict) and isinstance(self.test_dir_dict, dict):
            dict_query = self.query_dir_dict
            dict_test = self.test_dir_dict
        elif isinstance(self.query_dir_dict, Path) and isinstance(self.test_dir_dict, Path):
            dict_query = feature_generator(self.query_dir)
            dict_test = feature_generator(self.test_dir)
        else:
            raise ValueError('Query and Tests inputs should be either both path to respective directories or feature '
                             'dictionaries!')
        return dict_query, dict_test

    def _hashing_performance(self) -> None:
        hash_obj = Hashing(method=self.method_args['hash_method'])
        dict_query, dict_test = self._get_features(hash_obj.hash_dir)
        rs = HashEval(test=dict_test, queries=dict_query, hammer=hash_obj.hamming_distance,
                      cutoff=self.method_args['hashing_threshold'])
        returned_dict = rs.retrieve_result_list()
        metrics = Metrics(self.correct_dict, returned_dict)

        for metric, reading in metrics.get_all_metrics().items():  # saves by default a file named all_average_metrics.pkl
            print(f'{metric}\t{reading}')

    def _cnn_performance(self) -> None:
        cnn_obj = CNN()
        dict_query, dict_test = self._get_features(cnn_obj.cnn_dir)
        feat_vec_query, filemapping_generated_query = cnn_obj._get_file_mapping_feat_vec(dict_query)
        feat_vec_test, filemapping_generated_test = cnn_obj._get_file_mapping_feat_vec(dict_test)

        result_set = CosEval(feat_vec_query, feat_vec_test). \
            get_retrievals_at_thresh(file_mapping_query=filemapping_generated_query,
                                     file_mapping_ret=filemapping_generated_test,
                                     thresh=self.method_args['cosine_threshold'])

        returned_dict = cnn_obj._get_only_filenames(result_set)
        metrics = Metrics(self.correct_dict, returned_dict)
        all_metrics = metrics.get_all_metrics(self.save_filename)
        for metric, reading in all_metrics.items():
            print(f'{metric}\t{reading}')

