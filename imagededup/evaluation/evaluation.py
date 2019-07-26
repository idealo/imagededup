from imagededup.hashing import Hashing
from imagededup.cnn import CNN
from imagededup.retrieve.retrieval import HashEval
from imagededup.retrieve.retrieval import CosEval
from imagededup.evaluation.performance import Metrics
from pathlib import Path, PosixPath
from typing import Dict, Optional

# Todo: Check passed method_args explicitly?

#  eval.run
# Evaluate(hashing algos output, gound truth)

# FindOptimalEncoding()


class Evaluate:
    def __init__(self, duplicate_map: Dict, ground_truth_map: Dict) -> None:
        self.duplicate_map = duplicate_map
        self.ground_truth_map = ground_truth_map

        self._align_maps()

    def _align_maps(self):
        ground_truth_keys_set = set(self.ground_truth_map.keys())
        duplicate_map_keys_set = set(self.duplicate_map.keys())

        if not ground_truth_keys_set == duplicate_map_keys_set:
            diff = ground_truth_keys_set.difference(duplicate_map_keys_set)
            assert diff == set(), f'Following keys not in duplicate_map:\n{diff}'
            self.duplicate_map = {k: self.duplicate_map[k] for k in self.ground_truth_map.keys()}

    def get_metrics(self, outfile: Optional[str] = None):
        metrics = Metrics(self.ground_truth_map, self.duplicate_map)
        return metrics.get_all_metrics(save_filename=outfile)


class FindOptimalEncoding:
    def __init__(self, img_dir: PosixPath, ground_truth_map: Dict, grid: Dict) -> None:
        self.img_dir = img_dir
        self.ground_truth_map = ground_truth_map
        self.grid = grid

    def _get_features(self, feature_generator):
        self.feature_dict = feature_generator(self.img_dir)

    def _filter_query_retrieval(self):
        self.query_dict = {k: self.feature_dict[k] for k in self.ground_truth_map.keys()}
        self.retrieval_dict = {k: self.feature_dict[k] for k in self.feature_dict.keys() if k not in
                               self.query_dict.keys()}

    """def vla(self):
        feat_vec_query, filemapping_generated_query = cnn_obj._get_file_mapping_feat_vec(dict_query)
        feat_vec_test, filemapping_generated_test = cnn_obj._get_file_mapping_feat_vec(dict_test)

        result_set = CosEval(feat_vec_query, feat_vec_test). \
            get_retrievals_at_thresh(file_mapping_query=filemapping_generated_query,
                                     file_mapping_ret=filemapping_generated_test,
                                     thresh=self.method_args['cosine_threshold'])
        returned_dict = cnn_obj._get_only_filenames(result_set)"""

    def perform_grid_search(self):
        """
        grid: {'phash': {'threshold': [10, 15, 30]}, 'ahash': {'threshold': [10, 15, 30]},
        'cnn': {'threshold': [0.4, 0.8]}}
        :return:
        """
        feature_generation_method = {'phash': Hashing(method='phash').hash_dir,
         'ahash': Hashing(method='ahash').hash_dir,
         'dhash': Hashing(method='dhash').hash_dir,
         'whash': Hashing(method='whash').hash_dir,
         'cnn': CNN.cnn_dir}

        for method, params in self.grid.items():
            self._get_features(feature_generation_method[method])
            self._filter_query_retrieval()

            for param in params:
                rs = HashEval(test=self.retrieval_dict, queries=self.query_dict, hammer=hash_obj.hamming_distance,
                              cutoff=self.method_args['hashing_threshold'])




    def _show_metrics(self, returned_dict):
        metrics = Metrics(self.correct_dict, returned_dict)
        all_metrics = metrics.get_all_metrics(self.save_filename)

        for metric, reading in all_metrics.items():
            print(f'{metric}\t{reading}')

    def _hashing_performance(self) -> None:
        hash_obj = Hashing(method=self.method_args['hash_method'])
        dict_query, dict_test = self._get_features(hash_obj.hash_dir)
        rs = HashEval(test=dict_test, queries=dict_query, hammer=hash_obj.hamming_distance,
                      cutoff=self.method_args['hashing_threshold'])
        returned_dict = rs.retrieve_result_list()
        self._show_metrics(returned_dict)

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
        self._show_metrics(returned_dict)


"""
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
            dict_query = feature_generator(self.query_dir_dict)
            dict_test = feature_generator(self.test_dir_dict)
        else:
            raise ValueError('Query and Tests inputs should be either both path to respective directories or feature '
                             'dictionaries!')
        return dict_query, dict_test

    def _show_metrics(self, returned_dict):
        metrics = Metrics(self.correct_dict, returned_dict)
        all_metrics = metrics.get_all_metrics(self.save_filename)

        for metric, reading in all_metrics.items():
            print(f'{metric}\t{reading}')

    def _hashing_performance(self) -> None:
        hash_obj = Hashing(method=self.method_args['hash_method'])
        dict_query, dict_test = self._get_features(hash_obj.hash_dir)
        rs = HashEval(test=dict_test, queries=dict_query, hammer=hash_obj.hamming_distance,
                      cutoff=self.method_args['hashing_threshold'])
        returned_dict = rs.retrieve_result_list()
        self._show_metrics(returned_dict)

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
        self._show_metrics(returned_dict)
"""

