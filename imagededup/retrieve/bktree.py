import copy
from types import FunctionType
from typing import Tuple, Dict

# Implementation reference: https://signal-to-noise.xyz/post/bk-tree/


class BkTreeNode:
    def __init__(self, node_name: str, node_value: str, parent_name: str = None) -> None:
        self.node_name = node_name
        self.node_value = node_value
        self.parent_name = parent_name
        self.children = {}


class BKTree:
    def __init__(self, hash_dict: Dict, distance_function: FunctionType) -> None:
        self.hash_dict = hash_dict  # database
        self.distance_function = distance_function
        self.all_keys = list(self.hash_dict.keys())
        self.ROOT = self.all_keys[0]
        self.all_keys.remove(self.ROOT)
        self.dict_all = {self.ROOT: BkTreeNode(self.ROOT, self.hash_dict[self.ROOT])}
        self.candidates = [self.dict_all[self.ROOT].node_name]  # Initial value is root
        self.construct_tree()

    def __insert_in_tree(self, k: str, current_node: str) -> int:
        dist_current_node = self.distance_function(self.hash_dict[k], self.dict_all[current_node].node_value)
        if not self.dict_all[current_node].children:
            self.dict_all[current_node].children[k] = dist_current_node
            self.dict_all[k] = BkTreeNode(k, self.hash_dict[k], parent_name=current_node)
        elif dist_current_node not in list(self.dict_all[current_node].children.values()):
            self.dict_all[current_node].children[k] = dist_current_node
            self.dict_all[k] = BkTreeNode(k, self.hash_dict[k], parent_name=current_node)
        else:
            for i, val in self.dict_all[current_node].children.items():
                if val == dist_current_node:
                    node_to_add_to = i
                    break
            self.__insert_in_tree(k, node_to_add_to)
        return 0

    def construct_tree(self) -> None:
        for k in self.all_keys:
            self.__insert_in_tree(k, self.ROOT)

    def _get_next_candidates(self, query: str, candidate_obj: BkTreeNode, tolerance: int) -> Tuple[list, int, float]:
        dist = self.distance_function(candidate_obj.node_value, query)
        if dist <= tolerance:
            validity = 1
        else:
            validity = 0
        search_range_dist = list(range(dist - tolerance, dist + tolerance + 1))
        candidate_children = candidate_obj.children
        candidates = [k for k in candidate_children.keys() if candidate_children[k] in search_range_dist]
        return candidates, validity, dist

    def search(self, query: str, tol: int = 5) -> Dict:
        """
        Function to search the bktree given a hash of the query image
        :param query: hash string
        :param tol: distance upto which duplicate is valid
        :return: {valid_retrieval_filename: distance, ...}
        """
        valid_retrievals = {}
        candidates_local = copy.deepcopy(self.candidates)
        while len(candidates_local) != 0:
            candidate_name = candidates_local.pop()
            cand_list, valid_flag, dist = self._get_next_candidates(query, self.dict_all[candidate_name],
                                                                     tolerance=tol)
            if valid_flag:
                valid_retrievals[candidate_name] = dist
            candidates_local.extend(cand_list)
        return valid_retrievals
