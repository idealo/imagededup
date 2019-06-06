import copy


class BkTreeNode:
    def __init__(self, node_name, node_value, parent_name=None):
        self.node_name = node_name
        self.node_value = node_value
        self.parent_name = parent_name
        self.children = {}


class BKTree:
    def __init__(self, hash_dict, distance_function):
        self.hash_dict = hash_dict
        self.distance_function = distance_function
        self.all_keys = list(self.hash_dict.keys())
        self.ROOT = self.all_keys[0]
        self.all_keys.remove(self.ROOT)
        self.dict_all = {self.ROOT: BkTreeNode(self.ROOT, self.hash_dict[self.ROOT])}
        self.candidates = [self.dict_all[self.ROOT].node_name]
        self.construct_tree()

    def __insert_in_tree(self, k, current_node):
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

    def construct_tree(self):
        for k in self.all_keys:
            self.__insert_in_tree(k, self.ROOT)

    def __get_next_candidates(self, query, candidate_obj, tolerance):
        dist = self.distance_function(candidate_obj.node_value, query)
        if dist <= tolerance:
            validity = 1
        else:
            validity = 0
        search_range_dist = list(range(dist - tolerance, dist + tolerance + 1))
        candidate_children = candidate_obj.children
        candidates = [k for k in candidate_children.keys() if candidate_children[k] in search_range_dist]
        return candidates, validity, dist

    def search(self, query, tol=10):
        valid_retrievals = {}
        candidates_local = copy.deepcopy(self.candidates)
        while len(candidates_local) != 0:
            candidate_name = candidates_local.pop()
            cand_list, valid_flag, dist = self.__get_next_candidates(query, self.dict_all[candidate_name],
                                                                     tolerance=tol)
            if valid_flag:
                valid_retrievals[candidate_name] = dist
            candidates_local.extend(cand_list)
        return valid_retrievals
