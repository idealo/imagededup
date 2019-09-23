from collections import OrderedDict

from imagededup.methods.hashing import Hashing
from imagededup.handlers.search.bktree import BKTree, BkTreeNode

# Test BkTreeNode


def initialize_for_bktree():
    hash_dict = OrderedDict(
        {'a': '9', 'b': 'D', 'c': 'A', 'd': 'F', 'e': '2', 'f': '6', 'g': '7', 'h': 'E'}
    )
    dist_func = Hashing.hamming_distance
    return hash_dict, dist_func


def test_bktreenode_correct_initialization():
    node_name, node_value, parent_name = 'test_node', '1aef', None
    node = BkTreeNode(node_name, node_value, parent_name)
    assert node.node_name == 'test_node'
    assert node.node_value == '1aef'
    assert node.parent_name is None
    assert len(node.children) == 0


# test BKTree class


def test_insert_tree():
    # initialize root node and add 1 new node, check it goes as root's child and has it's parent as root
    _, dist_func = initialize_for_bktree()
    hash_dict = {'a': '9', 'b': 'D'}
    bk = BKTree(hash_dict, dist_func)
    assert bk.ROOT == 'a'
    assert 'b' in list(bk.dict_all['a'].children.keys())
    assert bk.dict_all['b'].parent_name == 'a'


def test_insert_tree_collision():
    # initialize root node, add 1 new node and enter another node with same distance from root, check it goes not as
    # root's child but the other node's child
    _, dist_func = initialize_for_bktree()
    hash_dict = OrderedDict(
        {'a': '9', 'b': 'D', 'c': '8'}
    )  # to guarantee that 'a' is the root of the tree
    bk = BKTree(hash_dict, dist_func)
    assert bk.ROOT == 'a'
    assert len(bk.dict_all[bk.ROOT].children) == 1
    assert 'c' in list(bk.dict_all['b'].children.keys())


def test_insert_tree_different_nodes():
    # initialize root node, add 1 new node and enter another node with different distance from root, check it goes as
    # root's child and not as the other node's child
    _, dist_func = initialize_for_bktree()
    hash_dict = OrderedDict(
        {'a': '9', 'b': 'D', 'c': 'F'}
    )  # to guarantee that 'a' is the root of the tree
    bk = BKTree(hash_dict, dist_func)
    assert bk.ROOT == 'a'
    assert len(bk.dict_all[bk.ROOT].children) == 2
    assert set(['b', 'c']) <= set(bk.dict_all[bk.ROOT].children.keys())


def test_insert_tree_check_distance():
    # initialize root node, add 1 new node and enter another node with different distance from root, check that the
    # distance recorded in the root's children dictionary is as expected
    _, dist_func = initialize_for_bktree()
    hash_dict = OrderedDict(
        {'a': '9', 'b': 'D', 'c': 'F'}
    )  # to guarantee that 'a' is the root of the tree
    bk = BKTree(hash_dict, dist_func)
    assert bk.ROOT == 'a'
    assert bk.dict_all[bk.ROOT].children['b'] == 1
    assert bk.dict_all[bk.ROOT].children['c'] == 2


def test_construct_tree():
    # Input a complete tree and check for each node the children and parents
    hash_dict, dist_func = initialize_for_bktree()
    bk = BKTree(hash_dict, dist_func)
    # check root
    assert bk.ROOT == 'a'
    # check that expected leaf nodes have no children (they're actually leaf nodes)
    leaf_nodes = set(
        [k for k in bk.dict_all.keys() if len(bk.dict_all[k].children) == 0]
    )
    expected_leaf_nodes = set(['b', 'd', 'f', 'h'])
    assert leaf_nodes == expected_leaf_nodes
    # check that root node ('a') has 4 children
    assert len(bk.dict_all[bk.ROOT].children) == 4
    # check that 'c' has 'd' as it's child at distance 2
    assert bk.dict_all['c'].children['d'] == 2


def test_search():
    # Input a tree and send a search query, check whether correct number of retrievals are returned
    hash_dict, dist_func = initialize_for_bktree()
    bk = BKTree(hash_dict, dist_func)
    query = '5'
    valid_retrievals = bk.search(query, tol=2)
    assert len(valid_retrievals) == 5


def test_search_correctness():
    # Input a tree and send a search query, check whether correct retrievals are returned
    hash_dict, dist_func = initialize_for_bktree()
    bk = BKTree(hash_dict, dist_func)
    query = '5'
    valid_retrievals = bk.search(query, tol=2)
    assert set([i[0] for i in valid_retrievals]) == set(['a', 'f', 'g', 'd', 'b'])


def test_search_zero_tolerance():
    # Input a tree and send a search query, check whether zero retrievals are returned for zero tolerance
    hash_dict, dist_func = initialize_for_bktree()
    bk = BKTree(hash_dict, dist_func)
    query = '5'
    valid_retrievals = bk.search(query, tol=0)
    assert len(valid_retrievals) == 0


def test_search_dist():
    # Input a tree and send a search query, check whether correct distance for a retrieval is returned
    hash_dict, dist_func = initialize_for_bktree()
    bk = BKTree(hash_dict, dist_func)
    query = '5'
    valid_retrievals = bk.search(query, tol=2)
    assert [i for i in valid_retrievals if i[0] == 'a'][0][1] == 2


def test_get_next_candidates_valid():
    # Give a partial tree as input and check that for a query, expected candidates and validity flag are obtained
    hash_dict, dist_func = initialize_for_bktree()
    bk = BKTree(hash_dict, dist_func)
    assert bk.ROOT == 'a'
    query = '5'
    candidates, validity, dist = bk._get_next_candidates(
        query, bk.dict_all[bk.ROOT], tolerance=2
    )
    candidates = set(candidates)
    assert candidates <= set(['b', 'c', 'e', 'f'])
    assert validity


def test_get_next_candidates_invalid():
    # Give a tree as input and check that for a query, validity flag is 0
    hash_dict, dist_func = initialize_for_bktree()
    bk = BKTree(hash_dict, dist_func)
    assert bk.ROOT == 'a'
    query = '5'
    _, validity, _ = bk._get_next_candidates(query, bk.dict_all[bk.ROOT], tolerance=1)
    assert not validity


def test_tolerance_affects_retrievals():
    # Give a partial tree as input and check that for a query, increased tolerance gives more retrievals as expected for
    # the input tree
    hash_dict, dist_func = initialize_for_bktree()
    bk = BKTree(hash_dict, dist_func)
    assert bk.ROOT == 'a'
    query = '5'
    candidates, _, _ = bk._get_next_candidates(query, bk.dict_all[bk.ROOT], tolerance=1)
    low_tolerance_candidate_len = len(candidates)
    candidates, _, _ = bk._get_next_candidates(query, bk.dict_all[bk.ROOT], tolerance=2)
    high_tolerance_candidate_len = len(candidates)
    assert high_tolerance_candidate_len > low_tolerance_candidate_len
