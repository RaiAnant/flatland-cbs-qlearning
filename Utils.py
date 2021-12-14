from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_generators import sparse_rail_generator

from advanced_rail_env import AdvancedRailEnv

import numpy as np
import networkx as nx
import random


def all_paths_to_one_length(pathways):
    """Lengthens shorter paths within a CBS solution to the same length as the longest solution.

    :param pathways: A solution given by the CBS algorithm.
    :return: An array of paths
    """
    max_len = max([len(p) for p in pathways])
    new_paths = []
    for p in pathways:
        p_len = len(p)
        if p_len < max_len:
            p += [p[-1] for _ in range(max_len - p_len)]
        new_paths.append(p)
    return np.array(new_paths)


def create_environment():
    """Creates a Flatland environment.

    :return: The environment and the maximum amount of allowed steps.
    """
    width = 24
    height = 24
    amount_agents = 6
    amount_cities = 2
    max_steps = int(4 * 2 * (width + height + (amount_agents / amount_cities)))
    seed = 420
    random.seed(seed)
    np.random.seed(seed)
    rail_env = AdvancedRailEnv(
        width=width,
        height=height,
        number_of_agents=amount_agents,
        rail_generator=sparse_rail_generator(
            max_num_cities=amount_cities,
            seed=seed,
            grid_mode=False,
            max_rails_between_cities=2
        ),
        obs_builder_object=TreeObsForRailEnv(max_depth=2)
    )

    return rail_env, max_steps


def pre_process_features(observation):
    """Takes the raw observation vector of the environment and amends it to our needs.

    :param observation: The observation of the environment
    :return: A vector which contains the features which we want.
    """
    IDX_SUC_NODES = 12
    features = np.array(observation[:IDX_SUC_NODES])

    feature_vector = np.zeros(9)

    # Channel 0: Distance from agent to target cell in number of cells,
    # if target is within explored branch
    feature_vector[0] = features[0] if features[0] != np.inf else -1
    # Channel 1: Distance from agent to target of another agent in number of cells
    feature_vector[1] = features[1] if features[1] != np.inf else -1
    # Channel 2: Distance from agent to another agent
    feature_vector[2] = features[2] if features[2] != np.inf else -1
    # Channel 3: Distance to 'unusable' switch in number of cells
    feature_vector[3] = features[4] if features[4] != np.inf else -1
    # Channel 4: Distance to next switch cell (or node respectively)
    feature_vector[4] = features[5] if features[5] != np.inf else -1
    # Channel 5: Minimum distance to agent's target
    feature_vector[5] = features[6] if features[6] != np.inf else -1
    # Channel 6: Number of agents going in the same direction found on path to node
    feature_vector[6] = features[7] if features[7] != np.inf else -1
    # Channel 7: Number of agents going in the opposite direction found on path to node
    feature_vector[7] = features[8] if features[8] != np.inf else -1
    # Channel 8: Number of agents ready to depart but no yet active
    feature_vector[8] = features[11] if features[11] != np.inf else -1

    return feature_vector


def add_nodes_to_tree(tree,
                      observation,
                      node_key,
                      prev_node_key=None,
                      idx_suc_nodes=12,
                      tree_depth=2):
    """Adds nodes recursively to the tree.

    :param tree: The tree to which nodes shall be added.
    :param observation: The observation which shall be stored in the tree
    :param node_key: The key of a node corresponds to the position of a cell.
    :param prev_node_key: The key of the parent node.
    :param idx_suc_nodes: The index of the successor nodes.
    :param tree_depth: The maximal tree depth
    :return: The id of a node.
    """

    if observation is None:
        node_features = np.full(9, fill_value=-1)
    else:
        node_features = pre_process_features(observation)

    tree.add_node(node_key, features=node_features)
    running_number = node_key[0] + 1
    if prev_node_key is not None:
        tree.add_edge(prev_node_key, node_key)

    next_level = node_key[2] + 1
    if next_level <= tree_depth and observation is None:
        for direction in ["F", "L", "B", "R"]:
            running_number = add_nodes_to_tree(
                tree,
                None,
                (running_number, direction, next_level),
                node_key
            )
    elif next_level <= tree_depth:
        for direction, node in observation[idx_suc_nodes].items():
            if node != -np.inf:
                running_number = add_nodes_to_tree(
                    tree,
                    node,
                    (running_number, direction, next_level),
                    node_key
                )
            else:
                running_number = add_nodes_to_tree(
                    tree,
                    None,
                    (running_number, direction, next_level),
                    node_key
                )

    return running_number


def preprocess_observation(observation_dict, amt_nodes=21, amt_features=9):
    """Preprocesses an observation dictionary to feature- and adjacency matrices.

    :param observation_dict: The observation dictionary returned by the "TreeObsForRailEnv"-
                             observation builder.
    :param amt_nodes: The amount of nodes which will be stored within the matrices.
    :param amt_features: The amount of features which will be stored within the feature matrices.
    :return: A batch of feature matrices and a batch of adjacency matrices.
    """
    feature_matrices = np.zeros((len(observation_dict), amt_nodes, amt_features), dtype=np.float32)
    adj_matrices = np.zeros((len(observation_dict), amt_nodes, amt_nodes), dtype=np.int32)

    for agent_handle, observation in observation_dict.items():
        tree = nx.DiGraph()
        add_nodes_to_tree(tree, observation, (0, "R", 0))

        feature_matrix = np.array([value["features"] for (_, value) in tree.nodes(data=True)])
        adjacency_matrix = np.array(nx.adjacency_matrix(tree).todense())

        feature_matrices[agent_handle] = feature_matrix
        adj_matrices[agent_handle] = adjacency_matrix

    return feature_matrices, adj_matrices
