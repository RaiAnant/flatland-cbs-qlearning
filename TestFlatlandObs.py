from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.utils.rendertools import RenderTool

from advanced_rail_env import AdvancedRailEnv

import random
import numpy as np
import networkx as nx


# def pre_process_features(observation):
#
#     IDX_SUC_NODES = 12
#     features = np.array(observation[:IDX_SUC_NODES])
#
#     feature_vector = np.zeros(9)
#
#     # Channel 0: Distance from agent to target cell in number of cells,
#     # if target is within explored branch
#     feature_vector[0] = features[0] if features[0] != np.inf else -1
#     # Channel 1: Distance from agent to target of another agent in number of cells
#     feature_vector[1] = features[1] if features[1] != np.inf else -1
#     # Channel 2: Distance from agent to another agent
#     feature_vector[2] = features[2] if features[2] != np.inf else -1
#     # Channel 3: Distance to 'unusable' switch in number of cells
#     feature_vector[3] = features[4] if features[4] != np.inf else -1
#     # Channel 4: Distance to next switch cell (or node respectively)
#     feature_vector[4] = features[5] if features[5] != np.inf else -1
#     # Channel 5: Minimum distance to agent's target
#     feature_vector[5] = features[6] if features[6] != np.inf else -1
#     # Channel 6: Number of agents going in the same direction found on path to node
#     feature_vector[6] = features[7] if features[7] != np.inf else -1
#     # Channel 7: Number of agents going in the opposite direction found on path to node
#     feature_vector[7] = features[8] if features[8] != np.inf else -1
#     # Channel 8: Number of agents ready to depart but no yet active
#     feature_vector[8] = features[11] if features[11] != np.inf else -1
#
#     return feature_vector
#
#
# def add_nodes_to_tree(tree,
#                       observation,
#                       node_key,
#                       prev_node_key=None,
#                       idx_suc_nodes=12,
#                       tree_depth=2):
#
#     if observation is None:
#         node_features = np.full(9, fill_value=-1)
#     else:
#         node_features = pre_process_features(observation)
#
#     tree.add_node(node_key, features=node_features)
#     running_number = node_key[0] + 1
#     if prev_node_key is not None:
#         tree.add_edge(prev_node_key, node_key)
#
#     next_level = node_key[2] + 1
#     if next_level <= tree_depth and observation is None:
#         for direction in ["F", "L", "B", "R"]:
#             running_number = add_nodes_to_tree(
#                 tree,
#                 None,
#                 (running_number, direction, next_level),
#                 node_key
#             )
#     elif next_level <= tree_depth:
#         for direction, node in observation[idx_suc_nodes].items():
#             if node != -np.inf:
#                 running_number = add_nodes_to_tree(
#                     tree,
#                     node,
#                     (running_number, direction, next_level),
#                     node_key
#                 )
#             else:
#                 running_number = add_nodes_to_tree(
#                     tree,
#                     None,
#                     (running_number, direction, next_level),
#                     node_key
#                 )
#
#     return running_number
#
#
# def preprocess_observation(observation_dict):
#     new_observation_dict = {}
#     for agent_handle, observation in observation_dict.items():
#         tree = nx.DiGraph()
#         add_nodes_to_tree(tree, observation, (0, "R", 0))
#         adjacency_matrix = np.array(nx.adjacency_matrix(tree).todense())
#         feature_matrix = np.array([value["features"] for (_, value) in tree.nodes(data=True)])
#         new_observation_dict[agent_handle] = (feature_matrix, adjacency_matrix)
#     return new_observation_dict


if __name__ == "__main__":
    width = 24
    height = 24
    max_episode_no = 10
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
    env_renderer = RenderTool(rail_env)

    episode_no = 0
    while episode_no < max_episode_no:

        rail_env.reset()
        env_renderer.reset()
        env_renderer.render_env(
            show=True,
            frames=True,
            show_rowcols=True,
            show_observations=True,
            show_predictions=False,
            step=True
        )

        episode_step = 0

        while episode_step < max_steps:

            env_renderer.render_env(
                show=True,
                frames=True,
                show_rowcols=True,
                show_observations=True,
                show_predictions=False,
                step=True
            )

            episode_progress = int((episode_step + 1) / max_steps)
            # actions = {agent_no: np.random.randint(5) for agent_no in env.get_agent_handles()}
            actions = {a: 2 for a in rail_env.get_agent_handles()}
            obs, rewards, done, info = rail_env.step(actions)
            obs = preprocess_observation(obs)

            # stop_episode = policy.step(actions, rewards)
            episode_step += 1

            if done["__all__"]:
                break

        episode_no += 1
