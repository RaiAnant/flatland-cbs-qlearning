import sys
import time
from collections import deque

from flatland.utils.rendertools import RenderTool

from Utils import create_environment, all_paths_to_one_length, preprocess_observation
from conflict_based_search import cbs, get_agent_position, translate_path_into_actions
from matplotlib import pyplot as plt

import tensorflow as tf
import numpy as np


def is_at_important_cell(agent_handle):
    """Compute whether or not an agent is at an important cell.

    :param agent_handle: The handle of the agent to check.
    :return: Returns a boolean that tells whether or not the agent is located at either its spawn,
             its target or at a switch.
    """

    agent_pos = get_agent_position(env, agent_handle)
    is_switch = np.count_nonzero(env.graph.nodes[agent_pos]["all_transitions"]) != 2
    is_spawn = env.agents[agent_handle].initial_position == agent_pos
    is_target = env.agents[agent_handle].target == agent_pos

    return is_switch or is_spawn or is_target


def get_actions(f_matrices, a_matrices, paths_, time_step):
    """Invokes the model and translates its decisions into actions.

    :param f_matrices: Latest observation's feature matrices of each agent.
    :param a_matrices: Latest observation's adjacency matrices of each agent.
    :param paths_: The solution paths by the CBS algorithm.
    :param time_step: The current time step.
    :return: A dictionary containing an action for each agent.
    """
    cbs_actions = translate_path_into_actions(env, paths_, time_step)
    model_actions = model.predict([f_matrices, a_matrices])
    model_actions = np.argmax(np.squeeze(model_actions), axis=1)
    acts = {}
    for handle in env.get_agent_handles():
        if is_at_important_cell(handle):
            agent_action = model_actions[handle]
            acts[handle] = 4 if agent_action == 0 else cbs_actions[handle]
        else:
            acts[handle] = cbs_actions[handle]

    return acts


if __name__ == "__main__":

    # Display reward improvement
    rewards = np.load("SavedModel/rewards.npy")
    running_rewards = []
    reward_window = deque(maxlen=100)
    for start_idx in range(len(rewards)):
        reward_window.append(rewards[start_idx])
        running_rewards.append(np.mean(reward_window))
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.suptitle("Reward Analysis of the Trained Model")
    ax1.plot(rewards)
    ax1.set_ylabel("Rewards")
    ax1.grid(True)
    ax2.plot(running_rewards)
    ax2.set_ylabel("Running Rewards")
    ax2.set_xlabel("Episodes")
    ax2.grid(True)
    plt.show()

    # Plot the model
    model = tf.keras.models.load_model('SavedModel/Model')
    # tf.keras.utils.plot_model(model, "model.png", show_shapes=True)

    # Test the model
    env, max_steps_per_episode = create_environment()
    env_renderer = RenderTool(env)
    max_episodes = 10
    for episode_n in range(max_episodes):
        observations, _ = env.reset()
        env_renderer.reset()
        env_renderer.render_env(
            show=True,
            frames=True,
            show_rowcols=True,
            show_observations=True,
            show_predictions=False,
            step=True
        )

        solution = cbs(env, verbose=False)
        while not bool(solution):
            obs, _ = env.reset()
            env_renderer.reset()
            solution = cbs(env, verbose=False)
        paths = list(zip(*solution))[1]
        paths = all_paths_to_one_length(paths)

        for ts in range(max_steps_per_episode):
            feature_matrices, adj_matrices = preprocess_observation(observations)
            env_renderer.render_env(
                show=True,
                frames=True,
                show_rowcols=True,
                show_observations=True,
                show_predictions=False,
                step=True
            )
            action_dictionary = get_actions(feature_matrices, adj_matrices, paths, ts)
            observations, _, dones, _ = env.step(action_dictionary)
            if np.all(np.array(list(dones.values())[:-1], dtype=np.int32)):
                break

            actions_as_array = np.array(list(action_dictionary.values()))
            if np.all(actions_as_array == 4):
                print(" ### All agents stopped! ###")
                break

            sys.stdout.write(
                f"\rTime Step: {ts} / {max_steps_per_episode}"
                f" Actions: {action_dictionary.values()}"
                f" Agent positions: {[get_agent_position(env, a) for a in env.get_agent_handles()]}"
                f" Done values: {dones}"
            )
            time.sleep(0.25)
