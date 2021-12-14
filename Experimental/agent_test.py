import sys
import time
from collections import deque

from flatland.utils.rendertools import RenderTool

from q_learning import is_at_important_cell, translate_path_into_actions
from train_loop import create_environment, preprocess_observation
from matplotlib import pyplot as plt

import tensorflow as tf
import numpy as np


def get_actions(f_matrices, a_matrices, paths, ts):

    cbs_actions = translate_path_into_actions(paths, ts)
    acts = {}
    for handle in env.get_agent_handles():
        if is_at_important_cell(handle):
            agent_action = model.predict([f_matrices[handle], a_matrices[handle]])
            acts[handle] = agent_action
        else:
            acts[handle] = cbs_actions[handle]

    return acts

if __name__ == "__main__":

    # Display reward improvement
    rewards = np.load("../SavedModel/rewards.npy")
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
    ax2.grid(True)
    plt.show()

    # Test the model
    model = tf.keras.models.load_model('SavedModel/Model')
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
        for time_step in range(max_steps_per_episode):
            feature_matrices, adj_matrices = preprocess_observation(observations)
            env_renderer.render_env(
                show=True,
                frames=True,
                show_rowcols=True,
                show_observations=True,
                show_predictions=False,
                step=True
            )
            action_logits, state_values = model([feature_matrices, adj_matrices])
            action_logits = tf.squeeze(action_logits)
            actions = tf.cast(tf.random.categorical(action_logits, 1)[:, 0], tf.int32).numpy()
            action_dictionary = {}
            for agent_handle, action in enumerate(actions):
                action_dictionary[agent_handle] = action
            sys.stdout.write(f"\rEpisode: {episode_n} Time Step: {time_step} Actions: {action_dictionary}")
            observations, _, dones, _ = env.step(action_dictionary)
            if np.all(np.array(list(dones.values()), dtype=np.int32)):
                break
            time.sleep(0.25)
