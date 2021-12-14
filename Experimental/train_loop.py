import random

from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_generators import sparse_rail_generator

from collections import deque

import tqdm
import tensorflow as tf
import numpy as np
import networkx as nx
import logging
import os
import warnings

from Experimental.a2c_agent import ActorCritic
from advanced_rail_env import AdvancedRailEnv
from conflict_based_search import cbs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
warnings.filterwarnings("ignore")


def env_step(actions, pred_actions):
    if len(actions.shape) > 1:
        actions = actions.reshape((actions.shape[-1],))

    if len(pred_actions.shape) > 1:
        pred_actions = pred_actions.reshape((pred_actions.shape[-1],))

    action_dic = {}
    for agent_handle, action in enumerate(actions):
        action_dic[agent_handle] = action

    obs, reward, done, _ = env.step(action_dic)
    feature_matrices, adj_matrices = preprocess_observation(obs)

    reward = np.array(list(reward.values()), dtype=np.int32)
    for idx, (a, p_a) in enumerate(zip(actions, pred_actions)):
        if a != p_a:
            reward[idx] -= 100
        else:
            reward[idx] += 100

    done = np.array(list(done.values()), dtype=np.int32)
    return feature_matrices, adj_matrices, np.array(reward, np.float32), np.array(done, np.int32)


def tf_env_step(actions, pred_actions):
    return tf.numpy_function(
        env_step,
        [actions, pred_actions],
        [tf.float32, tf.int32, tf.float32, tf.int32]
    )


def translate_path_into_actions_(pathways, time_step):

    acts = np.full(len(env.get_agent_handles()), -1, dtype=np.int32)
    use_network = np.full(len(env.get_agent_handles()), 0, dtype=np.int32)

    if time_step >= len(pathways[0]) - 1:
        return np.full(len(env.get_agent_handles()), 4, dtype=np.int32), use_network

    for agent_handle, pathway in enumerate(pathways):
        if np.all(pathway[time_step:] == pathway[time_step]):
            env.agents[agent_handle].status = RailAgentStatus.DONE_REMOVED
            acts[agent_handle] = 4

    current_positions = pathways[:, time_step]
    next_positions = pathways[:, time_step + 1]
    differences = next_positions - current_positions

    for agent_handle, diff in enumerate(differences):

        if acts[agent_handle] != -1:
            continue

        # Do not activate agents if they are located in their 'spawn' and shall not move
        if env.agents[agent_handle].status == RailAgentStatus.READY_TO_DEPART \
                and np.array_equal(diff, [0, 0]):
            acts[agent_handle] = 0
            continue
        # Activate agents otherwise
        elif env.agents[agent_handle].status == RailAgentStatus.READY_TO_DEPART:
            env.agents[agent_handle].position = env.agents[agent_handle].initial_position
            env.agents[agent_handle].status = RailAgentStatus.ACTIVE

        # Check whether or not the network shall be used
        use_network[agent_handle] = int(env.is_switch(current_positions[agent_handle]))

        if np.array_equal(diff, [-1, 0]):
            cardinal_dir_next_pos = 0
        elif np.array_equal(diff, [0, 1]):
            cardinal_dir_next_pos = 1
        elif np.array_equal(diff, [1, 0]):
            cardinal_dir_next_pos = 2
        elif np.array_equal(diff, [0, -1]):
            cardinal_dir_next_pos = 3
        elif np.array_equal(diff, [0, 0]):
            acts[agent_handle] = 4
        else:
            raise RuntimeError("Something went wrong!")
        if acts[agent_handle] == -1:
            agent_orientation = env.agents[agent_handle].direction
            action = (cardinal_dir_next_pos - agent_orientation + 2) % 4
            acts[agent_handle] = action

    return acts, use_network


def translate_path_into_actions(pathways, time_step):
    return tf.numpy_function(translate_path_into_actions_, [pathways, time_step],
                             [np.int32, np.int32])


def prepare_tensor_entries_(episode_action_probs, episode_state_value, episode_rewards, agents):
    all_handles = env.get_agent_handles()
    action_probs, state_values, rewards = [], [], []
    next_index = 0
    for agent_handle in all_handles:
        if agent_handle in agents:
            action_probs.append(episode_action_probs[next_index])
            state_values.append(episode_state_value[next_index])
            rewards.append(episode_rewards[next_index])
            next_index += 1
        else:
            action_probs.append(0.)
            state_values.append(0.)
            rewards.append(0.)

    action_probs = np.array(action_probs, dtype=np.float32)
    state_values = np.array(state_values, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)

    return action_probs, state_values, rewards


def prepare_tensor_entries(episode_action_probs, episode_state_value, episode_rewards, agents):
    return tf.numpy_function(
        prepare_tensor_entries_,
        [episode_action_probs, episode_state_value, episode_rewards, agents],
        [np.float32, np.float32, np.float32]
    )


def translate_(path, time_step, agent_handle):

    current_position = path[time_step]
    use_network = int(env.is_switch(current_position))

    if time_step >= len(path) - 1:
        return np.array(4), np.array(0)

    if np.all(path[time_step:] == path[time_step]):
        return np.array(4), np.array(0)

    next_position = path[time_step + 1]
    difference = next_position - current_position

    # Do not activate agents if they are located in their 'spawn' and shall not move
    if env.agents[agent_handle].status == RailAgentStatus.READY_TO_DEPART and np.array_equal(
            difference, [0, 0]):
        return np.array(0), np.array(use_network)
    # Activate agents otherwise, if necessary
    elif env.agents[agent_handle].status == RailAgentStatus.READY_TO_DEPART:
        env.agents[agent_handle].position = env.agents[agent_handle].initial_position
        env.agents[agent_handle].status = RailAgentStatus.ACTIVE

    if np.array_equal(difference, [-1, 0]):
        cardinal_dir_next_pos = 0
    elif np.array_equal(difference, [0, 1]):
        cardinal_dir_next_pos = 1
    elif np.array_equal(difference, [1, 0]):
        cardinal_dir_next_pos = 2
    elif np.array_equal(difference, [0, -1]):
        cardinal_dir_next_pos = 3
    elif np.array_equal(difference, [0, 0]):
        return np.array(4), np.array(use_network)
    else:
        raise RuntimeError("Something went wrong!")

    agent_orientation = env.agents[agent_handle].direction
    action = (cardinal_dir_next_pos - agent_orientation + 2) % 4
    return np.array(action), np.array(use_network)


def translate(path, time_step, agent_handle):

    return tf.numpy_function(translate_, [path, time_step, agent_handle], [np.int32, np.int32])


def run_episode_2(init_feature_matrices, init_adj_matrices, model, max_steps, cbs_solution):
    COL_IDX_START_ADJ = 9

    action_probabilities = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    state_values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    feature_matrices, adj_matrices = init_feature_matrices, init_adj_matrices
    feature_matrices_shape = feature_matrices.shape
    adj_matrices_shape = adj_matrices.shape

    for time_step in tf.range(max_steps):

        matrices = tf.concat([feature_matrices, adj_matrices], 2)
        for agent_handle, matrix in enumerate(matrices):
            feature_matrix = matrix[:, :COL_IDX_START_ADJ]
            adj_matrix = matrix[:, COL_IDX_START_ADJ:]
            path = cbs_solution[agent_handle]
            cbs_action, use_network = translate(path, time_step, agent_handle)

    return


def run_episode(init_feature_matrices, init_adj_matrices, model, max_steps, cbs_solution):
    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    state_values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    feature_matrices, adj_matrices = init_feature_matrices, init_adj_matrices
    feature_matrices_shape = feature_matrices.shape
    adj_matrices_shape = adj_matrices.shape

    for time_step in tf.range(max_steps):

        cbs_actions, use_network = translate_path_into_actions(cbs_solution, time_step)
        # feature_matrices_to_use, adj_matrices_to_use, agents = [], [], []
        # for agent_handle, use_n in enumerate(use_network):
        #     if use_n:
        #         feature_matrices_to_use.append(feature_matrices[agent_handle])
        #         adj_matrices_to_use.append(adj_matrices[agent_handle])
        #         agents.append(agent_handle)

        # feature_matrices, adj_matrices, episode_rewards, dones = tf_env_step(cbs_actions)

        # if bool(adj_matrices_to_use):
        #     feature_matrices_to_use = tf.stack(feature_matrices_to_use)
        #     adj_matrices_to_use = tf.stack(adj_matrices_to_use)

        # Action logits for each observation
        action_logits, episode_state_value = model([feature_matrices, adj_matrices])
        action_logits = tf.squeeze(action_logits)
        # if len(action_logits.shape) < 2:
        #     action_logits = tf.expand_dims(action_logits, 0)
        # tf.random.categorical([[logits]], #samples to draw from each element within batch)
        network_actions = tf.cast(tf.random.categorical(action_logits, 1)[:, 0], tf.int32)

        episode_action_probs = tf.nn.softmax(action_logits, axis=1)
        prob_indices = tf.stack([tf.range(action_logits.shape[0]), network_actions], axis=1)
        episode_action_probs = tf.gather_nd(episode_action_probs, prob_indices)

        episode_state_value = tf.squeeze(episode_state_value)
        # if len(episode_state_value.shape) < 1:
        #     episode_state_value = tf.expand_dims(episode_state_value, 0)

        # episode_action_probs, episode_state_value, episode_rewards = prepare_tensor_entries(
        #     episode_action_probs,
        #     episode_state_value,
        #     episode_rewards,
        #     agents
        # )

        feature_matrices, adj_matrices, episode_rewards, dones = tf_env_step(
            cbs_actions,
            network_actions
        )

        action_probs = action_probs.write(time_step, episode_action_probs)
        state_values = state_values.write(time_step, episode_state_value)
        rewards = rewards.write(time_step, episode_rewards)

        if tf.math.reduce_all(tf.cast(dones, tf.bool)):
            break

        adj_matrices.set_shape(adj_matrices_shape)
        feature_matrices.set_shape(feature_matrices_shape)

    action_probs = action_probs.stack()
    state_values = state_values.stack()
    rewards = rewards.stack()

    return action_probs, state_values, rewards


def get_expected_return(rewards, gamma, standardize=True, amt_agents=6):
    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.zeros(amt_agents)
    discounted_sum_shape = discounted_sum.shape
    for j in tf.range(n):
        reward = rewards[j]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(j, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
        returns = (returns - tf.math.reduce_mean(returns, axis=0)) / (
                    tf.math.reduce_std(returns, axis=0) + eps)

    return returns


def compute_loss(action_probs, state_values, returns):
    advantage = returns - state_values

    action_log_probs = tf.math.log(action_probs)
    actor_losses = -tf.math.reduce_sum(action_log_probs * advantage, axis=0)

    critic_losses = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    for j in tf.range(state_values.shape[1]):
        critic_loss = huber_loss(state_values[:, j], returns[:, j])
        critic_losses = critic_losses.write(j, critic_loss)
    critic_losses = critic_losses.stack()

    return actor_losses + critic_losses


@tf.function
def train_step(init_feature_matrices,
               init_adj_matrices,
               model,
               max_steps,
               gamma,
               optimizer,
               cbs_solution):
    # pa = per_agent
    with tf.GradientTape() as tape:
        action_probs, state_values, rewards = run_episode(
            init_feature_matrices,
            init_adj_matrices,
            model,
            max_steps,
            cbs_solution
        )
        returns = get_expected_return(rewards, gamma)
        loss = compute_loss(action_probs, state_values, returns)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    episode_rewards_pa = tf.math.reduce_sum(rewards, axis=0)
    return episode_rewards_pa


def compute_done_rate():
    reached_target = {a: False for a in env.get_agent_handles()}

    # Remember average done rate (Use 'reached_target', because in 'done', everything is set 'True'
    # when "__all__" is set to 'True'
    for handle in env.get_agent_handles():
        reached_target[handle] = env.agents[handle].status == RailAgentStatus.DONE_REMOVED \
                                 or env.agents[handle].status == RailAgentStatus.DONE

    task_finished = tf.math.reduce_sum(
        [int(reached_target[idx]) for idx in env.get_agent_handles()]
    )

    return task_finished / max(1, env.get_num_agents())





if __name__ == "__main__":

    network = ActorCritic(amt_actions=5, amt_nodes=21, amt_features=9)
    optimizer_ = tf.keras.optimizers.Adam(learning_rate=0.01)
    gamma_ = 0.99
    eps = np.finfo(np.float32).eps.item()
    huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    saving_rhythm = 10

    # Initialize environment
    env, max_steps_per_episode = create_environment()

    # min_episodes_criterion = 100
    max_episodes = 100_000

    # goal_avg_done_rate = 0.9
    # episodes_done_rate = deque(maxlen=min_episodes_criterion)
    all_episodes_reward = []
    window_episodes_reward = deque(maxlen=100)

    with tqdm.trange(max_episodes) as t:
        for i in t:

            if i == 485:
                print("asdf")

            obs, _ = env.reset()
            init_f_matrices, init_a_matrices = preprocess_observation(obs)

            solution = cbs(env, verbose=False)
            while not bool(solution):
                obs, _ = env.reset()
                solution = cbs(env, verbose=False)
            paths = list(zip(*solution))[1]
            paths = all_paths_to_one_length(paths)

            episode_reward = train_step(
                init_f_matrices,
                init_a_matrices,
                network,
                max_steps_per_episode,
                gamma_,
                optimizer_,
                paths
            )
            episode_reward = list(episode_reward.numpy())
            all_episodes_reward.append(episode_reward)
            window_episodes_reward.append(episode_reward)
            running_reward = tf.math.reduce_mean(window_episodes_reward, axis=0)

            # done_rate = compute_done_rate().numpy()
            # episodes_done_rate.append(done_rate)
            # running_done_rate = mean(episodes_done_rate)

            t.set_description(f'Episode {i}')
            t.set_postfix(
                # done_rate=done_rate,
                # running_done_rate=running_done_rate,
                episode_reward=episode_reward,
                running_reward=running_reward.numpy()
            )

            if i % saving_rhythm == 0:
                network.save("./SavedModel/Model")
                temp_arr = np.array(all_episodes_reward)
                temp_arr = np.mean(temp_arr, axis=1)
                np.save("../SavedModel/rewards.npy", temp_arr)
