import sys
from collections import deque
from typing import Tuple, List

from flatland.envs.agent_utils import RailAgentStatus
from flatland.utils.rendertools import RenderTool

from Utils import preprocess_observation, create_environment, all_paths_to_one_length
from conflict_based_search import cbs, get_agent_position, translate_path_into_actions
from q_learning_agent import deep_q_network

import tensorflow as tf
import numpy as np
import tqdm
import random


def env_step(actions: np.ndarray,
             pred_actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Runs an environment step.

    :param actions: Actions for each agent. Index of action corresponds to agent handle.
                    These actions are computed by the CBS algorithm.
    :param pred_actions: Actions for each agent. Index of action corresponds to agent handle.
                         These actions are predicted by the model.
    :return: New observation for each agent as feature matrices and adjacency matrices.
             These observations are a result from the actions given by the 'actions'-array.
             Thereby the i-th feature matrix goes along with the i-th adjacency matrix.
             Additionally, rewards and done-values for each agent are returned. Again, the indices
             correspond to the agent for which the value/matrix was computed.
    """

    # Check the shapes
    if len(actions.shape) > 1:
        actions = actions.reshape((actions.shape[-1],))
    if len(pred_actions.shape) > 1:
        pred_actions = pred_actions.reshape((pred_actions.shape[-1],))

    # Convert actions into a dictionary for the Flatland environment interface.
    action_dic = {}
    for agent_handle, action in enumerate(actions):
        action_dic[agent_handle] = action

    # Environment step
    observation, reward, done, _ = env.step(action_dic)
    feature_matrices, adj_matrices = preprocess_observation(observation)

    # Adjust rewards depending on whether the model predicted the same as the CBS
    # algorithm has computed.
    reward = np.array(list(reward.values()), dtype=np.float32)
    for idx, (a, p_a) in enumerate(zip(actions, pred_actions)):
        # Translate action into stop/go
        a = 1 if a in [1, 2, 3] else 0
        if a != p_a:
            reward[idx] -= 100
        else:
            reward[idx] += 100

    done = np.array(list(done.values())[:-1], dtype=np.int32)

    return feature_matrices, adj_matrices, reward, done


def tf_env_step(actions: np.ndarray, pred_actions: np.ndarray) -> List[tf.Tensor]:
    """Wraps the environment step in a Tensorflow operation.

    ---------------------------------------------------------------------
    It is not possible to pass tensors through this functions for which
    gradients shall be computed! A gradient tape cannot record operations
    for numpy arrays.

    See: https://www.tensorflow.org/api_docs/python/tf/numpy_function
    ---------------------------------------------------------------------

    :param actions: Actions for each agent. Index of action corresponds to agent handle.
                    These actions are computed by the CBS algorithm.
    :param pred_actions: Actions for each agent. Index of action corresponds to agent handle.
                         These actions are predicted by the model.
    :return: New observation for each agent as feature matrices and adjacency matrices.
             These observations are a result from the actions given by the 'actions'-array.
             Thereby the i-th feature matrix goes along with the i-th adjacency matrix.
             Additionally, rewards and done-values for each agent are returned. Again, the indices
             correspond to the agent for which the value/matrix was computed.
    """

    return tf.numpy_function(
        env_step,
        [actions, pred_actions],
        [tf.float32, tf.int32, tf.float32, tf.int32]
    )


def compute_target_values(q_values: np.ndarray,
                          actions: np.ndarray,
                          rewards: np.ndarray,
                          q_values_next_state: np.ndarray,
                          done_values: np.ndarray) -> np.ndarray:
    """Trains the model and updates the target model if necessary.

    :param q_values: The q-values predicted by the model for the initial observations.
    :param actions: The actions of the agents which led to the subsequent observations. The i-th
                    actions belongs to the agent with handle i.
    :param rewards: The rewards of the agents executing their actions. The i-th reward belongs to
                    the agent with handle i.
    :param q_values_next_state: The q-values predicted by the model for the subsequent observations.
    :param done_values: The done values of the agents. The i-th done value belongs to the agent with
                        handle i.
    :return: The target q-values for q-learning.
    """

    DISCOUNT_FACTOR = 0.9
    not_done_values = np.invert(done_values).astype(np.bool)

    # Predict the q values
    q_values[(not_done_values, 0, actions[not_done_values])] = \
        rewards[not_done_values] + DISCOUNT_FACTOR * np.max(
            q_values_next_state[not_done_values, 0], axis=1
        )
    q_values[(done_values, 0, actions[done_values])] = rewards[done_values]

    return q_values


def tf_compute_target_values(q_values: tf.Tensor,
                             actions: tf.Tensor,
                             rewards: tf.Tensor,
                             q_values_next_state: tf.Tensor,
                             done_values: np.ndarray) -> List[tf.Tensor]:
    """Wraps the computation of target q-values in a Tensorflow operation.

    ---------------------------------------------------------------------
    It is not possible to pass tensors through this functions for which
    gradients shall be computed! A gradient tape cannot record operations
    for numpy arrays.

    See: https://www.tensorflow.org/api_docs/python/tf/numpy_function
    ---------------------------------------------------------------------

    :param q_values: The q-values predicted by the model for the initial observations.
    :param actions: The actions of the agents which led to the subsequent observations. The i-th
                    actions belongs to the agent with handle i.
    :param rewards: The rewards of the agents executing their actions. The i-th reward belongs to
                    the agent with handle i.
    :param q_values_next_state: The q-values predicted by the model for the subsequent observations.
    :param done_values: The done values of the agents. The i-th done value belongs to the agent with
                        handle i.
    :return: The target q-values for q-learning.
    """

    return tf.numpy_function(
        compute_target_values,
        [q_values, actions, rewards, q_values_next_state, done_values],
        [tf.float32]
    )


@tf.function
def train_step(feature_matrices: tf.Tensor,
               adj_matrices: tf.Tensor,
               actions: tf.Tensor,
               rewards: tf.Tensor,
               next_feature_matrices: tf.Tensor,
               next_adj_matrices: tf.Tensor,
               done_values: tf.Tensor,
               model: tf.keras.Model,
               target_model: tf.keras.Model):
    """Executes one training step.

    :param feature_matrices: Sampled feature matrices from the replay memory.
    :param adj_matrices: Sampled adjacency matrices from the replay memory.
    :param actions: Sampled actions from the replay memory.
    :param rewards: Sampled rewards from the replay memory.
    :param next_feature_matrices: Sampled next states feature matrices from the replay memory.
    :param next_adj_matrices: Sampled next states adjacency matrices from the replay memory.
    :param done_values: Sampled done values from the replay memory.
    :param model: The neural network to be trained by the q-learning algorithm.
    :param target_model: The target network of the neural network to be trained.
    """

    q_values = model([feature_matrices, adj_matrices])
    next_q_values = target_model([next_feature_matrices, next_adj_matrices])

    q_values = tf_compute_target_values(
        q_values,
        actions,
        rewards,
        next_q_values,
        done_values
    )[0]

    # tf.print(tf.shape(q_values))

    with tf.GradientTape() as tape:
        predictions = model([feature_matrices, adj_matrices])
        loss = mse(q_values, predictions)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))


@tf.function
def tf_step(t_feature_matrices: tf.Tensor,
            t_adj_matrices: tf.Tensor,
            t_actions: tf.Tensor,
            t_rewards: tf.Tensor,
            t_next_feature_matrices: tf.Tensor,
            t_next_adj_matrices: tf.Tensor,
            t_done_values: tf.Tensor,
            o_feature_matrices: tf.Tensor,
            o_adj_matrices: tf.Tensor,
            o_cbs_actions: tf.Tensor,
            model: tf.keras.Model,
            target_model: tf.keras.Model,
            epsilon: tf.float32,
            amount_actions: tf.int32,
            training_flag: tf.bool) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Combines multiple smaller operations in a tensorflow function.

    Tensors with a 't_'-prefix are used for training the network. Tensors with a 'o_'-prefix
    represent the last observation of each agent. Those are used to compute the next action for
    each agent according to an epsilon-greedy strategy.

    :param t_feature_matrices: Sampled feature matrices from the replay memory.
    :param t_adj_matrices: Sampled adjacency matrices from the replay memory.
    :param t_actions: Sampled actions from the replay memory.
    :param t_rewards: Sampled rewards from the replay memory.
    :param t_next_feature_matrices: Sampled next states feature matrices from the replay memory.
    :param t_next_adj_matrices: Sampled next states adjacency matrices from the replay memory.
    :param t_done_values: Sampled done values from the replay memory.
    :param o_feature_matrices: Feature matrices of the latest observation from an agent.
    :param o_adj_matrices: Adjacency matrices of the latest observation from an agent.
    :param o_cbs_actions: The actions chosen by the CBS algorithm for the last observation
    :param model: The neural network to be trained by the q-learning algorithm.
    :param target_model: The target network of the neural network to be trained.
    :param epsilon: The latest epsilon value for the epsilon greedy strategy.
    :param amount_actions: The amount of actions the model can choose from.
    :param training_flag: Determines whether or not to train the network.
    :return: Returns new observations which follow by executing the actions from the
             'o_cbs_actions'-Tensor. The observations are described by actions, feature matrices,
             adjacency matrices, rewards and done values.
    """

    # Epsilon greedy strategy for action selection
    amount_agents = o_feature_matrices.shape[0]
    random_value = tf.random.uniform((), minval=0., maxval=1., dtype=tf.float32)
    o_actions = tf.cond(
        tf.less(epsilon, random_value),
        lambda: tf.cast(
            tf.squeeze(tf.math.argmax(model([o_feature_matrices, o_adj_matrices]), axis=2)),
            tf.int32
        ),
        lambda: tf.random.uniform(
            (amount_agents,),
            minval=0,
            maxval=amount_actions,
            dtype=tf.int32
        )
    )

    # Feed selected actions to environment and get new observation
    new_feature_matrices, new_adj_matrices, new_rewards, new_dones = tf_env_step(
        o_cbs_actions,
        o_actions
    )

    # Train the network
    if training_flag:
        train_step(
            t_feature_matrices,
            t_adj_matrices,
            t_actions,
            t_rewards,
            t_next_feature_matrices,
            t_next_adj_matrices,
            t_done_values,
            model,
            target_model
        )

    return o_actions, new_rewards, new_feature_matrices, new_adj_matrices, new_dones


def get_batch(batch_size=64, amt_nodes=21, amt_features=9) \
        -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Samples from the replay memory.

    :param batch_size: The amount of samples included in a batch.
    :param amt_nodes: The amount of nodes stored in the tree observation.
    :param amt_features: The amount of features stored by each node in the tree observation.
    :return: A batch of experiences and a flag that tells whether or not to train the network
             with this batch.
    """

    if len(replay_memory) < batch_size:
        f_matrices = tf.zeros((batch_size, amt_nodes, amt_features), dtype=tf.float32)
        a_matrices = tf.zeros((batch_size, amt_nodes, amt_nodes), dtype=tf.float32)
        actions = tf.zeros((batch_size,), dtype=tf.int32)
        rewards = tf.zeros((batch_size,), dtype=tf.float32)
        n_f_matrices = tf.zeros((batch_size, amt_nodes, amt_features), dtype=tf.float32)
        n_a_matrices = tf.zeros((batch_size, amt_nodes, amt_nodes), dtype=tf.float32)
        done_values = tf.zeros((batch_size,), dtype=tf.int32)
        training_flag = tf.constant(False, dtype=tf.bool)
    else:
        batch = np.array(random.sample(replay_memory, batch_size), dtype=object)
        f_matrices = tf.stack(batch[:, 0])
        a_matrices = tf.stack(batch[:, 1])
        actions = tf.stack(batch[:, 2])
        rewards = tf.stack(batch[:, 3])
        n_f_matrices = tf.stack(batch[:, 4])
        n_a_matrices = tf.stack(batch[:, 5])
        done_values = tf.stack(batch[:, 6])
        training_flag = tf.constant(True, dtype=tf.bool)

    return (f_matrices,
            a_matrices,
            actions,
            rewards,
            n_f_matrices,
            n_a_matrices,
            done_values,
            training_flag)


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


def determine_current_epsilon(time_step,
                              fifty_percent_chance=100,
                              start_epsilon=1.0,
                              min_epsilon=0.1):
    """Determines the current epsilon value.

    :param time_step: The current time step from the training procedure.
    :param fifty_percent_chance: The time step at which epsilon shall be 0.5.
    :param start_epsilon: The initial value for epsilon.
    :param min_epsilon: The minimal value for epsilon.
    :return: A probability for choosing a random action. (epsilon)
    """
    return max(
        min_epsilon,
        start_epsilon / (1 + (time_step / fifty_percent_chance))
    )


def remember(f_matrices: tf.Tensor,
             a_matrices: tf.Tensor,
             actions: tf.Tensor,
             rewards: tf.Tensor,
             next_f_matrices: tf.Tensor,
             next_a_matrices: tf.Tensor,
             done_values: tf.Tensor):
    """Stores new experiences in the replay memory.

    We define 'important cells' as cells which represent either the spawn of an agent, the target
    of an agent or a switch cell.

    The network shall only decide whether or not the agent shall stop or move. This only
    makes sense at important cells, since in other cases deadlocks cannot be prevented by stopping
    or moving an agent. Thus, the network is only needed at important cells and it makes no sense to
    use it anywhere else. Therefore, both first observations (f_matrices, a_matrices) and subsequent
    observations (next_f_matrices, next_a_matrices) have to be observations from important cells.

    ! The first observation (f_matrices, a_matrices) is assumed to be made at an important cell !

    The current position of each agent is now being checked on whether or not it is an important
    cell. If so, its observation (next_f_matrix, next_a_matrix) is used to make an entry within
    the replay memory. Subsequently, (next_f_matrix, next_a_matrix) replaces (f_matrix, a_matrix)
    for this agent. Thereby the new first observation will be an important cell again.

    :param f_matrices: Feature matrices of the first observations.
    :param a_matrices: Adjacency matrices of the first observations.
    :param actions: Actions of the agents, which were took upon the first observation.
    :param rewards: Rewards for the chosen action chosen at the first observation.
    :param next_f_matrices: Subsequent feature matrices for next observations.
    :param next_a_matrices: Subsequent adjacency matrices for next observations.
    :param done_values: Binary values that tell whether the agent is done or not.
    """

    important_cells = [is_at_important_cell(handle) for handle in env.get_agent_handles()]
    amount_agents = f_matrices.shape[0]
    f_matrices_list, a_matrices_list, update_actions_rewards = [], [], []

    for agent_handle in range(amount_agents):
        if important_cells[agent_handle]:
            replay_memory.append((
                f_matrices[agent_handle],
                a_matrices[agent_handle],
                actions[agent_handle],
                rewards[agent_handle],
                next_f_matrices[agent_handle],
                next_a_matrices[agent_handle],
                done_values[agent_handle]
            ))
            f_matrices_list.append(next_f_matrices[agent_handle])
            a_matrices_list.append(next_a_matrices[agent_handle])
            update_actions_rewards.append(True)
        else:
            f_matrices_list.append(f_matrices[agent_handle])
            a_matrices_list.append(a_matrices[agent_handle])
            update_actions_rewards.append(False)

    f_matrices_list = tf.stack(f_matrices_list)
    a_matrices_list = tf.stack(a_matrices_list)

    return f_matrices_list, a_matrices_list, update_actions_rewards


def run_episode(observation: dict,
                total_time_steps_passed: np.int64,
                verbose: bool = False) -> np.int64:
    """Runs an episode.

    :return: An updated total time step and the average reward from an agent gained in this episode.
    """

    # fo = first observation
    fo_f_matrices, fo_a_matrices = preprocess_observation(observation)
    fo_f_matrices, fo_a_matrices = tf.constant(fo_f_matrices), tf.constant(fo_a_matrices)
    agent_handles = env.get_agent_handles()
    fo_actions, fo_rewards = [-1 for _ in agent_handles], [-1 for _ in agent_handles]
    update_actions_rewards = [True for _ in agent_handles]

    all_rewards = []
    time_step = 0
    for time_step in range(1, max_steps_per_episode + 1):

        batch = get_batch()
        training_flag = batch[-1]
        batch = batch[:-1]

        epsilon = tf.constant(
            determine_current_epsilon(total_time_steps_passed + time_step),
            dtype=tf.float32
        )
        cbs_actions = tf.constant(translate_path_into_actions(env, paths_, time_step - 1))
        actions, rewards, f_matrices, a_matrices, new_dones = tf_step(
            *batch,
            fo_f_matrices,
            fo_a_matrices,
            cbs_actions,
            network,
            target_network,
            epsilon,
            tf.constant(amt_actions),
            training_flag
        )

        if verbose:
            env_renderer.render_env(
                show=True,
                frames=True,
                show_rowcols=True,
                show_observations=True,
                show_predictions=False,
                step=True
            )
            sys.stdout.write(
                f"\rTime Step: {time_step} / {max_steps_per_episode}"
                f" CBS actions: {cbs_actions}"
                f" Predicted actions: {actions}"
                f" Agent positions: {[get_agent_position(env, a) for a in env.get_agent_handles()]}"
                f" Done values: {new_dones.numpy()}"
            )

        # Within the entries of the replay memory, we want actions and rewards which were computed
        # upon the first observation. We do not care about actions and rewards which were made
        # between two important cells. As soon as the first observation for the i-th agent in
        # (fo_f_matrices, fo_a_matrices) changes we have a new 'first observation'. Then, an entry
        # with the action and reward for the old first observation was made. Subsequently, a new
        # action and a corresponding reward are computed for the new first observation in the
        # 'step'-function. Those replace the old action and reward.
        for handle, (update, action, reward) in enumerate(
                zip(update_actions_rewards, actions, rewards)
        ):
            if update:
                fo_actions[handle] = action
                all_rewards.append(fo_rewards[handle])
                fo_rewards[handle] = reward

        fo_f_matrices, fo_a_matrices, update_actions_rewards = remember(
            fo_f_matrices,
            fo_a_matrices,
            fo_actions,
            fo_rewards,
            f_matrices,
            a_matrices,
            new_dones
        )

        # if tf.math.reduce_all(tf.cast(new_dones, tf.bool)):
        #     break

        if [a.target for a in env.agents] == [get_agent_position(env, a) for a in env.get_agent_handles()]:
            break

    return total_time_steps_passed + time_step, np.sum(all_rewards) / len(agent_handles)


if __name__ == "__main__":

    # Miscellaneous parameters
    saving_rhythm = 10
    max_episodes = 100_000
    amt_actions = 2
    time_steps_passed = 0
    mse = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=.01)

    # Initialize networks
    network = deep_q_network(
        amt_nodes=21,
        amt_features=9,
        amt_actions=amt_actions
    )
    target_network = tf.keras.models.clone_model(network)

    # Initialize environment
    env, max_steps_per_episode = create_environment()
    env_renderer = RenderTool(env)

    # Initialize replay memory
    replay_memory = deque(maxlen=100_000)

    # For statistics
    all_episodes_reward = []
    window_episodes_reward = deque(maxlen=100)
    epsilon_values = []

    with tqdm.trange(max_episodes) as t:
        for i in t:
            obs, _ = env.reset()
            env_renderer.reset()
            solution = cbs(env, verbose=False)
            while not bool(solution):
                obs, _ = env.reset()
                env_renderer.reset()
                solution = cbs(env, verbose=False)
            paths_ = list(zip(*solution))[1]
            paths_ = all_paths_to_one_length(paths_)

            time_steps_passed, episode_reward = run_episode(obs, time_steps_passed, verbose=False)

            if episode_reward == -1:
                paths_ = list(zip(*solution))[1]
                paths_ = all_paths_to_one_length(paths_)

            # Collect training statistics
            all_episodes_reward.append(episode_reward)
            window_episodes_reward.append(episode_reward)
            running_reward = np.mean(window_episodes_reward)

            t.set_description(f'Episode {i}')
            t.set_postfix(
                episode_reward=episode_reward,
                running_reward=running_reward
            )

            if i % saving_rhythm == 0:
                # Update target network
                target_network.set_weights(network.get_weights())

                # Save statistics
                network.save("./SavedModel/Model")
                temp_arr = np.array(all_episodes_reward)
                np.save("./SavedModel/rewards.npy", temp_arr)
                temp_arr = np.array(epsilon_values)
                np.save("./SavedModel/epsilons.npy", temp_arr)
