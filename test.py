import os
import random

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import GlobalObsForRailEnv
import PIL
from flatland.utils.rendertools import RenderTool
from IPython.display import clear_output
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.core.grid.grid4_utils import get_new_position
import numpy as np
import cv2
from datetime import datetime
from generate_matrix import get_overlap, get_next_pos
import pickle
from flatland.envs.step_utils import env_utils
from flatland.envs.rail_env_action import RailEnvActions


def render_env(env, wait=True):
    env_renderer = RenderTool(env, gl="PILSVG")
    env_renderer.render_env()

    image = env_renderer.get_image()
    pil_image = PIL.Image.fromarray(image)
    clear_output(wait=True)
    pil_image.show()


def get_remainng_path(env, agent):
    try:
        possible_transitions = env.rail.get_transitions(*agent.position, agent.direction)
    except:
        possible_transitions = env.rail.get_transitions(*agent.initial_position, agent.direction)

    path = []

    min_distance = np.inf
    cur_direction = agent.direction
    cur_position = agent.position

    while min_distance != 0:

        min_distance = np.inf
        for direction in [(cur_direction + i) % 4 for i in range(-1, 2)]:
            if possible_transitions[direction]:
                new_position = get_new_position(cur_position, direction)
                if min_distance > env.distance_map.get()[agent.handle, new_position[0], new_position[1], direction]:
                    min_distance = env.distance_map.get()[agent.handle, new_position[0], new_position[1], direction]
                    next_pos = new_position
                    next_direction = direction

        cur_direction = next_direction
        cur_position = next_pos
        path.append(cur_position)
        possible_transitions = env.rail.get_transitions(*cur_position, cur_direction)

    return path


def naive_solver(env, obs):
    actions = {}
    for idx, agent in enumerate(env.agents):
        try:
            possible_transitions = env.rail.get_transitions(*agent.position, agent.direction)
        except:
            possible_transitions = env.rail.get_transitions(*agent.initial_position, agent.direction)
        num_transitions = np.count_nonzero(possible_transitions)

        if num_transitions == 1:
            actions[idx] = 2
        else:
            min_distances = []
            for direction in [(agent.direction + i) % 4 for i in range(-1, 2)]:
                if possible_transitions[direction]:
                    new_position = get_new_position(agent.position, direction)
                    min_distances.append(env.distance_map.get()[idx, new_position[0], new_position[1], direction])
                else:
                    min_distances.append(np.inf)

            actions[idx] = np.argmin(min_distances) + 1

    return actions


def random_stop(env, actions):
    for agent in env.agents:
        i_agent = agent.handle
        if agent.position is not None:
            next_pos, _ = env_utils.apply_action_independent(RailEnvActions(actions[i_agent]),
                                                             env.rail,
                                                             agent.position,
                                                             agent.direction)

            transitions = env.rail.get_full_transitions(next_pos[0], next_pos[1])
            if bin(transitions).count("1") > 2 and (i_agent in env.movements_blocked or random.random() <= 0.01):
                if i_agent not in env.movements_blocked:  env.movements_blocked[i_agent] = 10
                actions[i_agent] = 4
                env.movements_blocked[i_agent] -= 1
                if env.movements_blocked[i_agent] == 0:
                    env.movements_blocked.pop(i_agent)


# Call reset() to initialize the environment
# observation, info = random_env.reset()

def process_data(step, env, actions, data, position_map, blocked_agents, contenders):
    for agent in env.agents:
        if MODE == 0 and agent.handle in blocked_agents:
            continue
        pos = agent.position
        coord = {"coords": pos, "cell_type": None if pos is None else env.rail.grid[pos[0]][pos[1]]}
        data[agent.handle]["trajectory"].append(coord)
        if pos == None:
            continue
    for contender in contenders:
        e1 = contender[0]
        agent1 = env.agents[e1]
        next_pos1, _ = env_utils.apply_action_independent(RailEnvActions(actions[e1]),
                                                          env.rail,
                                                          agent1.old_position,
                                                          agent1.old_direction)
            # for j, e2 in enumerate(position_map[contender[1]]):
        e2 = contender[1]
        agent2 = env.agents[e2]
        next_pos2, _ = env_utils.apply_action_independent(RailEnvActions(actions[e2]),
                                                          env.rail,
                                                          agent2.old_position,
                                                          agent2.old_direction)

        if next_pos2 != next_pos1: continue
        data[e1]["contentions"].append(
            {"e_train": {"id": e1, "dir": agent1.direction, "coord": agent1.old_position},
             "train": {"id": e2, "dir": agent2.direction, "coord": agent2.old_position},
             "time": step})

        data[e2]["contentions"].append(
            {"e_train": {"id": e2, "dir": agent2.direction, "coord": agent2.old_position},
             "train": {"id": e1, "dir": agent1.direction, "coord": agent1.old_position},
             "time": step})

    for pos in position_map.keys():
        for i, e in enumerate(position_map[pos]):
            if MODE == 0 and e in blocked_agents:
                continue
            e_agent = env.agents[e]
            next_pos, _ = env_utils.apply_action_independent(RailEnvActions(actions[e]),
                                                             env.rail,
                                                             e_agent.position,
                                                             e_agent.direction)

            if next_pos not in position_map:
                continue

            for j, o in enumerate(position_map[next_pos]):
                o_agent = env.agents[o]
                next_pos_o, _ = env_utils.apply_action_independent(RailEnvActions(actions[o]),
                                                                   env.rail,
                                                                   o_agent.position,
                                                                   o_agent.direction)
                if pos != next_pos_o: continue
                data[e]["interactions"].append(
                    {"e_train": {"id": e, "dir": e_agent.direction, "coord": e_agent.old_position},
                     "train": {"id": o, "dir": o_agent.direction, "coord": o_agent.old_position},
                     "time": step})
                if MODE == 0:
                    blocked_agents.append(e_agent.handle)
                    rem_path = get_remainng_path(env, e_agent)
                    for pos in rem_path:
                        coord = {"coords": pos, "cell_type": None if pos is None else env.rail.grid[pos[0]][pos[1]]}
                        data[e_agent.handle]["trajectory"].append(coord)



def run_episode(env):
    observations, info = env.reset()
    env_renderer = RenderTool(env)
    # env.agents[0].
    # env.agents[0].speed_counter.speed = 0.6
    # env.agents[4].speed_counter.speed = 0.8
    # env.agents[4].earliest_departure = 4
    # env.agents[6].earliest_departure = 8
    score = 0
    actions = dict()
    dirname = "data/" + str(datetime.now()) + "_seed_" + str(SEED)
    os.mkdir(dirname)
    os.mkdir(dirname + "/images")
    data = {}
    for agent in env.agents:
        data[agent.handle] = {"start_time": agent.earliest_departure, "start_coords": agent.initial_position,
                              "end_coords": agent.target, "interactions": [], "contentions": [], "trajectory": []}

    wait_dict = {}
    obs_list = [observations]
    blocked_agents = []
    for step in range(2000):
        actions = naive_solver(env, observations)
        # random_stop(env, actions)  # TODO: uncomment this line to add random stops at junctions
        position_map = {}
        contenders = []
        next_observations, all_rewards, dones, info = env.step(actions, position_map, contenders, MODE)
        obs_list.append(next_observations)
        for agent_handle in env.get_agent_handles():
            score += all_rewards[agent_handle]

        process_data(step, env, actions, data, position_map, blocked_agents, contenders)

        img = env_renderer.render_env(show=True,
                                      show_inactive_agents=False,
                                      show_predictions=True,
                                      show_observations=True,
                                      frames=True,
                                      show_rowcols=True,
                                      return_image=True)
        cv2.imwrite(dirname + "/images/" + str(step).zfill(4) + ".jpg", img)
        print('Timestep {}, total score = {}'.format(step, score))
        observations = next_observations
        if dones['__all__']:
            print('All done!')
            break

    with open(dirname + "/data.pickle", 'wb') as handle:
        pickle.dump(data, handle)
    with open(dirname + "/obs.pickle", 'wb') as handle:
        pickle.dump(obs_list, handle)

    print("Episode didn't finish after 50 timesteps.")


SEED = 12345
MODE = 1  # TODO: Change mode accordingly

if __name__ == "__main__":
    NUMBER_OF_AGENTS = 20
    width = 22
    height = 21
    NUM_CITIES = 2

    # Initialize the properties of the environment
    speed_dist = {1: 0.4, 0.5: 0.3, 0.3: 0.3}
    random_env = RailEnv(
        width=width,
        height=height,
        number_of_agents=NUMBER_OF_AGENTS,
        rail_generator=sparse_rail_generator(max_num_cities=NUM_CITIES,
                                             grid_mode=True,
                                             max_rails_between_cities=1,
                                             max_rail_pairs_in_city=2,
                                             seed=SEED),
        line_generator=sparse_line_generator(),
        obs_builder_object=GlobalObsForRailEnv(), random_seed=SEED
    )

    run_episode(random_env)
