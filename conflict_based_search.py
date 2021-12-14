from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.utils.rendertools import RenderTool

from advanced_rail_env import go_north, go_east, go_south, go_west, AdvancedRailEnv
from matplotlib import pyplot as plt

import numpy as np
import heapq
import sys
import networkx as nx
import random
import time


def get_path(node):
    """Returns a path for a given node from the A-star algorithm.

    :param node: A node from the A-star algorithm.
    :return: A path for an agent.
    """
    path = [node[2]]
    while node[4] is not None:
        path.append(node[4][2])
        node = node[4]
    return path[::-1]


def get_successors(graph, node_position, cardinal_direction):
    """Given a position and an orientation, compute the possible successor positions.

    :param graph: THe tree, in which the successors shall be computed.
    :param node_position: Position, for which possible successor positions shall be computed.
    :param cardinal_direction: The orientation, with which successor positions shall be computed.
    :return: Successor positions for a given position and orientation.
    """

    node = graph.nodes[node_position]
    if cardinal_direction == 0:
        possible_tr = node["n_dir_transitions"]
    elif cardinal_direction == 1:
        possible_tr = node["e_dir_transitions"]
    elif cardinal_direction == 2:
        possible_tr = node["s_dir_transitions"]
    else:
        possible_tr = node["w_dir_transitions"]

    successors = []
    for c_dir, (moving_dir, transition) in enumerate(
            zip([go_north, go_east, go_south, go_west], possible_tr)):
        if transition:
            successors.append((moving_dir(*node_position), c_dir))

    return successors


def a_star(graph, start_node, start_orientation, goal_position, constraints, verbose=True):
    """A modified A-star algorithm, which pays respect to constraints of the CBS algorithm.

    :param graph: The graph representation of the Flatland environment.
    :param start_node: The starting node of an agent.
    :param start_orientation: The starting orientation of an agent.
    :param goal_position: The target of an agent.
    :param constraints: Modified constraints (missing first dimension) ONLY for the agent for which
                        a solution shall be computed.
    :param verbose: Bool, which tells whether or not information shall be displayed.
    :return: If possible, a list of positions which describe the path (and therefore a solution)
             for the agent. If no solution is found, an empty list is returned.
    """
    # Used heuristic for f value
    heuristic = lambda row_1, col_1, row_2, col_2: np.linalg.norm(
        np.array([row_1, col_1]) - np.array([row_2, col_2])
    )

    # Initialize heap, a frontier that contains nodes which can be explored
    # Nodes have the following structure:
    # (f value, g value, position, orientation, predecessor node)
    # Elements are then sorted by their first entry. On draw the next entry is considered.
    frontier = [(0, 0, start_node, start_orientation, None)]
    IDX_G_VALUE = 1
    IDX_POSITION = 2
    IDX_ORIENTATION = 3

    # The closed list contains nodes, which already have been explored.
    # Do not explore nodes multiple times.
    closed_list = []

    add_new_nodes = True

    while bool(frontier):
        if verbose:
            sys.stdout.write(f"\rA-star algorithm: Length of frontier: {len(frontier)}")

        if len(frontier) == 100:
            add_new_nodes = False

        # Get node with lowest cost (shortest path)
        current_node = heapq.heappop(frontier)

        # Return path if the drawn node is at the goal position
        if current_node[IDX_POSITION] == goal_position:
            return get_path(current_node)

        # As current node will now be explored, add it to the closed list
        closed_list.append((current_node[IDX_POSITION], current_node[IDX_ORIENTATION]))

        successor_nodes = get_successors(
            graph, current_node[IDX_POSITION], current_node[IDX_ORIENTATION]
        )

        for suc_pos, suc_ori in successor_nodes:

            # Assuming all edges have a weight of 1
            suc_g_value = current_node[IDX_G_VALUE] + 1

            # Do not add explored nodes to the open list
            if (suc_pos, suc_ori) in closed_list:
                continue

            # Compute new entry
            if (suc_pos, suc_g_value) in constraints:
                # In case that suc_pos is occupied at a certain time step,
                # pause the agent for one step on its current position
                suc_f_value = suc_g_value + heuristic(*current_node[IDX_POSITION], *goal_position)
                new_entry = (
                    suc_f_value,
                    suc_g_value,
                    current_node[IDX_POSITION],
                    current_node[IDX_ORIENTATION],
                    current_node
                )
            else:
                suc_f_value = suc_g_value + heuristic(*suc_pos, *goal_position)
                new_entry = (suc_f_value, suc_g_value, suc_pos, suc_ori, current_node)

            # Check whether successor node is already in frontier
            if len(frontier) > 0:
                splitted_frontier = list(zip(*frontier))
                frontier_positions = splitted_frontier[IDX_POSITION]
                frontier_g_values = splitted_frontier[IDX_G_VALUE]
                node_already_in_frontier, lower_g_value, index = False, False, -1
                for idx, f_pos in enumerate(frontier_positions):
                    node_already_in_frontier = suc_pos == f_pos
                    lower_g_value = suc_g_value > frontier_g_values[idx]
                    if node_already_in_frontier and lower_g_value:
                        index = idx
                        break

                # If successor already is in frontier with a lower g value, skip the current successor
                if node_already_in_frontier and lower_g_value:
                    continue

                # If successor already is in frontier with a higher g value, replace the old node in
                # the frontier.
                if node_already_in_frontier:
                    frontier[index] = new_entry
                    heapq.heapify(frontier)
                    continue

            # If successor is not stored in the frontier, store it now.
            if add_new_nodes:
                heapq.heappush(frontier, new_entry)

    return []


def low_level(env, constraints, verbose=True):
    """Low level part of the CBS algorithm. Executes the A-star algorithm for each agent.

    :param env: The environment.
    :param constraints: A set of constraints.
    :param verbose: Bool, which tells whether or not information shall be displayed.
    :return: A list of solutions for each agent and the solution's cost. If there is an agent for
             which no solution can be found, the list will be empty.
    """
    all_solutions = []

    # Collect agent solutions w.r.t. their current constraints
    agent_handles = env.get_agent_handles()
    for handle in agent_handles:
        agent_constraints = [c[1:] for c in constraints if c[0] == handle]
        agent = env.agents[handle]
        agent_path = a_star(
            env.graph,
            agent.initial_position,
            agent.direction,
            agent.target,
            agent_constraints,
            verbose=verbose
        )
        # If there is an agent for which no solution can be found, than no overall solution is possible
        if not agent_path:
            return []
        all_solutions.append((len(agent_path), agent_path))

    return all_solutions


def find_stop_position(initial_path, starting_position):
    """Returns a suffix from a given path.

    :param initial_path: The path for which a suffix is wanted.
    :param starting_position: The starting position from the suffix.
    :return: A suffix for a given path.
    """

    if isinstance(initial_path[0][0], tuple):
        path = list(zip(*initial_path))[0]
    else:
        path = initial_path

    for idx, pos in enumerate(path):
        if pos == starting_position:
            return initial_path[:idx + 1]
    raise RuntimeError("Stop position not found in path!")


def get_restricted_area(path1, path2, restricted_pos1, restricted_pos2, time_step):
    """Computes the restricted area and the start- and end-time steps for both agents.

    * start time-step: The first time step where an agent occupies a position within the restricted
                       area.
    * end time-step: The last time step where an agent occupies a position with the restricted area

    :param path1: Path (previous solution) from the first agent.
    :param path2: Path (previous solution) from the second agent.
    :param restricted_pos1: The first position which agent one would occupy within the restricted
                            area.
    :param restricted_pos2: The first position which agent two would occupy within the restricted
                            area.
    :param time_step: The time step where the agents would collide.
    :return: The positions included within the restricted area, the start time steps for both agents
             and the end time steps for both agents.
    """

    sub_sequence1 = find_stop_position(path1[:time_step + 2][::-1], restricted_pos1)[::-1]
    sub_sequence2 = find_stop_position(path2[:time_step + 2][::-1], restricted_pos2)

    restricted_area = list(dict.fromkeys(sub_sequence1)) + list(dict.fromkeys(sub_sequence2))

    # Determine time step where agent enters restricted area
    fst_enter_r = find_stop_position(
        list(zip(path1, range(len(path1))))[:time_step + 2], restricted_pos1
    )[-1][1]
    snd_enter_r = find_stop_position(
        list(zip(path2, range(len(path2))))[:time_step + 2], restricted_pos2
    )[-1][1]
    start_time_steps = [fst_enter_r, snd_enter_r]

    # Determine how long the agent remains within the restricted area
    end_time_steps = []
    for path, r, enter in [
        (path1, restricted_area, fst_enter_r), (path2, restricted_area[::-1], snd_enter_r)
    ]:
        path_idx = 0
        for idx in range(len(restricted_area)):
            # Agent might wait in the restricted area because of other constraints
            while path_idx < len(path[enter:]) \
                    and path[enter:][path_idx] == path[enter:][path_idx - 1]:
                path_idx += 1
            # The last position of the agent is within the restricted area
            if path_idx >= len(path[enter:]) - 1:
                path_idx = len(path[enter:])
                break
            if path[enter:][path_idx] != r[idx]:
                break
            path_idx += 1
        end_time_steps.append(path_idx)

    end_time_steps[0] += start_time_steps[0]
    end_time_steps[1] += start_time_steps[1]

    return restricted_area, start_time_steps, end_time_steps


def get_restricted_position(graph, agent_path, time_step):
    """Traces back an agent path to its last important position.

    Important positions: Switch cell or spawn of the agent.

    :param graph: The graph representation of the environment.
    :param agent_path: The path (previous solution) of the agent.
    :param time_step: The time step of colliding with another agent.
    :return: The first position of the restricted area which the agent would enter.
    """

    assert len(agent_path) > 1, "The agent is already on its target position!"

    # Backtrack to last switch cell or the initial position of the agent
    restricted_position = None
    backwards = agent_path[:time_step + 1][::-1]
    amt_positions = 0
    current_position = backwards[0]
    for amt_positions, position in enumerate(backwards):
        if np.count_nonzero(graph.nodes[position]["all_transitions"]) > 2 \
                and position != current_position:
            restricted_position = position
            break

    if restricted_position is None:
        restricted_position = agent_path[1]
        amt_positions -= 1

    return restricted_position, amt_positions


def get_restricted_area_constraints(graph,
                                    fst_handle,
                                    fst_agent_path,
                                    snd_handle,
                                    snd_agent_path,
                                    time_step):
    """Computes the constraints for two agents, given they are colliding.

    :param graph: The graph representation of the environment.
    :param fst_handle: The handle of the first agent.
    :param fst_agent_path: The path (previous solution) of the first agent.
    :param snd_handle: The handle of the second agent.
    :param snd_agent_path: The path (previous solution) of the second agent.
    :param time_step: The time step when the agents collide.
    :return: A list of length two. The first entry contains the constraints for the first agent.
             The second entry contains the constraints for the second agent.
    """

    restricted_positions = []
    backtracking_steps = []
    for path in [fst_agent_path, snd_agent_path]:
        r_pos, b_steps = get_restricted_position(graph, path, time_step)
        restricted_positions.append(r_pos)
        backtracking_steps.append(b_steps)

    restricted_area, start_times, end_times = get_restricted_area(
        fst_agent_path,
        snd_agent_path,
        restricted_positions[0],
        restricted_positions[1],
        time_step
    )

    # Add constraints for the first agent
    is_on_spawn = 2 * int(fst_agent_path[0] != fst_agent_path[1] and restricted_positions[0] == fst_agent_path[1])
    new_constraints = [[
        (fst_handle, restricted_positions[0], ts) for ts in
        range(start_times[0], end_times[1] + is_on_spawn)
    ]]

    # Add constraints for the second agent
    is_on_spawn = 2 * int(snd_agent_path[0] != snd_agent_path[1] and restricted_positions[1] == snd_agent_path[1])
    new_constraints.append([
        (snd_handle, restricted_positions[1], ts) for ts in
        range(start_times[1], end_times[0] + is_on_spawn)
    ])

    return new_constraints


def agent_is_active(agent_path, time_step):
    """Checks whether or not an agent changes his position at any time in the future.

    :param agent_path: The agent's solution.
    :param time_step: The time step, from which one the path should be checked.
    :return: Whether or not the agent will move at any time in the future.
    """
    first_pos = agent_path[0]
    for next_pos in agent_path[1:time_step + 1]:
        if next_pos != first_pos:
            return True
    return False


def validate(env, all_solutions):
    """The validation part of the CBS algorithm.

    Checks whether or not a given solution (containing solutions for all agents) causes a deadlock
    between any pair of agents. Breaks immediatly if a collision is found between two agents.

    :param env: The environment.
    :param all_solutions: A solution for all agents given by the low level part of the CBS
                          algorithm.
    :return: Returns a list of length two if a collision is found
             (see 'get_restricted_area_constraints'). Returns an empty list if no problems could
             be identified.
    """
    new_constraints = []
    for fst_handle, (fst_cost, fst_agent_path) in enumerate(all_solutions[:-1]):
        for snd_handle, (snd_cost, snd_agent_path) in enumerate(all_solutions[fst_handle + 1:],
                                                                fst_handle + 1):
            min_length = min(fst_cost, snd_cost)
            for time_step in range(min_length):

                fst_current_pos = fst_agent_path[time_step]
                snd_current_pos = snd_agent_path[time_step]
                fst_agent_active = agent_is_active(fst_agent_path, time_step)
                snd_agent_active = agent_is_active(snd_agent_path, time_step)

                if time_step < min_length - 1:
                    fst_next_pos = fst_agent_path[time_step + 1]
                    snd_next_pos = snd_agent_path[time_step + 1]
                    fst_agent_active = agent_is_active(fst_agent_path, time_step + 1)
                    snd_agent_active = agent_is_active(snd_agent_path, time_step + 1)

                    if all([fst_agent_active, snd_agent_active]):
                        # Two agents cannot pass each other on the same railway
                        if fst_current_pos == snd_next_pos and fst_next_pos == snd_current_pos:
                            return get_restricted_area_constraints(
                                env.graph,
                                fst_handle,
                                fst_agent_path,
                                snd_handle,
                                snd_agent_path,
                                time_step
                            )

                        # Driving agent tries to pass a stopped agent on the same railway
                        if fst_next_pos == snd_current_pos and fst_next_pos == snd_next_pos:
                            return get_restricted_area_constraints(
                                env.graph,
                                fst_handle,
                                fst_agent_path,
                                snd_handle,
                                snd_agent_path,
                                time_step
                            )

                        # Driving agent tries to pass a stopped agent on the same railway
                        if snd_next_pos == fst_current_pos and snd_next_pos == fst_next_pos:
                            return get_restricted_area_constraints(
                                env.graph,
                                fst_handle,
                                fst_agent_path,
                                snd_handle,
                                snd_agent_path,
                                time_step
                            )

                # Two agents cannot occupy one position
                if fst_current_pos == snd_current_pos and all([fst_agent_active, snd_agent_active]):
                    new_constraints.append([(fst_handle, snd_current_pos, time_step)])
                    new_constraints.append([(snd_handle, fst_current_pos, time_step)])
                    return new_constraints

    return new_constraints


def visualize_ct_node(env, node):
    """Debugging tool: Visualizes nodes from the constraint tree as colored graphs.

    :param env: The environment.
    :param node: A node from the constraint tree to be visualized.
    """

    solution = node[1]
    constraints = node[2]

    agent_positions = [get_agent_position(env, handle) for handle in env.get_agent_handles()]
    agent_targets = [a.target for a in env.agents]

    color_map = []
    label_dict = {}
    for node in env.graph:
        label_dict[node] = ""
        color = "grey"

        for handle, path in enumerate(solution):
            if node in path[1]:
                label_dict[node] += f"{handle},"

        if node in agent_positions:
            label_dict[node] = agent_positions.index(node)
            color = "green"

        elif node in agent_targets:
            label_dict[node] = agent_targets.index(node)
            color = "yellow"

        for handle, pos, time_step in constraints:
            if node == pos:
                label_dict[node] = f"{handle, time_step}"
                if color == "grey":
                    color = "red"
                elif color == "yellow":
                    color = "black"
                elif color == "green":
                    color = "purple"
                break
        color_map.append(color)

    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw(
        env.graph,
        pos={node_key: node_key[::-1] for node_key in list(env.graph.nodes)},
        node_size=50,
        node_color=color_map,
        labels=label_dict,
        with_labels=True,
        font_size=10
    )
    plt.gca().invert_yaxis()
    plt.axis("on")
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.grid(True)
    plt.show()


def cbs(env, verbose=True):
    """Conflict based search for the Flatland MAPF-problem.

    Conflict based search as described in:

    @article{SHARON201540,
        title = {Conflict-based search for optimal multi-agent pathfinding},
        journal = {Artificial Intelligence},
        volume = {219},
        pages = {40-66},
        year = {2015},
        issn = {0004-3702},
        doi = {https://doi.org/10.1016/j.artint.2014.11.006},
        url = {https://www.sciencedirect.com/science/article/pii/S0004370214001386},
        author = {Guni Sharon and Roni Stern and Ariel Felner and Nathan R. Sturtevant},
        keywords = {Heuristic search, Multi-agent, Pathfinding},
    }

    :return: Returns a list of paths with their costs as a solution for a given environment.
             Returns an empty list if no solution could be found in time (Execution terminates
             after a given amount of tries).
    """

    open_ = []
    root_constraints = []
    root_solution = low_level(env=env, constraints=[], verbose=verbose)
    if not bool(root_solution):
        return []
    root_cost = max(root_solution)[0]

    # A node is a tuple described by the following entries: (cost, solution, constraints)
    IDX_SOLUTION = 1
    IDX_CONSTRAINTS = 2
    root_node = (root_cost, root_solution, root_constraints)
    open_.append(root_node)

    create_new_nodes = True

    while bool(open_):
        if verbose:
            sys.stdout.write(f"\rCBS: Amount of unvalidated constraint nodes: {len(open_)}")
        # if len(open_) >= 500:
        #     return []

        if len(open_) == 50:
            create_new_nodes = False

        # Choose node with lowest cost from open_
        node = heapq.heappop(open_)

        # visualize_ct_node(env, node)

        # If validation finds no conflicts, return solution
        node_constraints = validate(env, node[IDX_SOLUTION])
        if not bool(node_constraints):
            return node[IDX_SOLUTION]

        # If validation finds conflicts, add new constraints to constraint tree
        if create_new_nodes:
            for new_constraints in node_constraints:
                if len(new_constraints) > 0:
                    new_node_constraints = node[IDX_CONSTRAINTS].copy()
                    new_node_constraints.extend(new_constraints)
                    new_node_solution = low_level(
                        env=env,
                        constraints=new_node_constraints,
                        verbose=verbose
                    )
                    if not bool(new_node_solution):
                        continue
                    new_node_cost = max(new_node_solution)[0]
                    heapq.heappush(open_, (new_node_cost, new_node_solution, new_node_constraints))

    return []


def get_agent_position(env, handle):
    """Returns the agent position of agent referenced by 'handle'.

    :param env: The environment.
    :param handle: The handle of the agent for which the position shall be found.
    :return: The agent's position.
    """

    agent = env.agents[handle]
    if agent.status == RailAgentStatus.READY_TO_DEPART:
        agent_position = agent.initial_position
    elif agent.status == RailAgentStatus.ACTIVE:
        agent_position = agent.position
    elif agent.status == RailAgentStatus.DONE:
        agent_position = agent.target
    else:
        agent_position = agent.target

    return agent_position


def translate_path_into_actions(rail_env, pathways, time_step):
    """Translates the solution paths of the CBS algorithm into an action for a given time step.

    :param rail_env: The Flatland environment.
    :param pathways: The solution paths of the CBS algorithm.
    :param time_step: The time step for which an action is needed.
    :return: An array of actions.
    """

    acts = np.full(len(rail_env.get_agent_handles()), -1, dtype=np.int32)

    if time_step >= len(pathways[0]) - 1:
        return np.full(len(rail_env.get_agent_handles()), 4, dtype=np.int32)

    for agent_handle, pathway in enumerate(pathways):
        if np.all(pathway[time_step:] == pathway[time_step]):
            rail_env.agents[agent_handle].status = RailAgentStatus.DONE_REMOVED
            acts[agent_handle] = 4

    current_positions = pathways[:, time_step]
    next_positions = pathways[:, time_step + 1]
    differences = next_positions - current_positions

    for agent_handle, diff in enumerate(differences):

        if acts[agent_handle] != -1:
            continue

        # Do not activate agents if they are located in their 'spawn' and shall not move
        if rail_env.agents[agent_handle].status == RailAgentStatus.READY_TO_DEPART \
                and np.array_equal(diff, [0, 0]):
            acts[agent_handle] = 0
            continue
        # Activate agents otherwise
        elif rail_env.agents[agent_handle].status == RailAgentStatus.READY_TO_DEPART:
            rail_env.agents[agent_handle].position = rail_env.agents[agent_handle].initial_position
            rail_env.agents[agent_handle].status = RailAgentStatus.ACTIVE

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
            agent_orientation = rail_env.agents[agent_handle].direction
            action = (cardinal_dir_next_pos - agent_orientation + 2) % 4
            acts[agent_handle] = action

    return acts


# Check the cbs algorithm
# (currently buggy because of changes at a different location in the code)
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
    r_env = AdvancedRailEnv(
        width=width,
        height=height,
        number_of_agents=amount_agents,
        rail_generator=sparse_rail_generator(
            max_num_cities=amount_cities,
            seed=seed,
            grid_mode=False,
            max_rails_between_cities=2
        ),
        obs_builder_object=GlobalObsForRailEnv()
    )
    env_renderer = RenderTool(r_env)
    episode_no = 0
    while episode_no < max_episode_no:
        r_env.reset()
        env_renderer.reset()
        env_renderer.render_env(
            show=True,
            frames=True,
            show_rowcols=True,
            show_observations=True,
            show_predictions=False,
            step=True
        )
        solution = cbs(r_env, verbose=False)
        if not bool(solution):
            continue
        paths = list(zip(*solution))[1]
        episode_step = 0
        while episode_step < max_steps:
            time.sleep(.25)
            env_renderer.render_env(
                show=True,
                frames=True,
                show_rowcols=True,
                show_observations=True,
                show_predictions=False,
                step=True
            )
            episode_progress = int((episode_step + 1) / max_steps)
            actions = translate_path_into_actions(r_env, paths, episode_step)
            sys.stdout.write(f"\rEpisode: {episode_no} Time step: {episode_step} Chosen actions: {actions}")
            obs, rewards, done, info = r_env.step(actions)
            episode_step += 1
            if done["__all__"]:
                break
        episode_no += 1
