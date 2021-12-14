from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.utils.rendertools import RenderTool

from matplotlib import pyplot as plt

import networkx as nx
import numpy as np

go_north = lambda r, c: (r - 1, c)
go_east = lambda r, c: (r, c + 1)
go_south = lambda r, c: (r + 1, c)
go_west = lambda r, c: (r, c - 1)


def all_transitions(transition_info):
    """Compute possible directions, irrespective of cardinal direction

    Returns a binary list, telling in which direction an agent
    can go to without paying respect to its direction.

        [0: North, 1: East, 2: South, 3: West]

    :param transition_info: The uint16 binary string, describing the transitions of the
                            cell.
    """
    binary_string = bin(transition_info)[2:]
    binary_string = [0 for _ in range(16 - len(binary_string))] + list(binary_string)

    # Matrix, where rows tell in which direction you have to look and
    # columns tell in which direction you can go
    bin_representation = np.array(binary_string).reshape((4, 4))

    all_tr = [
        int("1" in bin_representation[:, 0]),  # checks NN EN SN WN
        int("1" in bin_representation[:, 1]),  # checks NE EE SE WE
        int("1" in bin_representation[:, 2]),  # checks NS ES SS WS
        int("1" in bin_representation[:, 3])   # checks NW EW SW WW
    ]

    return all_tr


class AdvancedRailEnv(RailEnv):
    """The usual rail environment extended by a graph representation of the map."""

    def __init__(self,
                 width,
                 height,
                 number_of_agents,
                 rail_generator,
                 obs_builder_object,
                 schedule_generator=None):

        if schedule_generator is None:
            super().__init__(
                width=width,
                height=height,
                number_of_agents=number_of_agents,
                rail_generator=rail_generator,
                obs_builder_object=obs_builder_object
            )
        else:
            super().__init__(
                width=width,
                height=height,
                number_of_agents=number_of_agents,
                rail_generator=rail_generator,
                obs_builder_object=obs_builder_object,
                schedule_generator=schedule_generator
            )

        self.graph: nx.Graph = None

    def reset(self,
              regenerate_rail: bool = True,
              regenerate_schedule: bool = True,
              activate_agents: bool = False,
              random_seed: bool = None):

        observation_dict, info_dict = super().reset(
            regenerate_rail,
            regenerate_schedule,
            activate_agents,
            random_seed
        )

        self.graph = nx.Graph()
        self.init_graph()

        return observation_dict, info_dict

    def init_graph(self):
        """Computes a tree representation of the current environment.

        The tree representation does not contain information about the
        agents.
        """

        # Find first cell
        env_height, env_width = self.height, self.width
        row, col = -1, -1
        for row in range(env_height):
            for col in range(env_width):
                if self.rail.get_full_transitions(row, col) != 0:
                    self.map_railway((row, col))
                    return

        if row == 0 and col == 0:
            raise RuntimeError("There is no map.")
        elif row == self.height - 1 and col == self.width - 1:
            raise RuntimeError("There is no railway.")

    def map_railway(self, position, prev_position=None):
        """Maps the given railway by going through every available branch of it.

        :param position: The position of the currently explored cell.
        :param prev_position: The position, from which one moved to the current position.
        """

        transition_info = self.rail.get_full_transitions(*position)
        if transition_info != 0:
            all_tr = all_transitions(transition_info)
            self.graph.add_node(
                position,
                all_transitions=all_tr,
                n_dir_transitions=self.rail.get_transitions(*position, 0),
                e_dir_transitions=self.rail.get_transitions(*position, 1),
                s_dir_transitions=self.rail.get_transitions(*position, 2),
                w_dir_transitions=self.rail.get_transitions(*position, 3)
            )
            if prev_position is not None:
                self.graph.add_edge(prev_position, position)

            for moving_dir, transition in zip([go_north, go_east, go_south, go_west], all_tr):
                if transition:
                    new_position = moving_dir(*position)
                    if new_position not in self.graph:
                        self.map_railway(new_position, position)
                    elif (position, new_position) not in self.graph.edges:
                        self.graph.add_edge(position, new_position)

    def is_switch(self, position):
        """Tells whether or not a given position is a switch cell.

        :param position: The position to be checked
        :return: Boolean value, whether or not a given position is a switch cell
        """
        return np.count_nonzero([int(x) for x in f"{self.rail.get_full_transitions(*position):b}"]) != 2

# Sanity Check
if __name__ == "__main__":
    rail_env = AdvancedRailEnv(
        width=24,
        height=24,
        number_of_agents=7,
        rail_generator=sparse_rail_generator(max_num_cities=2),
        obs_builder_object=GlobalObsForRailEnv()
    )
    env_renderer = RenderTool(rail_env)
    rail_env.reset()
    env_renderer.reset()
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
    print(rail_env.graph)
    fig, ax = plt.subplots()
    nx.draw(
        rail_env.graph,
        pos={node_key: node_key[::-1] for node_key in list(rail_env.graph.nodes)},
        node_size=50
    )
    plt.gca().invert_yaxis()
    plt.axis("on")
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.grid(True)
    plt.show()
    input("Press enter to end inspection.")
