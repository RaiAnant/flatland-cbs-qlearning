from flatland.envs.rail_generators import RailGen, rail_from_manual_specifications_generator
from flatland.envs.schedule_generators import BaseSchedGen
from flatland.envs.schedule_utils import Schedule


class SimpleSchedGen(BaseSchedGen):

    def generate(self, rail, num_agents, hints=None, num_resets=0, np_random=None):
        agent_positions = hints['agent_positions']
        agent_directions = hints['agent_directions']
        agent_targets = hints['agent_targets']
        max_episode_steps = hints['max_episode_steps']

        agents_speed = [1.0] * len(agent_positions)

        return Schedule(
            agent_positions=agent_positions,
            agent_directions=agent_directions,
            agent_targets=agent_targets,
            agent_speeds=agents_speed,
            agent_malfunction_rates=None,
            max_episode_steps=max_episode_steps
        )


class InitToInit(RailGen):

    def __init__(self, symmetrical):
        super().__init__()
        self.symmetrical = symmetrical

    def generate(self, width, height, num_agents, num_resets=0, np_random=None):
        # RAILS
        specs = [[(0, 0) for _ in range(width)] for _ in range(height)]

        specs[3][1] = (7, 270)
        specs[3][2] = (1, 90)
        specs[3][3] = (1, 90)
        specs[3][4] = (1, 90)
        specs[3][5] = (1, 90)
        specs[3][6] = (1, 90)
        specs[3][7] = (1, 90)
        specs[3][8] = (1, 90)
        specs[3][9] = (1, 90)
        specs[3][10] = (7, 90)

        # CITIES
        city_positions = [(3, 1), (3, 10)]
        city_orientations = [1, 3]
        train_stations = [
            [((3, 1), 0)],
            [((3, 10), 0)]
        ]

        # AGENTS
        # agent_positions = [(3, 2), (5, 7)]
        agent_positions = [(3, 2), (3, 7)]
        # agent_directions = [1, 0]
        agent_directions = [1, 3]
        agent_targets = [(3, 10), (3, 1)]

        # OTHER STUFF
        max_episode_steps = 20

        grid_map, _ = rail_from_manual_specifications_generator(specs)(None, None, None)
        hints = {'agents_hints': {
            'num_agents': num_agents,
            'city_positions': city_positions,
            'train_stations': train_stations,
            'city_orientations': city_orientations,
            'agent_positions': agent_positions,
            'agent_directions': agent_directions,
            'agent_targets': agent_targets,
            'max_episode_steps': max_episode_steps
        }}

        return grid_map, hints


class SwitchToSwitch(RailGen):

    def __init__(self, symmetrical):
        super().__init__()
        self.symmetrical = symmetrical

    def generate(self, width, height, num_agents, num_resets=0, np_random=None):
        # RAILS
        specs = [[(0, 0) for _ in range(width)] for _ in range(height)]
        # horizontal rails
        specs[3][1] = (7, 270)
        specs[3][2] = (1, 90)
        specs[3][3] = (1, 90)
        specs[3][4] = (1, 90)
        specs[3][5] = (1, 90)
        specs[3][6] = (1, 90)
        specs[3][7] = (1, 90)
        specs[3][8] = (1, 90)
        specs[3][9] = (1, 90)
        specs[3][10] = (7, 90)
        # vertical rails
        specs[0][4] = (7, 0)
        specs[1][4] = (1, 0)
        specs[2][4] = (1, 0)
        specs[4][7] = (1, 0)
        specs[5][7] = (1, 0)
        specs[6][7] = (7, 180)
        # switches
        specs[3][7] = (10, 90)
        specs[3][4] = (10, 270)

        # CITIES
        city_positions = [(3, 1), (3, 10)]
        city_orientations = [1, 3]
        train_stations = [
            [((3, 1), 0)],
            [((3, 10), 0)]
        ]

        # AGENTS
        if self.symmetrical:
            agent_positions = [(1, 4), (5, 7)]
            agent_directions = [2, 0]
            agent_targets = [(3, 10), (3, 1)]
        else:
            agent_positions = [(3, 2), (3, 9)]
            agent_directions = [1, 3]
            agent_targets = [(3, 10), (3, 1)]

        # OTHER STUFF
        max_episode_steps = 20

        grid_map, _ = rail_from_manual_specifications_generator(specs)(None, None, None)
        hints = {'agents_hints': {
            'num_agents': num_agents,
            'city_positions': city_positions,
            'train_stations': train_stations,
            'city_orientations': city_orientations,
            'agent_positions': agent_positions,
            'agent_directions': agent_directions,
            'agent_targets': agent_targets,
            'max_episode_steps': max_episode_steps
        }}

        return grid_map, hints

