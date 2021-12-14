import argparse, pickle
import numpy as np


def get_next_pos(position, direction):
    dy = direction - 1 if direction % 2 == 0 else 0
    dx = 1 - (direction - 1) if direction % 2 == 1 else 0
    return position[0] + dy, position[1] + dx


def get_overlap(data, interaction):
    A = set([t['coords'] for t in data[interaction['e_train']['id']]['trajectory']])
    B = set([t['coords'] for t in data[interaction['train']['id']]['trajectory']])
    intersect = set.intersection(A, B)

    intersect.remove(None)
    intersect = list(intersect)
    ordered_intersect = []
    coords = [x['coords'] for x in data[interaction['e_train']['id']]['trajectory']]
    for t, c in enumerate(coords):
        if c in intersect:
            ordered_intersect.append(c)
            if c not in coords[t + 1:]:
                intersect.remove(c)
            intersect.sort(key=lambda tup: abs(tup[0] - c[0]) + abs(tup[1] - c[1]))
            t += 1
            while coords[t] in intersect:
                c2 = coords[t]
                if abs(c2[0] - ordered_intersect[-1][0]) + abs(c2[1] - ordered_intersect[-1][1]) == 1:
                    ordered_intersect.append(c2)
                if c2 not in coords[t + 1:]:
                    intersect.remove(c2)
                t += 1

            if interaction['train']['coord'] in ordered_intersect:
                break
            else:
                pass

    intersect = ordered_intersect
    for i in range(len(intersect) - 1):
        if abs(intersect[i][0] - intersect[i + 1][0]) + abs(intersect[i][1] - intersect[i + 1][1]) > 1:
            for j in range(i + 2, len(intersect)):
                if abs(intersect[i][0] - intersect[j][0]) + abs(intersect[i][1] - intersect[j][1]) <= 1:
                    temp = intersect[i + 1:j + 1]
                    temp.sort(key=lambda tup: -tup[1])
                    intersect = intersect[:i + 1] + temp + intersect[j + 1:]
                    i = j
                    break

    overlap = []
    flag = False
    for c in intersect:
        if len(overlap) == 0 or abs(overlap[-1][0] - c[0]) + abs(overlap[-1][1] - c[1]) <= 1:
            overlap.append(c)
            if c == interaction['train']['coord'] or c == interaction['e_train']['coord']:
                flag = True
        else:
            if flag:
                break
            overlap = [c]

    traj = [x['coords'] for x in data[interaction['e_train']['id']]['trajectory']]
    if traj.index(overlap[-1]) < traj.index(overlap[0]):
        overlap.reverse()

    return overlap


def get_wait_times(overlap, data, interaction):
    idx1 = 0
    while idx1 < len(overlap) and overlap[idx1] != interaction['e_train']['coord']:
        idx1 += 1
    if idx1 == len(overlap): return 0, 0

    flag = True
    if idx1 != 0 and get_next_pos(interaction['e_train']['coord'], interaction['e_train']['dir']) == overlap[idx1 - 1]:
        idx1 = len(overlap) - 1 - idx1
        flag = not flag

    idx2 = 0
    while idx2 < len(overlap) and overlap[idx2] != interaction['train']['coord']:
        idx2 += 1
    if idx2 == len(overlap): return 0, 0

    if idx2 != 0 and get_next_pos(interaction['train']['coord'], interaction['train']['dir']) == overlap[idx2 - 1]:
        idx2 = len(overlap) - 1 - idx2
        flag = not flag

    if flag:
        return 0, 0
    # assert not flag

    ewait = False
    owait = False

    for idx, c in enumerate(data[interaction['e_train']['id']]['trajectory']):
        if c['coords'] == data[interaction['e_train']['id']]['trajectory'][idx - 1] and c['coords'] == \
                interaction['e_train']['coord']:
            ewait = True
        if c['coords'] == interaction['e_train']['coord']:
            break

    for idx, c in enumerate(data[interaction['train']['id']]['trajectory']):
        if c['coords'] == data[interaction['train']['id']]['trajectory'][idx - 1] and c['coords'] == \
                interaction['train']['coord']:
            owait = True
        if c['coords'] == interaction['train']['coord']:
            break
    assert not (ewait and owait)

    if not ewait and not owait:
        return (idx1 + 1) * 2, (idx2 + 1) * 2
    elif ewait:
        return (idx1 + 1) * 2 + 1, idx2 * 2 + 1
    return idx1 * 2 + 1, (idx2 + 1) * 2 + 1


def generate_matrix(mode, path):
    data = pickle.load(
        open(path, "rb"))
    matrix = np.zeros((len(data.keys()), len(data.keys())))
    if mode == "boolean":
        for agent in data.keys():
            for interaction in data[agent]['interactions']:
                matrix[agent][interaction['train']['id']] = 1
    elif mode == "interactionTS":
        for agent in data.keys():
            for interaction in data[agent]['interactions']:
                matrix[agent][interaction['train']['id']] = interaction['time']
    elif mode == "startingTS":
        matrix = np.zeros((len(data.keys()), len(data.keys()), 2))
        for agent in data.keys():
            for interaction in data[agent]['interactions']:
                matrix[agent][interaction['train']['id']][0] = data[agent]['start_time']
                matrix[agent][interaction['train']['id']][1] = data[interaction['train']['id']]['start_time']
    elif mode == "overlapLength":
        for agent in data.keys():
            for interaction in data[agent]['interactions']:
                overlap = get_overlap(data, interaction)
                matrix[agent][interaction['train']['id']] = len(overlap)
    elif mode == "trafficLights":  # return
        matrix = np.zeros((len(data.keys()), len(data.keys()), 2))
        for agent in data.keys():
            for interaction in data[agent]['interactions']:
                overlap = get_overlap(data, interaction)
                matrix[agent][interaction['train']['id']][0] = overlap[0][0]
                matrix[agent][interaction['train']['id']][1] = overlap[0][1]
    elif mode == "waitTime":
        matrix = np.zeros((len(data.keys()), len(data.keys()), 2))
        for agent in data.keys():
            for interaction in data[agent]['interactions']:
                overlap = get_overlap(data, interaction)
                wt_time = get_wait_times(overlap, data, interaction)
                matrix[agent][interaction['train']['id']][0] = wt_time[0]
                matrix[agent][interaction['train']['id']][1] = wt_time[1]
    else:
        raise Exception

    return matrix


if __name__ == "__main__":
    PATH = "./"
    mode = "boolean"
    parser = argparse.ArgumentParser(description='Create a ArcHydro schema')
    parser.add_argument('--p', metavar='path', required=True, default="./",
                        help='path to data pickle')
    parser.add_argument('--m', metavar='mode', required=True)
    args = parser.parse_args()
    for mode in ["boolean", "interactionTS", "startingTS", "overlapLength", "trafficLights", "waitTime"]:
        mat = generate_matrix(mode)  # TODO change this part accordingly
        print(mode)
    # print(args['path'], args[])
