import random

import numpy as np

EDGES_PATH = "dataset/Gowalla_edges.txt"
CHECKINS_PATH = "dataset/Gowalla_totalCheckins.txt"
MAX_ROWS = 3000
MAX_ITERATIONS = 5
TEST_CHECKINS_PERCENT = 10
SMOOTHING_COEFFICIENT = 0.6
HIGH_UNIFORM = 1e-4
TOP_LOCATIONS_COUNT = 10
TEST_USER_COUNT = 2000


class Node:
    def __init__(self, frm, to, init_s_value, init_r_value, init_a_value):
        self.frm = frm
        self.to = to
        self.s_value = init_s_value
        self.r_value = init_r_value
        self.a_value = init_a_value

    def __str__(self):
        return "({}, {}, s:{}, r:{}, a:{})".format(self.frm, self.to, self.s_value, self.r_value, self.a_value)


def load_graph():
    edges = np.loadtxt(EDGES_PATH, delimiter="	", dtype=int)

    user_count = max(np.unique(edges)) + 1

    graph = [[] for _ in range(user_count)]
    # diag for first node
    for i in range(user_count):
        graph[i].append(Node(i, i, -1.5 - np.random.uniform(0., HIGH_UNIFORM), 0., 0.))
    for edge in edges:
        graph[edge[0]].append(Node(edge[0], edge[1], 1. + np.random.uniform(0., HIGH_UNIFORM), 0., 0.))

    return graph


def get_maxs(row):
    max1, max2, max1_indx = -10000, -10000, -1
    for i in range(len(row)):
        if row[i] >= max1:
            max2 = max1
            max1 = row[i]
            max1_indx = i

    return max1, max2, max1_indx


def smooth(x, y):
    return x * SMOOTHING_COEFFICIENT + (1 - SMOOTHING_COEFFICIENT) * y


def update_R_matrix(graph):
    for row in graph:
        a_s_row = [node.a_value + node.s_value for node in row]
        max1, max2, max1_indx = get_maxs(a_s_row)
        for i in range(len(row)):
            node = row[i]
            node.r_value = smooth(node.r_value, node.s_value - max1)
            if i == max1_indx:
                node.r_value = smooth(node.r_value, node.s_value - max2)


def get_r_columns_greater_than_0(graph):
    r_columns = [[] for _ in range(len(graph))]
    for i in range(len(graph)):
        for k in range(1, len(graph[i])):
            node = graph[i][k]
            if node.r_value > 0.:
                r_columns[node.to].append(node)

    return r_columns


def calculate_r_sum_without_ind(column, index):
    sum = 0
    for node in column:
        if node.to != index:
            sum += node.r_value

    return sum


def update_A_matrix(graph):
    r_columns = get_r_columns_greater_than_0(graph)

    for i in range(len(graph)):
        for k in range(1, len(graph[i])):
            node = graph[i][k]
            node.a_value = smooth(node.a_value, min(0, calculate_r_sum_without_ind(r_columns[node.to], i) + graph[node.to][0].r_value))

    for k in range(len(graph)):
        diag_node = graph[k][0]
        diag_node.a_value = smooth(diag_node.a_value, calculate_r_sum_without_ind(r_columns[k], -1))


def calculate_arg_max_ar_sum(row):
    max_ind = row[0].to
    max = row[0].a_value + row[0].r_value
    for i in range(1, len(row)):
        node = row[i]
        ar_sum = node.a_value + node.r_value
        if ar_sum > max:
            max = ar_sum
            max_ind = node.to

    return max_ind


def create_repres(graph):
    repres = []

    for i in range(len(graph)):
        repres.append(calculate_arg_max_ar_sum(graph[i]))

    return repres


def do_affinity_propagation(graph, max_iterations):
    for _ in range(max_iterations):
        update_R_matrix(graph)
        update_A_matrix(graph)

    return create_repres(graph)


def load_checkins():
    return np.loadtxt(CHECKINS_PATH, usecols=(0, 4), dtype=int)


def get_test_checkins_ids(max_id):
    return np.random.choice(max_id,
                                TEST_USER_COUNT,
                                replace=False)


def generate_user_top_locations(checkins, user):
    user_locations = {}
    for checkin in checkins:
        if checkin[0] != user:
            continue

        if checkin[1] not in user_locations:
            user_locations[checkin[1]] = 0
        user_locations[checkin[1]] += 1

    top = sorted(user_locations.items(), key=lambda item: item[1])[-TOP_LOCATIONS_COUNT:]
    top_user_locations = [t[0] for t in top]

    return top_user_locations


def generate_repr_top_locations(checkins, repr, repres):
    repr_locations = {}
    for checkin in checkins:
        if repres[checkin[0]] != repr:
            continue

        if checkin[1] not in repr_locations:
            repr_locations[checkin[1]] = 0
        repr_locations[checkin[1]] += 1

    top = sorted(repr_locations.items(), key=lambda item: item[1])[-TOP_LOCATIONS_COUNT:]
    top_repr_locations = [t[0] for t in top]

    return top_repr_locations


def calculate_top_locations(repres, checkins, test_checkins_ids):
    top_10_repr_locations_dic = {}
    results = []
    for test_checkins_id in test_checkins_ids:
        if test_checkins_id == repres[test_checkins_id]:
            results.append(0)
            continue

        top_10_user_locations = set(generate_user_top_locations(checkins, test_checkins_id))

        repr = repres[test_checkins_id]
        if repr not in top_10_repr_locations_dic:
            top_10_repr_locations_dic[repr] = set(generate_repr_top_locations(checkins, repres[test_checkins_id], repres))
        results.append(len(top_10_repr_locations_dic[repr].intersection(top_10_user_locations)))

    return results


def print_results(results):
    print(results)

    E = sum([result / float(len(results)) for result in results])
    print("Average: {}".format(E))
    D = sum([(result - E) ** 2 / float(len(results)) for result in results]) ** 0.5
    print("Dispersion: {}".format(D))


if __name__ == "__main__":
    graph = load_graph()
    repres = do_affinity_propagation(graph, MAX_ITERATIONS)

    checkins = load_checkins()
    test_checkins_ids = get_test_checkins_ids(checkins.max(axis=0)[0])

    results = calculate_top_locations(repres, checkins, test_checkins_ids)
    print_results(results)
