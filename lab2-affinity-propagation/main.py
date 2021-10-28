import numpy as np

EDGES_PATH = "dataset/Gowalla_edges.txt"
MAX_ROWS = 3000
MAX_ITERATIONS = 10


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
        graph[i].append(Node(i, i, -1., 0., 0.))
    for edge in edges:
        graph[edge[0]].append(Node(edge[0], edge[1], 1., 0., 0.))

    return graph


def get_maxs(row):
    max1, max2, max1_indx = -10000, -10000, -1
    for i in range(len(row)):
        if row[i] >= max1:
            max2 = max1
            max1 = row[i]
            max1_indx = i

    return max1, max2, max1_indx


def update_R_matrix(graph):
    for row in graph:
        a_s_row = [node.a_value + node.s_value for node in row]
        max1, max2, max1_indx = get_maxs(a_s_row)
        for i in range(len(row)):
            node = row[i]
            node.r_value = node.s_value - max1
            if i == max1_indx:
                node.r_value = node.s_value - max2


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
            node.a_value = min(0, calculate_r_sum_without_ind(r_columns[node.to], i) + graph[node.to][0].r_value)

    for k in range(len(graph)):
        diag_node = graph[k][0]
        diag_node.a_value = calculate_r_sum_without_ind(r_columns[k], -1)


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


if __name__ == "__main__":
    graph = load_graph()

    repres = do_affinity_propagation(graph, MAX_ITERATIONS)



    print("End")
