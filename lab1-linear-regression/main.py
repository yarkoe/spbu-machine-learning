import pandas as pd
import numpy as np
from typing import List

DATASET_PATH = "dataset/dataset.csv"
FOLDS_NUMBER = 5
TARGET_COLUMN_NUMBER = 53
EPSILON = 0.1


def normalize_column(column: pd.Series):
    min_value = column.min()
    max_value = column.max()
    diff = max_value - min_value

    if diff == 0:
        return column.apply(lambda _: 1)

    return (column - min_value) / diff


def normalize(data: pd.DataFrame):
    for (column_name, column) in data.iteritems():
        if column_name != TARGET_COLUMN_NUMBER:
            data[column_name] = normalize_column(column)

    return data


def create_folds(data: pd.DataFrame, number) -> List[pd.DataFrame]:
    data = data.sample(frac=1).reset_index(drop=True)
    return np.array_split(data, number)


def get_gradient(matrix_x, column_y, weights):
    internal = column_y - np.dot(matrix_x, weights)
    internal = internal.transpose()
    return (-2 / matrix_x.shape[0]) * np.dot(internal, matrix_x)


def gradient_descend_step(matrix_x, column_y, weights, k):
    gradient = get_gradient(matrix_x, column_y, weights).transpose()
    norm = np.linalg.norm(gradient)
    gradient_dir = gradient / norm

    return weights - (1 / np.math.log(k + 2)) * gradient_dir


def length_between(v1, v2):
    return ((v1 - v2) ** 2).sum(axis=0) ** 0.5


def gradient_descend(matrix_x, column_y):
    cur_weights = np.random.rand(1, matrix_x.shape[1]).transpose()
    k = 1

    while True:
        prev_weights = cur_weights
        cur_weights = gradient_descend_step(matrix_x, column_y, cur_weights, k)
        k += 1
        if length_between(prev_weights, cur_weights) < EPSILON:
            break

    return cur_weights


def get_matrix_x_from_data(data: pd.DataFrame):
    non_target_matrix: pd.DataFrame = data.iloc[:, :-1]
    non_target_matrix['b'] = 1

    return non_target_matrix.values


def get_column_y_from_data(data: pd.DataFrame):
    return data.iloc[:, -1:].values


def linear_regression(data: pd.DataFrame):
    return gradient_descend(get_matrix_x_from_data(data), get_column_y_from_data(data))


def append_folds_without_index(index, folds):
    return pd.concat(folds[:index] + folds[index+1:])


def cross_validation(folds: List[pd.DataFrame]):
    weights = []

    for index in range(len(folds)):
        weight = linear_regression(append_folds_without_index(index, folds))
        weights.append(weight)

    return weights


def RMSE(y, yw):
    return np.sqrt((1 / y.size) * ((y - yw) ** 2).sum(axis=0))


def R2(y, yw):
    diff = y - yw
    return 1 - ((diff ** 2).sum(axis=0) / ((y - y.mean()) ** 2).sum(axis=0))


def calculate_deviation(deviation_fun, matrix_x, column_y, weights):
    return deviation_fun(column_y, matrix_x.dot(weights))


def calculate_results_by_fold(fold, weight):
    matrix_x = get_matrix_x_from_data(fold)
    column_y = get_column_y_from_data(fold)

    train_result_R2 = calculate_deviation(R2, matrix_x, column_y, weight)
    train_result_RMSE = calculate_deviation(RMSE, matrix_x, column_y, weight)

    return train_result_R2, train_result_RMSE


def calculate_results_from_cross_validation(folds, weights):
    results = list()

    for i in range(len(weights)):
        cur_weight = weights[i]
        train_fold = append_folds_without_index(i, folds)
        test_fold = folds[i]

        train_results_R2, train_result_RMSE = calculate_results_by_fold(train_fold, cur_weight)
        test_results_R2, test_result_RMSE = calculate_results_by_fold(test_fold, cur_weight)

        results.append((train_results_R2, train_result_RMSE, test_results_R2, test_result_RMSE))

    return results


def create_data_result(weights, results):
    data_results = pd.DataFrame(columns=[''] + ['T' + str(i+1) for i in range(len(results))] + ['E', 'STD'])

    first_column = np.array(['R2-train', 'RMSE-train', 'R2-test', 'RMSE-test'])
    for i in range(len(weights[0])):
        first_column = np.append(first_column, ['f' + str(i)])

    data_results[''] = first_column
    for i in range(len(results)):
        cur_column = np.concatenate(results[i])
        cur_column = np.append(cur_column, weights[i].transpose())
        data_results['T' + str(i+1)] = cur_column

    mean = data_results.mean(axis=1)
    deviation = data_results.std(axis=1)
    data_results['E'] = mean
    data_results['STD'] = deviation

    return data_results


def data_results_to_csv(name, data_results):
    data_results.to_csv(name)


def main():
    data = pd.read_csv(DATASET_PATH, header=None)

    normalize(data)
    folds = create_folds(data, FOLDS_NUMBER)
    print(folds)
    weights = cross_validation(folds)
    results = calculate_results_from_cross_validation(folds, weights)

    data_results = create_data_result(weights, results)
    data_results_to_csv("data_results.csv", data_results)


if __name__ == "__main__":
    main()
