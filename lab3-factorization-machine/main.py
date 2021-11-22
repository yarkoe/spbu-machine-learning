import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from sklearn.model_selection import KFold

COMBINED_DATA_PATH_TEMPLATE = "dataset/combined_data_{}.txt"
PREPARED_DATA_PATH = "dataset/prepared_data.csv"
RESULT_DATA_PATH = "data_results.csv"

SPLIT_NUMBER = 5
FACT_DEGREE = 2
EPS = 1e-15
ERROR_EPS = 1.5
REGRESSION_STEPS = 50


def RMSE(y, yw):
    return np.sqrt((1 / y.shape[0]) * ((y - yw) ** 2).sum(axis=0))


def prepare_data():
    df = pd.read_csv(COMBINED_DATA_PATH_TEMPLATE.format(1), header=None, names=["user_id", "rating"], usecols=[0, 1])
    df['rating'] = df['rating'].astype(float)

    df_nan = pd.DataFrame(pd.isnull(df.rating))
    df_nan = df_nan[df_nan['rating'] == True]
    df_nan = df_nan.reset_index()

    movie_np = []
    movie_id = 1

    for i, j in zip(df_nan['index'][1:], df_nan['index'][:-1]):
        temp = np.full((1, i - j - 1), movie_id)
        movie_np = np.append(movie_np, temp)
        movie_id += 1

    last_record = np.full((1, len(df) - df_nan.iloc[-1, 0] - 1), movie_id)
    movie_np = np.append(movie_np, last_record)

    df = df[pd.notnull(df['rating'])]

    df['movie_id'] = movie_np.astype(int)
    df['user_id'] = df['user_id'].astype(int)

    df.to_csv(PREPARED_DATA_PATH, index=False)


def get_data():
    df = pd.read_csv(PREPARED_DATA_PATH, header=None, skiprows=1, names=["user_id", "rating", "movie_id"], dtype={'user_id': int, 'rating': float, 'movie_id': int}, )
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle dataset

    encoder = OneHotEncoder()
    one_hot_user_ids = encoder.fit_transform(np.asarray(df['user_id']).reshape(-1, 1))
    one_hot_movie_id = encoder.fit_transform(np.asarray(df['movie_id']).reshape(-1, 1))
    ratings = np.asarray(df['rating']).reshape(-1, 1)

    return hstack([one_hot_user_ids, one_hot_movie_id, np.ones(shape=(ratings.shape[0], 1))]), ratings


def calculate_new_Y(X, weights, V):
    return X * weights + 0.5 * ((X * V).power(2) - X.power(2) * V.power(2)).sum(axis=1)


def calculate_new_V(X, V, diff_Y, lamb):
    X_transposed = X.transpose()

    left = X.dot(V).multiply(diff_Y.reshape(-1, 1))
    left = X_transposed.dot(left)

    internal_copy_matrix = scipy.sparse.csr_matrix(X_transposed.power(2).dot(diff_Y))
    right = V.multiply(internal_copy_matrix)

    return V - (left - right) * lamb


def gradient_descend_step(X, Y, weights, V, step_count):
    lamb = np.math.log(step_count + 2)

    new_Y = calculate_new_Y(X, weights, V)
    internal_diff = -2 * (Y - new_Y) / Y.shape[0]
    new_weights = weights - X.transpose().dot(internal_diff) * lamb

    new_V = calculate_new_V(X, V, internal_diff, lamb)

    return new_weights, new_V


def linear_regression_fact_machine(X, Y):
    x_columns_count = X.shape[1]

    V = scipy.sparse.csr_matrix(np.random.normal(size=(x_columns_count, FACT_DEGREE))) * EPS
    weights = np.random.normal(size=(x_columns_count, 1)) * EPS

    for i in range(REGRESSION_STEPS):
        weights, V = gradient_descend_step(X, Y, weights, V, i)

        if RMSE(Y, calculate_new_Y(X, weights, V)) < ERROR_EPS:
            break

    return weights, V


def cross_validation(X, Y, fold_indices):
    weights = []
    V_matrices = []

    for _, (train_indices, _) in enumerate(fold_indices):
        X, Y = X[train_indices], Y[train_indices]

        weight, V = linear_regression_fact_machine(X, Y)
        weights.append(weight)
        V_matrices.append(V)

    return weights, V_matrices


def calculate_results_from_cross_validation(X, Y, weights, V_matrices, fold_indices):
    results = []

    for i, (train_indices, test_indices) in enumerate(fold_indices):
        train_X, train_Y = X[train_indices], Y[train_indices]
        test_X, test_Y = X[test_indices], Y[test_indices]

        train_RMSE = RMSE(train_Y, calculate_new_Y(train_Y, weights[i], V_matrices[i]))
        test_RMSE = RMSE(test_Y, calculate_new_Y(test_Y, weights[i], V_matrices[i]))

        results.append((train_RMSE, test_RMSE))

    return results


def create_data_results(results):
    data_results = pd.DataFrame(columns=[''] + ['T' + str(i + 1) for i in range(len(results))] + ['E', 'STD'])

    first_column = np.array(['RMSE-train', 'RMSE-test'])
    data_results[''] = first_column
    for i in range(len(results)):
        cur_column = np.concatenate(results[i])
        data_results['T' + str(i+1)] = cur_column

    mean = data_results.mean(axis=1)
    deviation = data_results.std(axis=1)
    data_results['E'] = mean
    data_results['STD'] = deviation

    return data_results


if __name__ == "__main__":
    data, ratings = get_data()

    fold_indices = KFold(n_splits=SPLIT_NUMBER).split(data)

    weights, V_matrices = cross_validation(data, ratings, fold_indices)
    results = calculate_results_from_cross_validation(data, ratings, weights, V_matrices, fold_indices)

    data_results = create_data_results(results)
    data_results.to_csv(RESULT_DATA_PATH)
