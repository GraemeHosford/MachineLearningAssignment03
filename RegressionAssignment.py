import numpy as np
import pandas as pd


def task1():
    data = pd.read_csv("diamonds.csv")

    datapoints = data[["cut", "color", "clarity"]].drop_duplicates()

    features = list()
    target = list()

    for index, row in datapoints.iterrows():
        cut = row["cut"]
        color = row["color"]
        clarity = row["clarity"]

        select = (data["cut"] == cut) & (data["color"] == color) & (data["clarity"] == clarity)

        features.append(data[select][["carat", "depth", "table"]])
        target.append(data[select]["price"])

    print("\nBefore filtering features by number of datapoints each number of datapoints is shown here\n")

    for feature in features:
        print("The number of datapoints in this feature is", len(feature))

    useable_features = list(filter(lambda f: len(f) > 800, features))
    useable_targets = list(filter(lambda t: len(t) > 800, target))

    print("\nAfter filtering the feature subsets there is", len(useable_features), "left to be used\n")

    for feature in useable_features:
        print("The number of datapoints in these features with more than 800 datapoints is", len(feature))

    feature_dataframe = useable_features[0]
    target_dataframe = useable_targets[0]

    for x in range(1, len(useable_features)):
        feature_dataframe.append(useable_features[x])

    for x in range(1, len(useable_targets)):
        target_dataframe.append(useable_targets[x])

    return feature_dataframe.to_numpy(), target_dataframe.to_numpy()


def num_coefficients(deg: int) -> int:
    t = 0
    for n in range(deg + 1):
        for i in range(n + 1):
            for j in range(n + 1):
                if i + j == n:
                    t += 1
    return t


def task2(data: np.array, p: np.array, deg: int) -> np.array:
    result = np.zeros(data.shape[0])
    k = 0
    for n in range(deg + 1):
        for i in range(n + 1):
            result += p[k] * (data[:, 0] ** i) * (data[:, 1] ** (n - i))
            k += 1
    return result


def task3(deg: int, data: np.array, p0: np.array):
    f0 = task2(data, p0, deg)
    j = np.zeros((len(f0), len(p0)))
    epsilon = 1e-6
    for i in range(len(p0)):
        p0[i] += epsilon
        fi = task2(data, p0, deg)
        p0[i] -= epsilon
        di = (fi - f0) / epsilon
        j[:, i] = di
    return f0, j


def task4(y, f0, j):
    ep = 1e-2
    mat = np.matmul(j.T, j) + ep * np.eye(j.shape[1])
    r = y - f0
    n = np.matmul(j.T, r)
    dp = np.linalg.solve(mat, n)
    return dp


def main():
    features, target = task1()

    max_iter = 10

    for deg in range(5):
        p0 = np.zeros(num_coefficients(deg + 1))
        for i in range(max_iter):
            f0, j = task3(deg, features, p0)
            dp = task4(target, f0, j)
            p0 += dp


main()
