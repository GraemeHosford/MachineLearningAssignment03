from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


# Name: Graeme Hosford
# Student ID: R00147327

def task1() -> Tuple[np.array, np.array]:
    """ Task 1: Getting features and target as numpy arrays """
    print("Task 1 output")
    data = pd.read_csv("diamonds.csv")

    # Get feature datapoints and drop duplicates
    datapoints = data[["cut", "color", "clarity"]].drop_duplicates()

    features = list()
    target = list()

    # For each feature retrieved add to the features and target lists
    for index, row in datapoints.iterrows():
        cut = row["cut"]
        color = row["color"]
        clarity = row["clarity"]

        select = (data["cut"] == cut) & (data["color"] == color) & (data["clarity"] == clarity)

        features.append(data[select][["carat", "depth", "table"]])
        target.append(data[select]["price"])

    print("Before filtering features by number of datapoints each number of datapoints is shown here\n")

    # Show number of rows in each dataframe contained in features list
    for feature in features:
        print("The number of datapoints in this feature is", len(feature))

    # Get the features with 800+ rows
    useable_features = list(filter(lambda f: len(f) > 800, features))
    useable_targets = list(filter(lambda t: len(t) > 800, target))

    # Show how many dataframes had more than 800 rows
    print("\nAfter filtering the feature subsets there is", len(useable_features), "left to be used")

    # Show number of rows in each dataframe contained in useable_features list
    for feature in useable_features:
        print("The number of datapoints in these features with more than 800 datapoints is", len(feature))

    # Group list of features and targets into a single dataframe to make using it easier
    feature_dataframe = useable_features[0]
    target_dataframe = useable_targets[0]

    # Append all features together into feature_dataframe
    for x in range(1, len(useable_features)):
        feature_dataframe.append(useable_features[x])

    # Append all targets together in target_dataframe
    for x in range(1, len(useable_targets)):
        target_dataframe.append(useable_targets[x])

    # Convert dataframes to numpy arrays and return them
    feature_array = feature_dataframe.to_numpy()
    target_array = target_dataframe.to_numpy()

    # Show the contents of the feature and target arrays
    print("Contents of feature_array:\n", feature_array, "\n")
    print("Contents of target_array:\n", target_array, "\n")

    return feature_array, target_array


def num_coefficients(deg: int) -> int:
    # Find the number of coefficients in a 3 featured formula
    t = 0
    for n in range(deg + 1):
        for i in range(n + 1):
            for j in range(n + 1):
                for k in range(n + 1):
                    if i + j + k == n:
                        t += 1

    # Show number of coefficients for this degree
    print("Number of coefficients for degree %d = %d" % (deg, t), "\n")
    return t


def task2(data: np.array, p: np.array, deg: int) -> np.array:
    """ Task 2: Model Function """
    # Starting with zeroes, calculate a model function which describes the problem
    result = np.zeros(data.shape[0])
    k = 0
    for n in range(deg + 1):
        for i in range(n + 1):
            result += p[k] * (data[:, 0] ** i) * (data[:, 1] ** (n - i))
            k += 1

    # Show the result of calculating the model function
    print("Task 2 output for degree =", deg)
    print("Result of model function")
    print(result, "\n")
    return result


def task3(deg: int, data: np.array, p0: np.array) -> Tuple[np.array, np.array]:
    """ Task 3: Linearize """
    # Apply linearization to the model function generated in task 2
    f0 = task2(data, p0, deg)
    j = np.zeros((len(f0), len(p0)))
    epsilon = 1e-6
    for i in range(len(p0)):
        p0[i] += epsilon
        fi = task2(data, p0, deg)
        p0[i] -= epsilon
        di = (fi - f0) / epsilon
        j[:, i] = di

    # Show the results of linearization, showing the f0 array and Jacobian matrix
    print("Task 3 output for degree =", deg)
    print("Result of linearize function")
    print("f0 =", f0, "\n")
    print("j =", j, "\n")
    return f0, j


def task4(y: np.array, f0: np.array, j: np.array) -> np.array:
    """ Task 4: Calculate Update """
    # Calculate the update to apply to the p0 array in task 5
    ep = 1e-2

    # Multiply Jacobian matrix with its transpose then multiply by identity matrix
    mat = np.matmul(j.T, j) + ep * np.eye(j.shape[1])

    # Get difference between target y and f0 array
    r = y - f0

    # Multiply Jacobian transpose with r
    n = np.matmul(j.T, r)

    # Solve for update matrix
    dp = np.linalg.solve(mat, n)

    # Show the results of calculating the update
    print("Task 4 output")
    print("Result of calculating update", dp, "\n")
    return dp


def task5(deg: int, features: np.array, target: np.array) -> np.array:
    """ Task 5: Regression """
    p0 = np.zeros(num_coefficients(deg))
    # Starting from zeroes calculate the update for the model function
    # which should start to move toward a certain point
    for i in range(10):
        # Alternating between retrieving f0 and Jacobian with getting update
        f0, j = task3(deg, features, p0)
        dp = task4(target, f0, j)

        # Add update to p0
        p0 += dp

    # Show results of calculating update
    print("Task 5 output for degree =", deg)
    print("Result of regression", p0, "\n")
    return p0


def task6(features: np.array, target: np.array) -> int:
    """ Task 6: Model Selection """
    price_means = []

    for train_index, test_index in KFold(n_splits=5).split(features):
        # Not using train_index as regression does not use training in the same sense as other ML algorithms
        feature_test_data = features[test_index]
        target_test_data = target[test_index]

        mean_list = []

        # Find the degree which best describes the data
        for deg in range(4):
            # First get p0
            p0 = task5(deg, feature_test_data, target_test_data)

            # Get the mean absolute difference between p0 and target_test_data
            mean_diff = np.abs(np.mean(p0) - np.mean(target_test_data))

            # Add this mean difference to mean_list
            mean_list.append(mean_diff)

        # Add mean_list to price_means
        price_means.append(mean_list)

    lowest_mean = None
    best_deg = None

    # Of all degrees found get the one which has the lowest mean difference
    for means_list in price_means:
        if lowest_mean is None or np.mean(means_list) < lowest_mean:
            lowest_mean = np.mean(means_list)

            # Get the place in the price_means list where the lowest mean value can be found
            # This index will be equal to the degree to be used
            best_deg = price_means.index(means_list)

    # Show result of finding best degree to use
    print("Task 6 output")
    print("Result for getting best degree to use", best_deg, "\n")
    return best_deg


def task7(target: np.array, p0: np.array) -> None:
    """ Task 7: Visualize Results """
    # Show graph of actual prices plotted against predicted prices
    plt.title("Actual Diamond Prices vs. Estimated Prices")
    plt.xlabel("Estimated Diamond Prices ($)")
    plt.ylabel("Actual Diamond Prices ($)")
    plt.scatter(x=p0, y=target)

    # Straight line through origin to highlight correlation between actual and predicted prices
    plt.plot([0, np.max(target)], [0, np.max(p0)], c="r")
    plt.show()


def main():
    # Get features and target
    features, target = task1()

    # Find the best degree
    best_degree = task6(features, target)

    # Get the price predictions based on the degree found
    p = task5(best_degree, features, target)
    predictions = task2(features, p, best_degree)

    # Show the results of regression plotted against actual prices
    task7(target, predictions)


main()
