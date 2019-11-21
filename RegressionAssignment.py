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

    useable_features = list(filter(lambda x: len(x) > 800, features))
    useable_targets = list(filter(lambda x: len(x) > 800, target))

    print("\nAfter filtering the feature subsets there is", len(useable_features), "left to be used\n")

    for feature in useable_features:
        print("The number of datapoints in these features with more than 800 datapoints is", len(feature))

    return useable_features, useable_targets


def task2(features: pd.DataFrame, coff: np.array, deg: int):
    print()


def main():
    features, target = task1()

main()
