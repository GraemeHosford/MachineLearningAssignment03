import pandas as pd


def task1():
    data = pd.read_csv("diamonds.csv")

    cuts = data["cut"]
    color = data["color"]
    clarity = data["clarity"]

    datapoints = cuts.to_frame().join(color)
    datapoints = datapoints.join(clarity)

    print(datapoints.drop_duplicates())


def main():
    task1()


main()
