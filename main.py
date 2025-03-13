import numpy as np
import pandas as pd

TEST_DATA_SPLIT = 2000


def get_data():
    data = pd.read_csv("data/train.csv").to_numpy()
    np.random.shuffle(data) 

    train_data = data[TEST_DATA_SPLIT:].T
    test_data = data[:TEST_DATA_SPLIT].T

    x_train, y_train = train_data[1:], train_data[0]
    x_test, y_test = test_data[1:], test_data[0]

    return x_train, y_train, x_test, y_test


def main():
    get_data()


if __name__ == "__main__":
    main()

