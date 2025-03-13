import numpy as np
import pandas as pd

TEST_DATA_SPLIT = 2000


def one_hot_encoding(y: np.ndarray) -> np.ndarray:
    one_hot_y = np.zeros((y.size, np.max(y) + 1))
    for one_hot_y_element, y_index in zip(one_hot_y, y):
        one_hot_y_element[y_index] = 1
    return one_hot_y


def get_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = pd.read_csv("data/train.csv").to_numpy()
    np.random.shuffle(data)

    train_data = data[TEST_DATA_SPLIT:].T
    test_data = data[:TEST_DATA_SPLIT].T

    x_train, y_train = train_data[1:], train_data[0]
    x_test, y_test = test_data[1:], test_data[0]

    y_train = one_hot_encoding(y_train)
    y_test = one_hot_encoding(y_test)

    return x_train, y_train, x_test, y_test


def main():
    x_train, y_train, x_test, y_test = get_data()


if __name__ == "__main__":
    main()
