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


def init_params() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w1 = np.random.rand(10, 784)
    b1 = np.random.rand(10, 1)

    w2 = np.random.rand(10, 10)
    b2 = np.random.rand(10, 1)

    return w1, b1, w2, b2


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    return z >= 0


# NOTE: change axis to 1 for using batches
def softmax(z: np.ndarray) -> np.ndarray:
    exps = np.exp(z - z.max(axis=0, keepdims=True))
    return exps / np.sum(exps, axis=0, keepdims=True)


def softmax_derivative(z: np.ndarray) -> np.ndarray: ...


def forward_prop(
    x: np.ndarray, params: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    w1, b1, w2, b2 = params

    z1 = np.matmul(w1, x) + b1
    a1 = relu(z1)

    z2 = np.matmul(w2, a1) + b2
    a2 = softmax(z2)

    return a2


def cross_entropy(y: np.ndarray, y_pred: np.ndarray):
    return -np.sum(y * np.log(y_pred))


def main():
    x_train, y_train, x_test, y_test = get_data()
    w1, b1, w2, b2 = init_params()
    a2 = forward_prop(x_train, (w1, b1, w2, b2))
    loss = cross_entropy(y_train, a2)
    print(loss)


if __name__ == "__main__":
    main()
