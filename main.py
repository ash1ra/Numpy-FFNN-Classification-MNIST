import numpy as np
import pandas as pd
from dataclasses import dataclass

TEST_DATA_SPLIT = 2000
HIDDEN_NEURONS_COUNT = 10
LEARNING_RATE = 0.01
EPOCHS = 100
BATCH_SIZE = 250
EPS = 1e-15


@dataclass
class Params:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray


@dataclass
class Grads:
    dw1: np.ndarray
    db1: np.ndarray
    dw2: np.ndarray
    db2: np.ndarray


def one_hot_encoding(y: np.ndarray) -> np.ndarray:
    one_hot_y = np.zeros((y.size, np.max(y) + 1))
    one_hot_y[np.arange(y.size), y] = 1
    return one_hot_y


def get_data() -> tuple[np.ndarray, ...]:
    data = pd.read_csv("data/train.csv").to_numpy()

    train_data = data[TEST_DATA_SPLIT:]
    test_data = data[:TEST_DATA_SPLIT]

    x_train, y_train = train_data[:, 1:] / 255, train_data[:, 0]
    x_test, y_test = test_data[:, 1:] / 255, test_data[:, 0]

    y_train = one_hot_encoding(y_train)
    y_test = one_hot_encoding(y_test)

    return x_train, y_train, x_test, y_test


def init_params() -> Params:
    w1 = np.random.randn(784, HIDDEN_NEURONS_COUNT) * np.sqrt(2 / 784)
    b1 = np.zeros((1, HIDDEN_NEURONS_COUNT))

    w2 = np.random.randn(HIDDEN_NEURONS_COUNT, 10) * np.sqrt(2 / HIDDEN_NEURONS_COUNT)
    b2 = np.zeros((1, 10))

    return Params(w1, b1, w2, b2)


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    return z > 0


def softmax(z: np.ndarray) -> np.ndarray:
    exps = np.exp(z - z.max(axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)


def forward_prop(x: np.ndarray, params: Params) -> tuple[np.ndarray, ...]:
    z1 = np.matmul(x, params.w1) + params.b1
    a1 = relu(z1)

    z2 = np.matmul(a1, params.w2) + params.b2
    a2 = softmax(z2)

    return z1, a1, a2


def cross_entropy(y: np.ndarray, y_pred: np.ndarray) -> np.float64:
    return (-np.sum(y * np.log(y_pred + EPS))) / y.shape[0]


def backward_prop(
    x: np.ndarray,
    y: np.ndarray,
    a2: np.ndarray,
    a1: np.ndarray,
    w2: np.ndarray,
    z1: np.ndarray,
) -> Grads:
    dz2 = a2 - y
    dw2 = np.matmul(dz2.T, a1) / BATCH_SIZE
    db2 = np.sum(dz2, axis=0, keepdims=True) / BATCH_SIZE

    da1 = np.matmul(w2, dz2.T)
    dz1 = da1.T * relu_derivative(z1)
    dw1 = np.matmul(x.T, dz1) / BATCH_SIZE
    db1 = np.sum(dz1, axis=0, keepdims=True) / BATCH_SIZE

    return Grads(dw1, db1, dw2, db2)


def gradient_descent(params: Params, grads: Grads) -> Params:
    params.w1 -= LEARNING_RATE * grads.dw1
    params.b1 -= LEARNING_RATE * grads.db1
    params.w2 -= LEARNING_RATE * grads.dw2
    params.b2 -= LEARNING_RATE * grads.db2

    return params


def calc_accuracy(y_true: np.ndarray, y_preds: np.ndarray) -> np.float64:
    true_labels = np.argmax(y_true, axis=1)
    predictions = np.argmax(y_preds, axis=1)
    accuracy = np.mean(true_labels == predictions)
    return accuracy


def train_model(x_train: np.ndarray, y_train: np.ndarray, params: Params) -> Params:
    train_samples_count = x_train.shape[0]

    for epoch in range(EPOCHS):
        perm = np.random.permutation(train_samples_count)
        x_train, y_train = x_train[perm], y_train[perm]

        start_idx = 0
        loss_per_epoch, accuracy_per_epoch = 0, 0
        num_batches = 0

        while start_idx < train_samples_count:
            end_idx = min(start_idx + BATCH_SIZE, train_samples_count)
            x_batch, y_batch = x_train[start_idx:end_idx], y_train[start_idx:end_idx]

            z1, a1, a2 = forward_prop(x_batch, params)

            loss_per_epoch += cross_entropy(y_batch, a2)
            accuracy_per_epoch += calc_accuracy(y_batch, a2)

            grads = backward_prop(x_batch, y_batch, a2, a1, params.w2, z1)
            params = gradient_descent(params, grads)

            start_idx = end_idx
            num_batches += 1

        loss_per_epoch /= num_batches
        accuracy_per_epoch /= num_batches

        print(
            f"Epoch: {epoch + 1} | Loss: {loss_per_epoch:.4f} | Accuracy: {(accuracy_per_epoch * 100):.2f}%"
        )

    return params


def test_model(x_test: np.ndarray, y_test: np.ndarray, params: Params) -> None:
    _, _, a2 = forward_prop(x_test, params)
    loss = cross_entropy(y_test, a2)
    accuracy = calc_accuracy(y_test, a2)

    print(f"Test loss: {loss:.4f} | Test accuracy: {(accuracy * 100):.2f}%")


def main() -> None:
    x_train, y_train, x_test, y_test = get_data()
    params = init_params()
    params = train_model(x_train, y_train, params)
    test_model(x_test, y_test, params)


if __name__ == "__main__":
    main()
