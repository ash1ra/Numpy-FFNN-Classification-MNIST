import numpy as np
import pandas as pd

TEST_DATA_SPLIT = 2000
LEARNING_RATE = 0.01
EPOCHS = 100
BATCH_SIZE = 250
EPS = 1e-15


def one_hot_encoding(y: np.ndarray) -> np.ndarray:
    one_hot_y = np.zeros((y.size, np.max(y) + 1))
    for one_hot_y_element, y_index in zip(one_hot_y, y):
        one_hot_y_element[y_index] = 1
    return one_hot_y


def get_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = pd.read_csv("data/train.csv").to_numpy()
    np.random.shuffle(data)

    train_data = data[TEST_DATA_SPLIT:]
    test_data = data[:TEST_DATA_SPLIT]

    x_train, y_train = train_data[:, 1:] / 255, train_data[:, 0]
    x_test, y_test = test_data[:, 1:] / 255, test_data[:, 0]

    y_train = one_hot_encoding(y_train)
    y_test = one_hot_encoding(y_test)

    return x_train, y_train, x_test, y_test


def init_params() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w1 = np.random.randn(784, 10) * np.sqrt(2 / 784)
    b1 = np.zeros((1, 10))

    w2 = np.random.randn(10, 10) * np.sqrt(2 / 10)
    b2 = np.zeros((1, 10))

    return w1, b1, w2, b2


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    return z > 0


def softmax(z: np.ndarray) -> np.ndarray:
    exps = np.exp(z - z.max(axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)


def forward_prop(
    x: np.ndarray, params: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w1, b1, w2, b2 = params

    z1 = np.matmul(x, w1) + b1
    a1 = relu(z1)

    z2 = np.matmul(a1, w2) + b2
    a2 = softmax(z2)

    return z1, a1, z2, a2


def cross_entropy(y: np.ndarray, y_pred: np.ndarray) -> np.float64:
    return (-np.sum(y * np.log(y_pred + EPS))) / y.shape[0]


def backward_prop(x, y, a2, a1, w2, z1):
    dz2 = a2 - y
    dw2 = np.matmul(dz2.T, a1) / BATCH_SIZE
    db2 = np.sum(dz2, axis=0, keepdims=True) / BATCH_SIZE

    da1 = np.matmul(w2, dz2.T)
    dz1 = da1.T * relu_derivative(z1)
    dw1 = np.matmul(x.T, dz1) / BATCH_SIZE
    db1 = np.sum(dz1, axis=0, keepdims=True) / BATCH_SIZE

    return dw1, db1, dw2, db2


def gradient_descent(w1, b1, w2, b2, dw1, db1, dw2, db2):
    w1 -= LEARNING_RATE * dw1
    b1 -= LEARNING_RATE * db1
    w2 -= LEARNING_RATE * dw2
    b2 -= LEARNING_RATE * db2

    return w1, b1, w2, b2


def calc_accuracy(y_true: np.ndarray, y_preds: np.ndarray) -> np.float64:
    true_labels = np.argmax(y_true, axis=1)
    predictions = np.argmax(y_preds, axis=1)
    accuracy = np.mean(true_labels == predictions)
    return accuracy


def train_model(x_train, y_train, w1, b1, w2, b2):
    TOTAL_TRAIN_SAMPLES = x_train.shape[0]
    for epoch in range(EPOCHS):
        start_idx, end_idx = 0, BATCH_SIZE
        loss_per_epoch, accuracy_per_epoch = 0, 0
        num_batches = 0
        while end_idx < TOTAL_TRAIN_SAMPLES:
            z1, a1, z2, a2 = forward_prop(
                x_train[start_idx:end_idx, :], (w1, b1, w2, b2)
            )

            loss_per_epoch += cross_entropy(y_train[start_idx:end_idx, :], a2)
            accuracy_per_epoch += calc_accuracy(y_train[start_idx:end_idx, :], a2)

            dw1, db1, dw2, db2 = backward_prop(
                x_train[start_idx:end_idx, :],
                y_train[start_idx:end_idx, :],
                a2,
                a1,
                w2,
                z1,
            )
            w1, b1, w2, b2 = gradient_descent(w1, b1, w2, b2, dw1, db1, dw2, db2)

            start_idx += BATCH_SIZE
            end_idx = min(end_idx + BATCH_SIZE, TOTAL_TRAIN_SAMPLES)
            num_batches += 1

        loss_per_epoch /= num_batches
        accuracy_per_epoch /= num_batches

        print(
            f"Epoch: {epoch + 1} | Loss: {loss_per_epoch:.4f} | Accuracy: {(accuracy_per_epoch * 100):.2f}%"
        )

    return w1, b1, w2, b2


def test_model(x_test, y_test, w1, b1, w2, b2):
    _, _, _, a2 = forward_prop(x_test, (w1, b1, w2, b2))
    loss = cross_entropy(y_test, a2)
    accuracy = calc_accuracy(y_test, a2)

    print(f"Test loss: {loss:.4f} | Test accuracy: {(accuracy * 100):.2f}%")


def main():
    x_train, y_train, x_test, y_test = get_data()
    w1, b1, w2, b2 = init_params()
    w1, b1, w2, b2 = train_model(x_train, y_train, w1, b1, w2, b2)
    test_model(x_test, y_test, w1, b1, w2, b2)


if __name__ == "__main__":
    main()
