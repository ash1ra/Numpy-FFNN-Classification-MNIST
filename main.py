import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field

TEST_DATA_SPLIT = 2000
EPS = 1e-15
LOGGER_LEVEL = logging.INFO


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


@dataclass
class ModelData:
    train_loss: list = field(default_factory=list)
    train_accuracy: list = field(default_factory=list)
    test_loss: np.float64 = field(default_factory=np.float64)
    test_accuracy: np.float64 = field(default_factory=np.float64)


logger = logging.getLogger(__name__)
logger.setLevel(LOGGER_LEVEL)
console_handler = logging.StreamHandler()
console_handler.setLevel(LOGGER_LEVEL)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


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


def init_params(hidden_neurons_count: int) -> Params:
    w1 = np.random.randn(784, hidden_neurons_count) * np.sqrt(2 / 784)
    b1 = np.zeros((1, hidden_neurons_count))

    w2 = np.random.randn(hidden_neurons_count, 10) * np.sqrt(2 / hidden_neurons_count)
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
    batch_size: int,
) -> Grads:
    dz2 = a2 - y
    dw2 = np.matmul(dz2.T, a1) / batch_size
    db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size

    da1 = np.matmul(w2, dz2.T)
    dz1 = da1.T * relu_derivative(z1)
    dw1 = np.matmul(x.T, dz1) / batch_size
    db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size

    return Grads(dw1, db1, dw2, db2)


def gradient_descent(params: Params, grads: Grads, learning_rate: float) -> Params:
    params.w1 -= learning_rate * grads.dw1
    params.b1 -= learning_rate * grads.db1
    params.w2 -= learning_rate * grads.dw2
    params.b2 -= learning_rate * grads.db2

    return params


def calc_accuracy(y_true: np.ndarray, y_preds: np.ndarray) -> np.float64:
    true_labels = np.argmax(y_true, axis=1)
    predictions = np.argmax(y_preds, axis=1)
    accuracy = np.mean(true_labels == predictions)
    return accuracy


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    params: Params,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[Params, ModelData]:
    model_data = ModelData()
    train_samples_count = x_train.shape[0]

    for epoch in range(epochs):
        perm = np.random.permutation(train_samples_count)
        x_train, y_train = x_train[perm], y_train[perm]

        start_idx = 0
        loss_per_epoch, accuracy_per_epoch = 0, 0
        num_batches = 0

        while start_idx < train_samples_count:
            end_idx = min(start_idx + batch_size, train_samples_count)
            x_batch, y_batch = x_train[start_idx:end_idx], y_train[start_idx:end_idx]

            z1, a1, a2 = forward_prop(x_batch, params)

            loss_per_epoch += cross_entropy(y_batch, a2)
            accuracy_per_epoch += calc_accuracy(y_batch, a2)

            grads = backward_prop(x_batch, y_batch, a2, a1, params.w2, z1, batch_size)
            params = gradient_descent(params, grads, learning_rate)

            start_idx = end_idx
            num_batches += 1

        loss_per_epoch /= num_batches
        accuracy_per_epoch /= num_batches

        model_data.train_loss.append(loss_per_epoch)
        model_data.train_accuracy.append(accuracy_per_epoch)

        logger.info(
            f"Epoch: {epoch + 1} | Loss: {loss_per_epoch:.4f} | Accuracy: {(accuracy_per_epoch * 100):.2f}%"
        )

    return params, model_data


def test_model(
    x_test: np.ndarray, y_test: np.ndarray, params: Params, model_data: ModelData
) -> None:
    _, _, a2 = forward_prop(x_test, params)
    loss = cross_entropy(y_test, a2)
    accuracy = calc_accuracy(y_test, a2)

    model_data.test_loss = loss
    model_data.test_accuracy = accuracy

    logger.info(f"Test loss: {loss:.4f} | Test accuracy: {(accuracy * 100):.2f}%")


def run_model(
    hidden_neurons_count: int, learning_rate: float, epochs: int, batch_size: int
) -> ModelData:
    logger.info(
        f"Running model with {hidden_neurons_count=}, {learning_rate=}, {epochs=}, {batch_size=}"
    )
    x_train, y_train, x_test, y_test = get_data()
    params = init_params(hidden_neurons_count)
    params, model_data = train_model(
        x_train, y_train, params, epochs, batch_size, learning_rate
    )
    test_model(x_test, y_test, params, model_data)

    return model_data


def main() -> None:
    model1_data = run_model(hidden_neurons_count=10, learning_rate=0.01, epochs=100, batch_size=250)
    print(model1_data)


if __name__ == "__main__":
    main()
