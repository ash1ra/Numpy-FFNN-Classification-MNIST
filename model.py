import numpy as np
from dataclasses import dataclass, field
import logging

EPS = 1e-15
LOGGER_LEVEL = logging.INFO

logger = logging.getLogger(__name__)
logger.setLevel(LOGGER_LEVEL)
console_handler = logging.StreamHandler()
console_handler.setLevel(LOGGER_LEVEL)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


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
class Activations:
    z1: np.ndarray
    a1: np.ndarray
    a2: np.ndarray


@dataclass
class ModelData:
    hidden_neurons_count: int
    learning_rate: float
    epochs: int
    batch_size: int
    train_loss: list = field(default_factory=list)
    train_accuracy: list = field(default_factory=list)
    test_loss: np.float64 = field(default_factory=np.float64)
    test_accuracy: np.float64 = field(default_factory=np.float64)


class Model:
    def __init__(
        self,
        hidden_neurons_count: int,
        learning_rate: float,
        epochs: int,
        batch_size: int,
    ) -> None:
        self.model_data = ModelData(
            hidden_neurons_count=hidden_neurons_count,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
        )
        self.params = self._init_params()
        self.activations: Activations
        self.grads: Grads

    def _init_params(self) -> Params:
        w1 = np.random.randn(784, self.model_data.hidden_neurons_count) * np.sqrt(
            2 / 784
        )
        b1 = np.zeros((1, self.model_data.hidden_neurons_count))

        w2 = np.random.randn(self.model_data.hidden_neurons_count, 10) * np.sqrt(
            2 / self.model_data.hidden_neurons_count
        )
        b2 = np.zeros((1, 10))

        return Params(w1, b1, w2, b2)

    def _relu(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    def _relu_derivative(self, z: np.ndarray) -> np.ndarray:
        return z > 0

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        exps = np.exp(z - z.max(axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def _cross_entropy(self, y: np.ndarray) -> np.float64:
        return (-np.sum(y * np.log(self.activations.a2 + EPS))) / y.shape[0]

    def _forward_prop(self, x: np.ndarray) -> None:
        z1 = np.matmul(x, self.params.w1) + self.params.b1
        a1 = self._relu(z1)

        z2 = np.matmul(a1, self.params.w2) + self.params.b2
        a2 = self._softmax(z2)

        self.activations = Activations(z1, a1, a2)

    def _backward_prop(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        dz2 = self.activations.a2 - y
        dw2 = np.matmul(dz2.T, self.activations.a1) / self.model_data.batch_size
        db2 = np.sum(dz2, axis=0, keepdims=True) / self.model_data.batch_size

        da1 = np.matmul(self.params.w2, dz2.T)
        dz1 = da1.T * self._relu_derivative(self.activations.z1)
        dw1 = np.matmul(x.T, dz1) / self.model_data.batch_size
        db1 = np.sum(dz1, axis=0, keepdims=True) / self.model_data.batch_size

        self.grads = Grads(dw1, db1, dw2, db2)

    def _gradient_descent(self) -> None:
        self.params.w1 -= self.model_data.learning_rate * self.grads.dw1
        self.params.b1 -= self.model_data.learning_rate * self.grads.db1
        self.params.w2 -= self.model_data.learning_rate * self.grads.dw2
        self.params.b2 -= self.model_data.learning_rate * self.grads.db2

    def _calc_accuracy(self, y_true: np.ndarray) -> np.float64:
        true_labels = np.argmax(y_true, axis=1)
        predictions = np.argmax(self.activations.a2, axis=1)
        accuracy = np.mean(true_labels == predictions)
        return accuracy

    def train_model(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        train_samples_count = x_train.shape[0]

        for epoch in range(self.model_data.epochs):
            perm = np.random.permutation(train_samples_count)
            x_train, y_train = x_train[perm], y_train[perm]

            start_idx = 0
            loss_per_epoch, accuracy_per_epoch = 0, 0
            num_batches = 0

            while start_idx < train_samples_count:
                end_idx = min(
                    start_idx + self.model_data.batch_size, train_samples_count
                )
                x_batch, y_batch = (
                    x_train[start_idx:end_idx],
                    y_train[start_idx:end_idx],
                )

                self._forward_prop(x_batch)
                self._backward_prop(x_batch, y_batch)
                self._gradient_descent()

                loss_per_epoch += self._cross_entropy(y_batch)
                accuracy_per_epoch += self._calc_accuracy(y_batch)

                start_idx = end_idx
                num_batches += 1

            loss_per_epoch /= num_batches
            accuracy_per_epoch /= num_batches

            self.model_data.train_loss.append(loss_per_epoch)
            self.model_data.train_accuracy.append(accuracy_per_epoch)

            logger.info(
                f"Epoch: {epoch + 1} | Loss: {loss_per_epoch:.4f} | Accuracy: {(accuracy_per_epoch * 100):.2f}%"
            )

    def test_model(self, x_test: np.ndarray, y_test: np.ndarray) -> None:
        self._forward_prop(x_test)
        loss = self._cross_entropy(y_test)
        accuracy = self._calc_accuracy(y_test)

        self.model_data.test_loss = loss
        self.model_data.test_accuracy = accuracy

        logger.info(f"Test loss: {loss:.4f} | Test accuracy: {(accuracy * 100):.2f}%\n")

    def calc_avarage(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        count: int,
    ) -> None:
        all_train_losses = []
        all_train_accuracies = []
        all_test_losses = []
        all_test_accuracies = []

        for i in range(count):
            logger.info(f"Run {i + 1}/{count}")

            self.params = self._init_params()
            self.model_data.train_loss = []
            self.model_data.train_accuracy = []

            self.train_model(x_train, y_train)
            self.test_model(x_test, y_test)

            all_train_losses.append(self.model_data.train_loss)
            all_train_accuracies.append(self.model_data.train_accuracy)
            all_test_losses.append(self.model_data.test_loss)
            all_test_accuracies.append(self.model_data.test_accuracy)

        avg_train_loss = np.mean(np.array(all_train_losses), axis=0).tolist()
        avg_train_accuracy = np.mean(np.array(all_train_accuracies), axis=0).tolist()

        avg_test_loss = np.mean(all_test_losses)
        avg_test_accuracy = np.mean(all_test_accuracies)

        self.model_data.train_loss = avg_train_loss
        self.model_data.train_accuracy = avg_train_accuracy
        self.model_data.test_loss = avg_test_loss
        self.model_data.test_accuracy = avg_test_accuracy

        logger.info(
            f"Average results over {count} runs:\n"
            f"Train loss: {avg_train_loss[-1]:.4f} (last epoch)\n"
            f"Train accuracy: {(avg_train_accuracy[-1] * 100):.2f}% (last epoch)\n"
            f"Test loss: {avg_test_loss:.4f}\n"
            f"Test accuracy: {(avg_test_accuracy * 100):.2f}%\n"
        )
