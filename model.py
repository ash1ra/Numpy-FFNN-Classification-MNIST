import datetime
import logging
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveFloat, PositiveInt

INPUT_SIZE = 28 * 28
OUTPUT_SIZE = 10
EPS = 1e-15
LOGGER_LEVEL = logging.INFO
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(LOGGER_LEVEL)

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

log_filename = LOGS_DIR / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(LOGGER_LEVEL)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(LOGGER_LEVEL)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


class Params(BaseModel):
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Grads(BaseModel):
    dw1: np.ndarray
    db1: np.ndarray
    dw2: np.ndarray
    db2: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Activations(BaseModel):
    z1: np.ndarray
    a1: np.ndarray
    a2: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Velocities(BaseModel):
    v_w1: np.ndarray
    v_b1: np.ndarray
    v_w2: np.ndarray
    v_b2: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Cache(BaseModel):
    s_w1: np.ndarray
    s_b1: np.ndarray
    s_w2: np.ndarray
    s_b2: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ModelData(BaseModel):
    hidden_neurons_count: PositiveInt
    hidden_activation_func: tuple[str, Callable, Callable]
    optimizer: tuple[str, Callable]
    learning_rate: PositiveFloat
    epochs: PositiveInt
    batch_size: PositiveInt
    patience: PositiveInt
    min_delta: NonNegativeFloat
    train_time: float | np.float64 = Field(default_factory=float)
    train_epochs: PositiveInt = Field(default_factory=int)
    train_loss: list = Field(default_factory=list)
    train_accuracy: list = Field(default_factory=list)
    val_loss: list = Field(default_factory=list)
    val_accuracy: list = Field(default_factory=list)
    test_loss: np.float64 = Field(default_factory=np.float64)
    test_accuracy: np.float64 = Field(default_factory=np.float64)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Model:
    def __init__(
        self,
        hidden_neurons_count: PositiveInt,
        hidden_activation_func: str,
        optimizer: str,
        learning_rate: PositiveFloat,
        epochs: PositiveInt,
        batch_size: PositiveInt,
        patience: PositiveInt = 10,
        min_delta: PositiveFloat = 0.0,
    ) -> None:
        self._activation_map = {
            "sigmoid": ("Sigmoid", self._sigmoid, self._sigmoid_derivative),
            "tanh": ("tanh", self._tanh, self._tanh_derivative),
            "relu": ("ReLU", self._relu, self._relu_derivative),
            "leaky_relu": ("Leaky ReLU", self._leaky_relu, self._leaky_relu_derivative),
            "elu": ("ELU", self._elu, self._elu_derivative),
        }
        self._optimizer_map = {
            "gradient_descent": ("Gradient Descent", self._gradient_descent),
            "momentum": ("Momentum", self._momentum),
            "nesterov": ("Nesterov", self._nesterov),
            "rmsprop": ("RMSprop", self._rmsprop),
            "adam": ("Adam", self._adam),
        }
        self._t = 0
        self.model_data = ModelData(
            hidden_neurons_count=hidden_neurons_count,
            hidden_activation_func=self._activation_map[hidden_activation_func],
            optimizer=self._optimizer_map[optimizer],
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            min_delta=min_delta,
        )
        self.params = self._init_params()
        self.activations: Activations
        self.grads: Grads

        if optimizer in ["momentum", "nesterov", "adam"]:
            self.velocities = self._init_velocities()

        if optimizer in ["rmsprop", "adam"]:
            self.cache = self._init_cache()

        logger.info("Model initialization with parameters:")
        logger.info(f"Number of hidden neurons: {self.model_data.hidden_neurons_count}")
        logger.info(f"Hidden layer activation function: {self.model_data.hidden_activation_func[0]}")
        logger.info(f"Optimizer: {self.model_data.optimizer[0]}")
        logger.info(f"Learning rate: {self.model_data.learning_rate}")
        logger.info(f"Epochs: {self.model_data.epochs}")
        logger.info(f"Batch size: {self.model_data.batch_size}")
        logger.info(f"Patience: {self.model_data.patience}")
        logger.info(f"Minimum delta: {self.model_data.min_delta}\n")

    def _init_params(self) -> Params:
        activation = self.model_data.hidden_activation_func[0]

        if activation in ["ReLU", "Leaky ReLU", "ELU"]:
            scale_w1 = np.sqrt(2 / INPUT_SIZE)
            scale_w2 = np.sqrt(2 / self.model_data.hidden_neurons_count)
        else:
            scale_w1 = np.sqrt(2 / (INPUT_SIZE + self.model_data.hidden_neurons_count))
            scale_w2 = np.sqrt(2 / (self.model_data.hidden_neurons_count + OUTPUT_SIZE))

        w1 = np.random.randn(INPUT_SIZE, self.model_data.hidden_neurons_count) * scale_w1
        b1 = np.zeros((1, self.model_data.hidden_neurons_count))

        w2 = np.random.randn(self.model_data.hidden_neurons_count, OUTPUT_SIZE) * scale_w2
        b2 = np.zeros((1, OUTPUT_SIZE))

        return Params(w1=w1, b1=b1, w2=w2, b2=b2)

    def _init_velocities(self) -> Velocities:
        return Velocities(
            v_w1=np.zeros_like(self.params.w1),
            v_b1=np.zeros_like(self.params.b1),
            v_w2=np.zeros_like(self.params.w2),
            v_b2=np.zeros_like(self.params.b2),
        )

    def _init_cache(self) -> Cache:
        return Cache(
            s_w1=np.zeros_like(self.params.w1),
            s_b1=np.zeros_like(self.params.b1),
            s_w2=np.zeros_like(self.params.w2),
            s_b2=np.zeros_like(self.params.b2),
        )

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, z: np.ndarray) -> np.ndarray:
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def _tanh(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def _tanh_derivative(self, z: np.ndarray) -> np.ndarray:
        return 1 - np.power(self._tanh(z), 2)

    def _relu(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    def _relu_derivative(self, z: np.ndarray) -> np.ndarray:
        return z > 0

    def _leaky_relu(self, z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(z > 0, z, z * alpha)

    def _leaky_relu_derivative(self, z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(z > 0, 1, alpha)

    def _elu(self, z: np.ndarray, alpha: float = 1) -> np.ndarray:
        return np.where(z > 0, z, alpha * (np.exp(z) - 1))

    def _elu_derivative(self, z: np.ndarray, alpha: float = 1) -> np.ndarray:
        return np.where(z > 0, 1, alpha * np.exp(z))

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        exps = np.exp(z - z.max(axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def _cross_entropy(self, y: np.ndarray, batch_size: int | None = None) -> np.float64:
        if not batch_size:
            batch_size = self.model_data.batch_size

        return (-np.sum(y * np.log(self.activations.a2 + EPS))) / batch_size

    def _forward_prop(self, x: np.ndarray) -> None:
        z1 = np.matmul(x, self.params.w1) + self.params.b1
        a1 = self.model_data.hidden_activation_func[1](z1)

        z2 = np.matmul(a1, self.params.w2) + self.params.b2
        a2 = self._softmax(z2)

        self.activations = Activations(z1=z1, a1=a1, a2=a2)

    def _backward_prop(self, x: np.ndarray, y: np.ndarray) -> None:
        dz2 = self.activations.a2 - y
        dw2 = np.matmul(self.activations.a1.T, dz2) / self.model_data.batch_size
        # dw2 = np.matmul(dz2.T, self.activations.a1) / self.model_data.batch_size
        db2 = np.sum(dz2, axis=0, keepdims=True) / self.model_data.batch_size

        da1 = np.matmul(self.params.w2, dz2.T)
        dz1 = da1.T * self.model_data.hidden_activation_func[2](self.activations.z1)
        dw1 = np.matmul(x.T, dz1) / self.model_data.batch_size
        db1 = np.sum(dz1, axis=0, keepdims=True) / self.model_data.batch_size

        self.grads = Grads(dw1=dw1, db1=db1, dw2=dw2, db2=db2)

    def _gradient_descent(self) -> None:
        self.params.w1 -= self.model_data.learning_rate * self.grads.dw1
        self.params.b1 -= self.model_data.learning_rate * self.grads.db1
        self.params.w2 -= self.model_data.learning_rate * self.grads.dw2
        self.params.b2 -= self.model_data.learning_rate * self.grads.db2

    def _momentum(self, momentum: float = 0.9) -> None:
        self.velocities.v_w1 = momentum * self.velocities.v_w1 - self.model_data.learning_rate * self.grads.dw1
        self.velocities.v_b1 = momentum * self.velocities.v_b1 - self.model_data.learning_rate * self.grads.db1
        self.velocities.v_w2 = momentum * self.velocities.v_w2 - self.model_data.learning_rate * self.grads.dw2
        self.velocities.v_b2 = momentum * self.velocities.v_b2 - self.model_data.learning_rate * self.grads.db2

        self.params.w1 += self.velocities.v_w1
        self.params.b1 += self.velocities.v_b1
        self.params.w2 += self.velocities.v_w2
        self.params.b2 += self.velocities.v_b2

    def _nesterov(self, momentum: float = 0.9) -> None:
        w1_prev = self.params.w1.copy()
        b1_prev = self.params.b1.copy()
        w2_prev = self.params.w2.copy()
        b2_prev = self.params.b2.copy()

        self.params.w1 += momentum * self.velocities.v_w1
        self.params.b1 += momentum * self.velocities.v_b1
        self.params.w2 += momentum * self.velocities.v_w2
        self.params.b2 += momentum * self.velocities.v_b2

        self.velocities.v_w1 = momentum * self.velocities.v_w1 - self.model_data.learning_rate * self.grads.dw1
        self.velocities.v_b1 = momentum * self.velocities.v_b1 - self.model_data.learning_rate * self.grads.db1
        self.velocities.v_w2 = momentum * self.velocities.v_w2 - self.model_data.learning_rate * self.grads.dw2
        self.velocities.v_b2 = momentum * self.velocities.v_b2 - self.model_data.learning_rate * self.grads.db2

        self.params.w1 = w1_prev + self.velocities.v_w1
        self.params.b1 = b1_prev + self.velocities.v_b1
        self.params.w2 = w2_prev + self.velocities.v_w2
        self.params.b2 = b2_prev + self.velocities.v_b2

    def _rmsprop(self, rho: float = 0.9) -> None:
        self.cache.s_w1 = rho * self.cache.s_w1 + (1 - rho) * (np.power(self.grads.dw1, 2))
        self.cache.s_b1 = rho * self.cache.s_b1 + (1 - rho) * (np.power(self.grads.db1, 2))
        self.cache.s_w2 = rho * self.cache.s_w2 + (1 - rho) * (np.power(self.grads.dw2, 2))
        self.cache.s_b2 = rho * self.cache.s_b2 + (1 - rho) * (np.power(self.grads.db2, 2))

        self.params.w1 -= self.model_data.learning_rate * self.grads.dw1 / (np.sqrt(self.cache.s_w1) + EPS)
        self.params.b1 -= self.model_data.learning_rate * self.grads.db1 / (np.sqrt(self.cache.s_b1) + EPS)
        self.params.w2 -= self.model_data.learning_rate * self.grads.dw2 / (np.sqrt(self.cache.s_w2) + EPS)
        self.params.b2 -= self.model_data.learning_rate * self.grads.db2 / (np.sqrt(self.cache.s_b2) + EPS)

    def _adam(self, beta1: float = 0.9, beta2: float = 0.999):
        self._t += 1

        self.velocities.v_w1 = beta1 * self.velocities.v_w1 + (1 - beta1) * self.grads.dw1
        self.velocities.v_b1 = beta1 * self.velocities.v_b1 + (1 - beta1) * self.grads.db1
        self.velocities.v_w2 = beta1 * self.velocities.v_w2 + (1 - beta1) * self.grads.dw2
        self.velocities.v_b2 = beta1 * self.velocities.v_b2 + (1 - beta1) * self.grads.db2

        self.cache.s_w1 = beta2 * self.cache.s_w1 + (1 - beta2) * (np.power(self.grads.dw1, 2))
        self.cache.s_b1 = beta2 * self.cache.s_b1 + (1 - beta2) * (np.power(self.grads.db1, 2))
        self.cache.s_w2 = beta2 * self.cache.s_w2 + (1 - beta2) * (np.power(self.grads.dw2, 2))
        self.cache.s_b2 = beta2 * self.cache.s_b2 + (1 - beta2) * (np.power(self.grads.db2, 2))

        v_w1_corr = self.velocities.v_w1 / (1 - np.power(beta1, self._t))
        v_b1_corr = self.velocities.v_b1 / (1 - np.power(beta1, self._t))
        v_w2_corr = self.velocities.v_w2 / (1 - np.power(beta1, self._t))
        v_b2_corr = self.velocities.v_b2 / (1 - np.power(beta1, self._t))

        s_w1_corr = self.cache.s_w1 / (1 - np.power(beta2, self._t))
        s_b1_corr = self.cache.s_b1 / (1 - np.power(beta2, self._t))
        s_w2_corr = self.cache.s_w2 / (1 - np.power(beta2, self._t))
        s_b2_corr = self.cache.s_b2 / (1 - np.power(beta2, self._t))

        self.params.w1 -= self.model_data.learning_rate * v_w1_corr / (np.sqrt(s_w1_corr) + EPS)
        self.params.b1 -= self.model_data.learning_rate * v_b1_corr / (np.sqrt(s_b1_corr) + EPS)
        self.params.w2 -= self.model_data.learning_rate * v_w2_corr / (np.sqrt(s_w2_corr) + EPS)
        self.params.b2 -= self.model_data.learning_rate * v_b2_corr / (np.sqrt(s_b2_corr) + EPS)

    def _calc_accuracy(self, y_true: np.ndarray) -> np.float64:
        true_labels = np.argmax(y_true, axis=1)
        predictions = np.argmax(self.activations.a2, axis=1)
        accuracy = np.mean(true_labels == predictions)
        return accuracy

    def train_model(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> None:
        timer_start = perf_counter()
        train_samples_count = x_train.shape[0]

        self._t = 0
        best_val_loss = float("inf")
        patience_counter = 0
        best_params = Params(
            w1=self.params.w1.copy(),
            b1=self.params.b1.copy(),
            w2=self.params.w2.copy(),
            b2=self.params.b2.copy(),
        )

        for epoch in range(self.model_data.epochs):
            perm = np.random.permutation(train_samples_count)
            x_train, y_train = x_train[perm], y_train[perm]

            start_idx = 0
            loss_per_epoch, accuracy_per_epoch = 0, 0
            num_batches = 0

            while start_idx < train_samples_count:
                if start_idx + self.model_data.batch_size <= train_samples_count:
                    end_idx = start_idx + self.model_data.batch_size
                else:
                    break

                x_batch, y_batch = x_train[start_idx:end_idx], y_train[start_idx:end_idx]

                self._forward_prop(x_batch)
                self._backward_prop(x_batch, y_batch)
                self.model_data.optimizer[1]()

                loss_per_epoch += self._cross_entropy(y_batch)
                accuracy_per_epoch += self._calc_accuracy(y_batch)

                start_idx = end_idx
                num_batches += 1

            loss_per_epoch /= num_batches
            accuracy_per_epoch /= num_batches

            self._forward_prop(x_val)
            val_loss = self._cross_entropy(y_val)
            val_accuracy = self._calc_accuracy(y_val)

            self.model_data.train_loss.append(loss_per_epoch)
            self.model_data.train_accuracy.append(accuracy_per_epoch)
            self.model_data.val_loss.append(val_loss)
            self.model_data.val_accuracy.append(val_accuracy)

            logger.info(
                f"Epoch: {epoch + 1} | Train loss: {loss_per_epoch:.4f} | Train acc: {(accuracy_per_epoch * 100):.2f}% | "
                f"Val loss: {val_loss:.4f} | Val acc: {(val_accuracy * 100):.2f}%"
            )

            if val_loss < best_val_loss - self.model_data.min_delta:
                best_val_loss = val_loss
                best_params = Params(
                    w1=self.params.w1.copy(),
                    b1=self.params.b1.copy(),
                    w2=self.params.w2.copy(),
                    b2=self.params.b2.copy(),
                )
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"No improvement in val loss for {patience_counter}/{self.model_data.patience} epochs")
                if patience_counter >= self.model_data.patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    self.params = best_params
                    break

        self.model_data.train_time = perf_counter() - timer_start

    def test_model(self, x_test: np.ndarray, y_test: np.ndarray) -> None:
        self._forward_prop(x_test)
        loss = self._cross_entropy(y_test)
        accuracy = self._calc_accuracy(y_test)

        self.model_data.test_loss = loss
        self.model_data.test_accuracy = accuracy

        logger.info(f"Test loss: {loss:.4f} | Test acc: {(accuracy * 100):.2f}%\n")

    def predict(self, x: np.ndarray) -> int:
        if x.ndim == 1:
            x = x.reshape(1, -1)

        self._forward_prop(x)
        predicted_label = np.argmax(self.activations.a2, axis=1)[0]

        return predicted_label

    def calc_avarage(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        count: int,
    ) -> None:
        all_train_times = []
        all_train_epochs = []
        all_train_losses = []
        all_train_accuracies = []
        all_val_losses = []
        all_val_accuracies = []
        all_test_losses = []
        all_test_accuracies = []

        for i in range(count):
            logger.info(f"Run {i + 1}/{count}")

            self.params = self._init_params()

            if self.model_data.optimizer in ["momentum", "nesterov", "adam"]:
                self.velocities = self._init_velocities()
            if self.model_data.optimizer in ["rmsprop", "adam"]:
                self.cache = self._init_cache()

            self.model_data.train_loss = []
            self.model_data.train_accuracy = []
            self.model_data.val_loss = []
            self.model_data.val_accuracy = []

            self.train_model(x_train, y_train, x_val, y_val)
            self.test_model(x_test, y_test)

            all_train_times.append(self.model_data.train_time)
            all_train_epochs.append(len(self.model_data.train_loss))
            all_train_losses.append(self.model_data.train_loss)
            all_train_accuracies.append(self.model_data.train_accuracy)
            all_val_losses.append(self.model_data.val_loss)
            all_val_accuracies.append(self.model_data.val_accuracy)
            all_test_losses.append(self.model_data.test_loss)
            all_test_accuracies.append(self.model_data.test_accuracy)

        avg_train_time = np.mean(all_train_times)
        avg_train_epochs = np.mean(all_train_epochs)

        max_length = max(len(lst) for lst in all_train_losses)
        avg_train_loss = []
        avg_train_accuracy = []
        for epoch in range(max_length):
            epoch_losses = [lst[epoch] for lst in all_train_losses if len(lst) > epoch]
            epoch_accuracies = [lst[epoch] for lst in all_train_accuracies if len(lst) > epoch]
            avg_train_loss.append(np.mean(epoch_losses))
            avg_train_accuracy.append(np.mean(epoch_accuracies))

        max_length = max(len(lst) for lst in all_val_losses)
        avg_val_loss = []
        avg_val_accuracy = []
        for epoch in range(max_length):
            epoch_losses = [lst[epoch] for lst in all_val_losses if len(lst) > epoch]
            epoch_accuracies = [lst[epoch] for lst in all_val_accuracies if len(lst) > epoch]
            avg_val_loss.append(np.mean(epoch_losses))
            avg_val_accuracy.append(np.mean(epoch_accuracies))

        avg_test_loss = np.mean(all_test_losses)
        avg_test_accuracy = np.mean(all_test_accuracies)

        self.model_data.train_time = avg_train_time
        self.model_data.train_epochs = avg_train_epochs
        self.model_data.train_loss = avg_train_loss
        self.model_data.train_accuracy = avg_train_accuracy
        self.model_data.val_loss = avg_val_loss
        self.model_data.val_accuracy = avg_val_accuracy
        self.model_data.test_loss = avg_test_loss
        self.model_data.test_accuracy = avg_test_accuracy

        logger.info(f"Average results over {count} runs:")
        logger.info(
            f"Train time: {avg_train_time:.2f}s | Train epochs: {avg_train_epochs} | Train time per epoch: {(avg_train_time / len(avg_train_loss)):.2f}s"
        )
        logger.info(f"Train loss: {avg_train_loss[-1]:.4f} (last epoch)")
        logger.info(f"Train accuracy: {(avg_train_accuracy[-1] * 100):.2f}% (last epoch)")
        logger.info(f"Validation loss: {avg_val_loss[-1]:.4f} (last epoch)")
        logger.info(f"Validation accuracy: {(avg_val_accuracy[-1] * 100):.2f}% (last epoch)")
        logger.info(f"Test loss: {avg_test_loss:.4f}")
        logger.info(f"Test accuracy: {(avg_test_accuracy * 100):.2f}%\n")

    def save_model(self, filepath: str | Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        arrays_to_save = {
            "w1": self.params.w1,
            "b1": self.params.b1,
            "w2": self.params.w2,
            "b2": self.params.b2,
        }

        np.savez(filepath.with_suffix(".npz"), **arrays_to_save)

    def load_model(self, filepath: str | Path) -> None:
        filepath = Path(filepath)

        arrays = np.load(filepath.with_suffix(".npz"))
        self.params = Params(
            w1=arrays["w1"],
            b1=arrays["b1"],
            w2=arrays["w2"],
            b2=arrays["b2"],
        )
