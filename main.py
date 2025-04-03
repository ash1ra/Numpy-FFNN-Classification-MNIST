import numpy as np
import pandas as pd

import plots
from model import Model

TEST_DATA_SPLIT = 2000
VAL_DATA_SPLIT = 6000


def one_hot_encoding(y: np.ndarray) -> np.ndarray:
    one_hot_y = np.zeros((y.size, np.max(y) + 1))
    one_hot_y[np.arange(y.size), y] = 1
    return one_hot_y


def get_data() -> tuple[np.ndarray, ...]:
    data = pd.read_csv("data/train.csv").to_numpy()

    test_data = data[:TEST_DATA_SPLIT]
    val_data = data[TEST_DATA_SPLIT : TEST_DATA_SPLIT + VAL_DATA_SPLIT]
    train_data = data[TEST_DATA_SPLIT + VAL_DATA_SPLIT :]

    x_train, y_train = train_data[:, 1:] / 255, train_data[:, 0]
    x_val, y_val = val_data[:, 1:] / 255, val_data[:, 0]
    x_test, y_test = test_data[:, 1:] / 255, test_data[:, 0]

    y_train = one_hot_encoding(y_train)
    y_val = one_hot_encoding(y_val)
    y_test = one_hot_encoding(y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test


def main() -> None:
    x_train, y_train, x_val, y_val, x_test, y_test = get_data()

    model1 = Model(
        hidden_neurons_count=10,
        hidden_activation_func="relu",
        optimizer="gradient_descent",
        learning_rate=0.001,
        epochs=250,
        batch_size=250,
    )

    model1.calc_avarage(x_train, y_train, x_val, y_val, x_test, y_test, 2)
    plots.plot_predictions(model1, x_test, y_test, np.random.randint(0, x_test.shape[-1], 20))

    model2 = Model(
        hidden_neurons_count=10,
        hidden_activation_func="relu",
        optimizer="momentum",
        learning_rate=0.001,
        epochs=250,
        batch_size=250,
    )

    model2.calc_avarage(x_train, y_train, x_val, y_val, x_test, y_test, 2)

    model3 = Model(
        hidden_neurons_count=10,
        hidden_activation_func="relu",
        optimizer="nesterov",
        learning_rate=0.001,
        epochs=250,
        batch_size=250,
    )

    model3.calc_avarage(x_train, y_train, x_val, y_val, x_test, y_test, 2)

    model4 = Model(
        hidden_neurons_count=10,
        hidden_activation_func="relu",
        optimizer="rmsprop",
        learning_rate=0.001,
        epochs=250,
        batch_size=250,
    )

    model4.calc_avarage(x_train, y_train, x_val, y_val, x_test, y_test, 2)

    model5 = Model(
        hidden_neurons_count=10,
        hidden_activation_func="relu",
        optimizer="adam",
        learning_rate=0.001,
        epochs=250,
        batch_size=250,
    )

    model5.calc_avarage(x_train, y_train, x_val, y_val, x_test, y_test, 2)

    plots.plot_train_and_val_loss_and_accuracy(
        (
            model1.model_data,
            model2.model_data,
            model3.model_data,
            model4.model_data,
            model5.model_data,
        ),
        "optimizer",
    )

    plots.plot_test_loss_and_accuracy(
        (
            model1.model_data,
            model2.model_data,
            model3.model_data,
            model4.model_data,
            model5.model_data,
        ),
        "optimizer",
    )


if __name__ == "__main__":
    main()
