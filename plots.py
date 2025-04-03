import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from model import Model, ModelData

sns.set_theme()


def create_suptitle(model_data: ModelData, comparable_attr: str | None = None) -> str:
    attrs_to_skip = [
        comparable_attr,
        "train_time",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "test_loss",
        "test_accuracy",
    ]

    attrs = []
    for attr in vars(model_data):
        if attr not in attrs_to_skip:
            formated_attr = attr.capitalize().replace("_", " ")
            if attr in ["hidden_activation_func", "optimizer"]:
                attr_value = getattr(model_data, attr)[0]
            else:
                attr_value = getattr(model_data, attr)
            attrs.append(f"{formated_attr}: {attr_value}")

    mid_point = (len(attrs) + 1) // 2
    first_line = " | ".join(attrs[:mid_point])
    second_line = " | ".join(attrs[mid_point:])

    return f"{first_line}\n{second_line}"


def plot_train_and_val_loss_and_accuracy(models_data: tuple[ModelData, ...], comparable_attr: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    fig.suptitle(create_suptitle(models_data[0], comparable_attr))

    axes[0, 0].set_title("Train loss")
    axes[0, 0].set(xlabel="Epochs", ylabel="Loss")
    axes[0, 0].set_yticks([])

    axes[0, 1].set_title("Train accuracy")
    axes[0, 1].set(xlabel="Epochs", ylabel="Accuracy")
    axes[0, 1].set_yticks([])

    axes[1, 0].set_title("Validation loss")
    axes[1, 0].set(xlabel="Epochs", ylabel="Loss")
    axes[1, 0].set_yticks([])

    axes[1, 1].set_title("Validation accuracy")
    axes[1, 1].set(xlabel="Epochs", ylabel="Accuracy")
    axes[1, 1].set_yticks([])

    lines, labels = [], []
    for model_data in models_data:
        if comparable_attr in ["hidden_activation_func", "optimizer"]:
            comparable_attr_data = getattr(model_data, comparable_attr)[0]
        else:
            comparable_attr_data = getattr(model_data, comparable_attr)

        label = f"{comparable_attr.capitalize().replace('_', ' ')}: {comparable_attr_data}"

        (line,) = axes[0, 0].plot(model_data.train_loss, label=label)

        lines.append(line)
        labels.append(label)

        axes[0, 1].plot(model_data.train_accuracy)

        axes[1, 0].plot(model_data.val_loss)
        axes[1, 1].plot(model_data.val_accuracy)

    fig.legend(lines, labels, loc="outside lower center")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    plt.show()


def plot_test_loss_and_accuracy(models_data: tuple[ModelData, ...], comparable_attr: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    fig.suptitle(create_suptitle(models_data[0], comparable_attr))

    if comparable_attr in ["hidden_activation_func", "optimizer"]:
        x_labels = [getattr(model_data, comparable_attr)[0] for model_data in models_data]
    else:
        x_labels = [getattr(model_data, comparable_attr) for model_data in models_data]

    test_losses = [model_data.test_loss for model_data in models_data]
    test_accuracies = [model_data.test_accuracy for model_data in models_data]
    train_times = [model_data.train_time for model_data in models_data]
    x_positions = range(len(models_data))

    axes[0].set_title("Test loss (lower is better)")
    test_loss_bars = axes[0].bar(x_positions, test_losses)
    axes[0].set_xticks(x_positions)
    axes[0].set_yticks([])
    axes[0].set_xticklabels(x_labels)
    axes[0].set_xlabel(comparable_attr.capitalize().replace("_", " "))

    for bar, train_time in zip(test_loss_bars, train_times):
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            0,
            f"{train_time:.2f}s",
            ha="center",
            va="bottom",
            color="white",
        )

    axes[1].set_title("Test accuracy (higher is better)")
    test_accuracy_bars = axes[1].bar(x_positions, test_accuracies)
    axes[1].set_xticks(x_positions)
    axes[1].set_yticks([])
    axes[1].set_xticklabels(x_labels)
    axes[1].set_xlabel(comparable_attr.capitalize().replace("_", " "))

    for bar, train_time in zip(test_accuracy_bars, train_times):
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{(height * 100):.2f}%",
            ha="center",
            va="bottom",
        )
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            0,
            f"{train_time:.2f}s",
            ha="center",
            va="bottom",
            color="white",
        )
    plt.tight_layout()

    plt.show()


def plot_predictions(model: Model, x: np.ndarray, y: np.ndarray, indices: np.ndarray) -> None:
    cols = 5
    rows = (len(indices) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle(create_suptitle(model.model_data))

    if rows == 1:
        axes = np.array([axes])

    for i, index in enumerate(indices):
        row = i // cols
        col = i % cols
        ax = axes[row, col]

        image = x[index].reshape(28, 28)
        true_label = np.argmax(y[index])
        pred_label = model.predict(x[index])

        ax.imshow(image, cmap="gray")
        color = "red" if true_label != pred_label else "green"
        ax.set_title(f"True: {true_label} | Pred: {pred_label}", color=color)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
