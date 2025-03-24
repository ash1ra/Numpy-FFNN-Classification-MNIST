import matplotlib.pyplot as plt
from model import ModelData


def plot_train_loss_and_accuracy(
    models_data: tuple[ModelData, ...], comparable_attr: str
) -> None:
    fig, axes = plt.subplots(1, 2)

    attrs_to_skip = [
        comparable_attr,
        "train_time",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
    ]
    suptitle = "| "
    for attr in vars(models_data[0]):
        if attr not in attrs_to_skip:
            formated_attr = attr.capitalize().replace("_", " ")
            attr_value = eval(f"models_data[0].{attr}")
            suptitle += f"{formated_attr}: {attr_value} | "

    fig.suptitle(suptitle)

    axes[0].set_title("Train loss")
    axes[0].set(xlabel="Epochs", ylabel="Loss")
    axes[0].set_yticks([])

    axes[1].set_title("Train accuracy")
    axes[1].set(xlabel="Epochs", ylabel="Accuracy")
    axes[1].set_yticks([])

    lines, labels = [], []
    for model_data in models_data:
        comparable_attr_data = eval(f"model_data.{comparable_attr}")
        label = (
            f"{comparable_attr.capitalize().replace("_", " ")}: {comparable_attr_data}"
        )

        (line,) = axes[0].plot(model_data.train_loss, label=label)

        lines.append(line)
        labels.append(label)

        axes[1].plot(model_data.train_accuracy)

    fig.legend(lines, labels, loc="outside lower center")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    plt.show()


def plot_test_loss_and_accuracy(
    models_data: tuple[ModelData, ...], comparable_attr: str
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    attrs_to_skip = [
        comparable_attr,
        "train_time",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
    ]
    suptitle = "| "
    for attr in vars(models_data[0]):
        if attr not in attrs_to_skip:
            formated_attr = attr.capitalize().replace("_", " ")
            attr_value = eval(f"models_data[0].{attr}")
            suptitle += f"{formated_attr}: {attr_value} | "
    fig.suptitle(suptitle)

    x_labels = [
        str(eval(f"model_data.{comparable_attr}")) for model_data in models_data
    ]
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
