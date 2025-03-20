import matplotlib.pyplot as plt
from model import ModelData


def plot_train_loss_and_accuracy(
    models_data: tuple[ModelData, ...], comparable_attr: str
) -> None:
    fig, axes = plt.subplots(1, 2)

    attrs_to_skip = [
        comparable_attr,
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
