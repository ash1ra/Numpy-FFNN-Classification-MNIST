import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from model import Model, ModelData

pio.templates.default = "plotly"


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

    return " | ".join(attrs)


def create_comparable_label(model_data: ModelData, comparable_attr: str) -> str:
    if comparable_attr in ["hidden_activation_func", "optimizer"]:
        label_value = getattr(model_data, comparable_attr)[0]
    else:
        label_value = getattr(model_data, comparable_attr)

    return label_value


def plot_train_and_val_loss_and_accuracy(models_data: tuple[ModelData, ...], comparable_attr: str) -> None:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Train loss (less is better)",
            "Train accuracy (greater is better)",
            "Validate loss (less is better)",
            "Validate accuracy (greater is better)",
        ),
    )

    unique_values = {create_comparable_label(model_data, comparable_attr) for model_data in models_data}
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    color_map = {value: colors[i % len(colors)] for i, value in enumerate(unique_values)}

    for model_data in models_data:
        label_value = create_comparable_label(model_data, comparable_attr)
        epochs = list(range(len(model_data.train_loss)))
        color = color_map[label_value]

        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=model_data.train_loss,
                name=label_value,
                legendgroup=label_value,
                showlegend=True,
                line=dict(color=color),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=model_data.train_accuracy,
                name=label_value,
                legendgroup=label_value,
                showlegend=False,
                line=dict(color=color),
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=model_data.val_loss,
                name=label_value,
                legendgroup=label_value,
                showlegend=False,
                line=dict(color=color),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=model_data.val_accuracy,
                name=label_value,
                legendgroup=label_value,
                showlegend=False,
                line=dict(color=color),
            ),
            row=2,
            col=2,
        )

    fig.update_xaxes(title_text="Epochs")
    fig.update_yaxes(title_text="Loss", col=1, showticklabels=False)
    fig.update_yaxes(title_text="Accuracy", col=2, showticklabels=False)

    fig.update_layout(
        title_text=create_suptitle(models_data[0], comparable_attr),
        title=dict(
            x=0.5,
            y=0.98,
            xanchor="center",
            yanchor="top",
            font=dict(size=20),
        ),
        legend_title_text=f"{comparable_attr.capitalize().replace('_', ' ')}:",
        legend=dict(
            x=0.5,
            y=-0.05,
            xanchor="center",
            yanchor="bottom",
            orientation="h",
            font=dict(size=16),
        ),
    )

    fig.show()


def plot_test_loss_and_accuracy(models_data: tuple[ModelData, ...], comparable_attr: str) -> None:
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Test loss (less is better)", "Test accuracy (greater is better)")
    )

    x_labels = [create_comparable_label(model_data, comparable_attr) for model_data in models_data]

    test_losses = [model_data.test_loss for model_data in models_data]
    test_accuracies = [model_data.test_accuracy for model_data in models_data]
    train_times = [model_data.train_time for model_data in models_data]
    epochs = [len(model_data.train_loss) for model_data in models_data]

    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=test_losses,
            text=[
                f"Loss: {loss:.2f}<br>Train time: {time:.2f}s<br>Train epochs: {epochs}"
                for loss, time, epochs in zip(test_losses, train_times, epochs)
            ],
            textfont=dict(size=16),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=test_accuracies,
            text=[
                f"Accuracy: {acc * 100:.2f}%<br>Train time: {time:.2f}s<br>Train epochs: {epochs}"
                for acc, time, epochs in zip(test_accuracies, train_times, epochs)
            ],
            textfont=dict(size=16),
        ),
        row=1,
        col=2,
    )

    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(tickfont=dict(size=16))

    fig.update_layout(
        title_text=create_suptitle(models_data[0], comparable_attr),
        title=dict(
            x=0.5,
            y=0.98,
            xanchor="center",
            yanchor="top",
            font=dict(size=20),
        ),
        showlegend=False,
    )

    fig.show()


def plot_predictions(model: Model, x: np.ndarray, y: np.ndarray, indices: np.ndarray) -> None:
    heatmap_size = 475
    cols = 5
    rows = (len(indices) + cols - 1) // cols

    subplot_titles = []
    for index in indices:
        image = x[index]
        true_label = np.argmax(y[index])
        pred_label = model.predict(image)
        color = "#d62728" if true_label != pred_label else "#2ca02c"
        subplot_titles.append(f'<span style="color:{color}">True: {true_label} | Pred: {pred_label}</span>')

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.1,
        row_heights=[heatmap_size] * rows,
        column_widths=[heatmap_size] * cols,
    )

    for i, index in enumerate(indices):
        row = (i // cols) + 1
        col = (i % cols) + 1

        image = np.flipud(x[index].reshape(28, 28))

        fig.add_trace(go.Heatmap(z=image, colorscale="Gray", showscale=False, zmin=0, zmax=1), row=row, col=col)

    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, showticklabels=False)

    fig.update_layout(
        title_text=create_suptitle(model.model_data),
        title=dict(
            x=0.5,
            y=0.98,
            xanchor="center",
            yanchor="top",
            font=dict(size=20),
        ),
        width=heatmap_size * cols + (cols - 1) * heatmap_size * 0.05,
        height=heatmap_size * rows + (rows - 1) * heatmap_size * 0.1 + 100,
        showlegend=False,
    )

    fig.show()
