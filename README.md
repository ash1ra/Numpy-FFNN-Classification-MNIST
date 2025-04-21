# Neural Network for Digit Classification (MNIST) using NumPy
This project implements a feedforward neural network from scratch using NumPy to classify handwritten digits from the MNIST dataset. The model is designed to explore the impact of various hyperparameters on performance, including the number of epochs, optimizer, activation function, learning rate, hidden layer size, and batch size. 

## Project Structure
- `main.py`: main script for loading data, training models, and generating visualizations;
- `model.py`: implementation of the neural network, including forward/backward propagation, optimizers, and activation functions;
- `plots.py`: visualization utilities for plotting training/validation metrics and predictions;
- `data/train.csv`: MNIST dataset in CSV format;
- `logs/`: directory for storing training logs;
- `images/`: directory for storing results of the testing model parameters.

## Model Architecture
The neural network is a two-layer feedforward network:
- **Input Layer**: 784 neurons (28x28 pixel images flattened);
- **Hidden Layer**: configurable number of neurons;
- **Output Layer**: 10 neurons (one for each digit, 0-9);
- **Loss Function**: Cross-entropy loss;
- **Output Activation**: Softmax (for training) and ArgMax (for predicting);
- **Hidden Layer Activations**: Sigmoid, tanh, ReLU, Leaky ReLU, ELU;
- **Optimizers**: Gradient Descent, Momentum, Nesterov (simplified version), RMSprop, Adam.

The model supports batch training, model performance averaging over multiple runs, early stopping based on validation loss to prevent overfitting and logs training progress to both console and files.

## Data Preprocessing
The MNIST dataset is loaded from `data/train.csv`, contain 10 samples and split into:
- **Validation Set**: configurable (by default 6000 samples);
- **Test Set**: configurable (by default 2000 samples);
- **Training Set**: remaining samples (by default, 34000).

Pixel values are normalized to [0, 1] by dividing by 255. Labels are one-hot encoded for training.
