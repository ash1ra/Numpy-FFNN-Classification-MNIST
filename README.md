# Neural Network for Digit Classification (MNIST) using NumPy
This project implements a feedforward neural network from scratch using NumPy to classify handwritten digits from the MNIST dataset. The model is designed to explore the impact of various hyperparameters on performance, including the number of epochs, optimizer, activation function, learning rate, hidden layer size, and batch size. 

**Important**: this research is not 100% objective, as the test results may vary due to different computer load during the tests, variations in the implementation of the neural network, etc. The essence of the research for me is to understand the dependencies between the parameters of neural network training and to find the optimal variant for solving the task at hand.

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

The parameters `patience` and `min_delta` are required for tuning the early stopping algorithm and are not considered in this research.

## Data Preprocessing
The MNIST dataset is loaded from `data/train.csv`, contain 10 samples and split into:
- **Validation Set**: configurable (by default 6000 samples);
- **Test Set**: configurable (by default 2000 samples);
- **Training Set**: remaining samples (by default, 34000).

Pixel values are normalized to [0, 1] by dividing by 255. Labels are one-hot encoded for training.

## Investigating Parameter Effects on Model Performance
Optimizer tests are conducted with `learning_rate = 0.001` and `learning_rate = 0.0001` respectively. All other tests use Nesterov optimizer as a demonstration of a simpler solution and Adam as one of the best algorithms. All model parameters are shown at the top of the images. The first image shows the result of model training and validation, and the second image shows the result of testing.

### 1. Optimizers (`learning_rate = 0.001` with early stopping)
![Optimizers with learning_rate = 0.001 with early stopping](images/optimizers_lr001_train.png)
![Optimizers with learning_rate = 0.001 with early stopping](images/optimizers_lr001_test.png)

| Activation function | Train time | Epochs | Test loss | Test accuracy |
|---------------------|------------|--------|-----------|---------------|
| Gradient Descent    | 27.04s     | 250    | 2.88      | 90.00%        |
| Momentum            | 26.88s     | 250    | 2.01      | 92.80%        |
| Nesterov            | 28.34s     | 250    | 2.01      | 92.83%        |
| RMRprop             | 6.72s      | 55.9   | 1.89      | 93.38%        |
| Adam                | 7.14s      | 55.6   | 1.84      | 93.38%        |

**Conclusions:**  
...

### 2. Optimizers (`learning_rate = 0.0001` with early stopping)
![Optimizers with learning_rate = 0.0001 with early stopping](images/optimizers_lr0001_train.png)
![Optimizers with learning_rate = 0.0001 with early stopping](images/optimizers_lr0001_test.png)

| Activation function | Train time | Epochs | Test loss | Test accuracy |
|---------------------|------------|---------|-----------|---------------|
| Gradient Descent    | 25.83s     | 250     | 8.30      | 74.02%        |
| Momentum            | 25.76s     | 250     | 2.89      | 90.17%        |
| Nesterov            | 27.85s     | 250     | 2.91      | 90.08%        |
| RMRprop             | 30.13s     | 250     | 1.90      | 93.04%        |
| Adam                | 31.93s     | 247.1   | 1.88      | 93.37%        |

**Conclusions:**  
...

### 3. Hidden Layer Activation Functions (Nesterov with early stopping)
![Nesterov training with early stopping](images/act_func_nesterov_train_w_early_stopping.png)
![Nesterov testing with early stopping](images/act_func_nesterov_test_w_early_stopping.png)

| Activation function | Train time | Epochs | Test loss | Test accuracy |
|---------------------|------------|--------|-----------|---------------|
| Sigmoid             | 0          | 0      | 0         | 0             |
| tanh                | 0          | 0      | 0         | 0             |
| ReLU                | 0          | 0      | 0         | 0             |
| Leaky ReLU          | 0          | 0      | 0         | 0             |
| ELU                 | 0          | 0      | 0         | 0             |

**Conclusions:**  
...

### 4. Hidden Layer Activation Functions (Adam with early stopping)
![Adam training with early stopping](images/act_func_adam_train_w_early_stopping.png)
![Adam testing with early stopping](images/act_func_adam_test_w_early_stopping.png)

| Activation function | Train time | Epochs | Test loss | Test accuracy |
|---------------------|------------|--------|-----------|---------------|
| Sigmoid             | 0          | 0      | 0         | 0             |
| tanh                | 0          | 0      | 0         | 0             |
| ReLU                | 0          | 0      | 0         | 0             |
| Leaky ReLU          | 0          | 0      | 0         | 0             |
| ELU                 | 0          | 0      | 0         | 0             |

**Conclusions:**  
...
