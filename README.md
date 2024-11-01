# README

## Project Overview
This project is a demonstration of a simple feed-forward neural network applied to the Iris dataset. The objective is to showcase the implementation of a neural network using PyTorch and explain how it can be trained and evaluated for classification tasks. The project includes code for data preparation, model training, evaluation, and making predictions.

## Dataset
The dataset used for this project is the classic **Iris dataset**. It contains 150 samples from three different species of Iris flowers (setosa, versicolor, and virginica), with four features each:
- Sepal length
- Sepal width
- Petal length
- Petal width

The target variable is the species of the Iris flower. The dataset is split into training and testing sets to evaluate the model's performance.

## Model Description
The model implemented in this project is a **feed-forward neural network** with the following structure:
- **Input Layer**: Takes in 4 features (input size).
- **Hidden Layer**: Contains 10 neurons and uses a ReLU activation function.
- **Output Layer**: Contains 3 output nodes (one for each class) with a softmax activation function to produce class probabilities.

### Model Architecture
```
Input Layer (4 units)
   -> Hidden Layer (10 units, ReLU activation)
   -> Output Layer (3 units, Softmax activation)
```

### Loss Function and Optimizer
- **Loss Function**: Cross-Entropy Loss, suitable for multi-class classification.
- **Optimizer**: Adam optimizer with a learning rate of 0.01.

### Training
The model is trained for 100 epochs, with the loss being printed every 10 epochs to monitor training progress.

## Code Structure
The project includes the following main files:

1. **`build_dataset.py`**: Contains code to load and preprocess the dataset, split it into training and testing sets, and standardize the features.
2. **`nn_model.py`**: Defines the `SimpleFeedForwardNN` class, as well as the `train_model` and `evaluate_model` functions.
3. **`demo.ipynb`**: Demonstrates the entire process, including loading data, training the model, evaluating it, and making predictions.
4. **`README.md`**: This file, explaining the project and how to use it.

## Usage
### Prerequisites
Ensure you have Python and the following libraries installed:
- `torch`
- `pandas`
- `scikit-learn`
- `numpy`

Install these packages using:
```bash
pip install torch pandas scikit-learn numpy
```

### Running the Code
1. **Clone the repository** and navigate to the project directory.
2. **Run the Jupyter Notebook** (`demo.ipynb`) to execute the entire workflow.
   ```bash
   jupyter notebook example_notebook.ipynb
   ```

### Running Python Scripts Individually
You can run `nn_model.py` directly as a standalone script for training and evaluating the model:
```bash
python nn_model.py
```

### Making Predictions
To make predictions on new data, follow the demonstration in `demo.ipynb`:
1. Load the trained model.
2. Pass test data to the model for inference.
3. Display the predicted classes.

## Results
The model achieves a high level of accuracy on the test set. If the accuracy is 100%, verify that there is no data leakage or overfitting.

## Conclusion
This project demonstrates how to build, train, and evaluate a simple feed-forward neural network using PyTorch. It is a foundational example that can be expanded with more complex architectures or different datasets for further exploration.

## Future Work
- Implement additional regularization techniques to prevent overfitting.
- Experiment with deeper architectures or different activation functions.
- Test on more complex datasets to generalize the approach.



