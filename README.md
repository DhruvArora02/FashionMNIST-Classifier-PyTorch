# FashionMNIST-Classifier-PyTorch

__Project Overview__ -

This project demonstrates how to build and train a neural network model for classifying the **FashionMNIST** dataset using **PyTorch**. The model is trained using hierarchical techniques such as **SGD optimization**, and it includes a **fully connected neural network** with ReLU activations. It evaluates the model's accuracy and loss, and predicts the top 3 class labels for a given test image.

__Key Features__ -
- **Data Loading**: Loads the FashionMNIST dataset using PyTorch's DataLoader, with transformations such as normalization.
- **Neural Network Model**: Implements a feedforward neural network with multiple layers for image classification.
- **Model Training**: Trains the model using stochastic gradient descent (SGD) and evaluates its performance on the training set.
- **Model Evaluation**: Evaluates the model on a test set, printing the accuracy and loss.
- **Prediction**: Uses the trained model to predict the top 3 class labels for an image from the test set.

__Technologies Used__ -
- **PyTorch**: Framework for building and training the neural network model.
- **torchvision**: Used to load and preprocess the FashionMNIST dataset.
- **Python**: The main programming language used for the implementation.

__How It Works__ -

1. **Data Preprocessing**: The FashionMNIST dataset is loaded, and images are normalized for better model performance.
2. **Model Architecture**: A fully connected neural network is built using PyTorch's `nn.Sequential`. The architecture includes:
   - Flatten layer
   - Two dense layers with ReLU activations
   - A final dense layer for classification into 10 classes
3. **Training**: The model is trained for 5 epochs using the **SGD** optimizer and **CrossEntropyLoss** criterion.
4. **Evaluation**: After training, the model's accuracy and loss on the test set are computed and displayed.
5. **Prediction**: The trained model predicts the top 3 class labels for a sample image from the test set.
