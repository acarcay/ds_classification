CIFAR-10 Image Classification with PyTorch
This project is a straightforward implementation of a Convolutional Neural Network (CNN) using PyTorch to classify images from the CIFAR-10 dataset. The script handles data loading, model training, performance evaluation, and visualization of both the data and training results.

Features
Data Handling: Automatically downloads and prepares the CIFAR-10 dataset using torchvision.

Data Visualization: Includes a function to display a sample of training images with their corresponding labels.

CNN Model: A simple but effective CNN architecture for image classification.

Training Loop: Trains the model and prints the average loss for each epoch.

Loss Curve: Plots a graph of the training loss over epochs after training is complete.

Evaluation: Calculates and displays the final classification accuracy on the test dataset.

GPU/CPU Agnostic: Automatically uses a CUDA-enabled GPU if available, otherwise defaults to the CPU.

Model Architecture
The neural network is a simple CNN with the following layers:

Convolutional Layer 1: 3 input channels, 32 output channels, 3x3 kernel.

ReLU Activation

Max Pooling Layer: 2x2 kernel.

Convolutional Layer 2: 32 input channels, 64 output channels, 3x3 kernel.

ReLU Activation

Max Pooling Layer: 2x2 kernel.

Flatten Layer: Reshapes the tensor for the fully connected layers.

Fully Connected Layer 1: 4096 input features, 128 output features.

ReLU Activation

Dropout Layer: With a probability of 0.2 to prevent overfitting.

Fully Connected Layer 2 (Output): 128 input features, 10 output features (one for each class).

The model uses a Stochastic Gradient Descent (SGD) optimizer and Cross-Entropy Loss as the loss function.

Requirements
To run this project, you'll need Python 3 and the following libraries:

PyTorch

Torchvision

Matplotlib

NumPy

You can install them using pip:

Bash

pip install torch torchvision matplotlib numpy
How to Run
Save the code as a Python file (e.g., cifar_classifier.py).

Open your terminal or command prompt.

Navigate to the directory where you saved the file.

Run the script with the following command:

Bash

python cifar_classifier.py
The script will start by downloading the dataset (if it's not already in a ./data folder), display 10 sample images, then proceed to train the model for 10 epochs. Finally, it will plot the loss curve and print the model's final accuracy on the test set.
