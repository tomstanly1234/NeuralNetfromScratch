# Handwritten Digit Recognition with NumPy Neural Network

This project implements a simple **feedforward neural network (from scratch using NumPy)** to classify handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).  

The network is trained using **stochastic gradient descent** with backpropagation and ReLU activations, and achieves decent test accuracy without relying on high-level deep learning libraries like TensorFlow or PyTorch.  

## 📂 Project Structure
.
├── train-images.idx3-ubyte   # MNIST training images
├── train-labels.idx1-ubyte   # MNIST training labels
├── t10k-images.idx3-ubyte    # MNIST test images
├── t10k-labels.idx1-ubyte    # MNIST test labels
└── main.py            # Main Python script

## 🚀 Features
- Loads MNIST dataset in IDX format directly.
- 3-layer fully connected neural network:
  - Input layer: 784 units (28×28 images flattened)
  - Hidden layer 1: 128 units (ReLU)
  - Hidden layer 2: 64 units (ReLU)
  - Output layer: 10 units (softmax)
- Implements **forward propagation**, **cross entropy loss**, and **backpropagation** manually.
- Trains using **mini-batch gradient descent**.
- Reports test accuracy per epoch.
- Includes plotting support (predictions vs. true labels or confusion matrix).

## 📦 Requirements
Install the required Python libraries:
```bash
pip install numpy matplotlib seaborn scikit-learn
