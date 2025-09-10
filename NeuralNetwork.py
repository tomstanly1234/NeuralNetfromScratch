import numpy as np
import struct
import matplotlib.pyplot as plt
def load_idx_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num_images, rows, cols)

def load_idx_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_items = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data
X_train = load_idx_images("train-images.idx3-ubyte").reshape(-1, 784) / 255.0
y_train = load_idx_labels("train-labels.idx1-ubyte")
X_test = load_idx_images("t10k-images.idx3-ubyte").reshape(-1, 784) / 255.0
y_test = load_idx_labels("t10k-labels.idx1-ubyte")
np.random.seed(42)
w1 = np.random.randn(784, 128) * 0.01
b1 = np.zeros(128)
w2 = np.random.randn(128, 64) * 0.01
b2 = np.zeros(64)
w3 = np.random.randn(64, 10) * 0.01
b3 = np.zeros(10)
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

def forward(X):
    z1 = X @ w1 + b1
    a1 = relu(z1)
    z2 = a1 @ w2 + b2
    a2 = relu(z2)
    z3 = a2 @ w3 + b3
    out = softmax(z3)
    return out, a1, a2, z1, z2
def cross_entropy_loss(y_true_labels, y_prob):
    m = y_prob.shape[0]
    p = y_prob[np.arange(m), y_true_labels]
    return -np.mean(np.log(p + 1e-12))
def backpropogate(x, y, out, A1, A2, Z1, Z2):
    global w1, b1, w2, b2, w3, b3
    m = x.shape[0]
    Y_onehot = np.zeros((y.size, 10))
    Y_onehot[np.arange(y.size), y] = 1
    dZ3 = out - Y_onehot
    dW3 = (A2.T @ dZ3) / m
    dB3 = np.sum(dZ3, axis=0) / m
    dA2 = dZ3 @ w3.T
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = (A1.T @ dZ2) / m
    dB2 = np.sum(dZ2, axis=0) / m
    dA1 = dZ2 @ w2.T
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (x.T @ dZ1) / m
    dB1 = np.sum(dZ1, axis=0) / m

    return dW1, dB1, dW2, dB2, dW3, dB3
def update_params(lr, grads):
    global w1, b1, w2, b2, w3, b3
    dW1, dB1, dW2, dB2, dW3, dB3 = grads
    w1 -= lr * dW1
    b1 -= lr * dB1
    w2 -= lr * dW2
    b2 -= lr * dB2
    w3 -= lr * dW3
    b3 -= lr * dB3
def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_pred, axis=1) == y_true)
epochs = 5
batch_size = 64
lr = 0.66
for epoch in range(epochs):
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)

    for start in range(0, X_train.shape[0], batch_size):
        end = start + batch_size
        batch_X = X_train[indices[start:end]]
        batch_Y = y_train[indices[start:end]]
        out, a1, a2, z1, z2 = forward(batch_X)
        loss = cross_entropy_loss(batch_Y, out)
        grads = backpropogate(batch_X, batch_Y, out, a1, a2, z1, z2)
        update_params(lr, grads)
    test_out, *_ = forward(X_test)
    test_acc = accuracy(y_test, test_out)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Test Accuracy: {test_acc*100:.2f}%")
    plt.scatter(test_out,y_test)
    plt.show()
