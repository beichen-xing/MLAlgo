import numpy as np
import numpy.random


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    weights1 = np.random.randn(input_size, hidden_size) * 0.01
    biases1 = np.zeros((1, hidden_size))

    weights2 = np.random.randn(hidden_size, output_size) * 0.01
    biases2 = np.zeros((1, output_size))

    return weights1, biases1, weights2, biases2


def forward_pass(X, weights1, biases1, weights2, biases2):
    Z1 = np.dot(X, weights1) + biases1
    A1 = relu(Z1)
    Z2 = np.dot(A1, weights2) + biases2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def backward_pass(X, Y, Z1, A1, A2, weights2):
    m = X.shape[0] # batch size
    dZ2 = A2 - Y # cross-entropy(softmax(Z))
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, weights2.T) # dL/dA1 = dL/dZ2 * dZ2/dA1
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2


def update_parameters(weights1, biases1, weights2, biases2, dW1, db1, dW2, db2, learning_rate):
    weights1 -= learning_rate * dW1
    biases1 -= learning_rate * db1
    weights2 -= learning_rate * dW2
    biases2 -= learning_rate * db2
    return weights1, biases1, weights2, biases2


def train_nn(X, Y, input_size, hidden_size, output_size, learning_rate, epochs):
    weights1, biases1, weights2, biases2 = initialize_parameters(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        Z1, A1, Z2, A2 = forward_pass(X, weights1, biases1, weights2, biases2)

        loss = -np.mean(np.sum(Y * np.log(A2 + 1e-8), axis=1))

        dW1, db1, dW2, db2 = backward_pass(X, Y, Z1, A1, A2, weights2)

        weights1, biases1, weights2, biases2 = update_parameters(weights1, biases1, weights2, biases2, dW1, db1, dW2, db2, learning_rate)

        if epoch % 100 == 0:
            print(loss)

    return weights1, biases1, weights2, biases2


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(300, 3)
    Y = np.eye(2)[np.random.choice(2, 300)]

    input_size = 3
    hidden_size = 5
    output_size = 2
    learning_rate = 0.01
    epochs = 1000

    trained_params = train_nn(X, Y, input_size, hidden_size, output_size, learning_rate, epochs)
