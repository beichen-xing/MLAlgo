import numpy as np
import numpy.random


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(predictions, labels):
    return -np.sum(labels * np.log(predictions + 1e-9)) / predictions.shape[0]


def compute_accuracy(predictions, labels):
    return np.mean(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))


class ConvLayer:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / (filter_size * filter_size)

    def convolve(self, input_image):
        h, w = input_image.shape
        output_height = h - self.filter_size + 1
        output_width = w - self.filter_size + 1
        output = np.zeros((self.num_filters, output_height, output_width))

        for f in range(self.num_filters):
            for i in range(output_height):
                for j in range(output_width):
                    region = input_image[i: i + self.filter_size, j: j + self.filter_size]
                    output[f, i, j] = np.sum(region * self.filters[f])

        return output

    def forward(self, input_images):
        self.input_images = input_images
        self.outputs = np.array([self.convolve(img) for img in input_images])
        return self.outputs

    def backward(self, grad_output, learning_rate):
        grad_filters = np.zeros_like(self.filters)

        for f in range(self.num_filters):
            for i in range(self.filters.shape[1]):
                for j in range(self.filters.shape[2]):
                    region = self.input_images[:,i: i + self.filter_size,j: j + self.filter_size]
                    grad_filters[f] += np.sum(
                        region * grad_output[:, f, i, j][:, np.newaxis, np.newaxis],
                        axis=0
                    )

        grad_input = np.zeros_like(self.input_images)
        for f in range(self.num_filters):
            flipped_filters = np.flip(self.filters[f], axis=(0, 1))
            for i in range(self.filters.shape[1]):
                for j in range(self.filters.shape[2]):
                    grad_input[:, i: i + self.filter_size, j: j + self.filter_size] += (
                        grad_output[:, f, i, j][:, np.newaxis, np.newaxis] * flipped_filters
                    )

        self.filters -= learning_rate * grad_filters

        return grad_input


class FlattenLayer:
    def forward(self, input_data):
        self.input_shape = input_data.shape
        return input_data.reshape(input_data.shape[0], -1)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)


class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros(output_size)

    def forward(self, input_data):
        self.input_data = input_data
        return np.dot(input_data, self.weights) + self.biases

    def backward(self, grad_output, learning_rate):
        grad_weights = np.dot(self.input_data.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0)

        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return np.dot(grad_output, self.weights.T)


class SimpleCNN:
    def __init__(self):
        self.conv = ConvLayer(num_filters=8, filter_size=3)
        self.flatten = FlattenLayer()
        self.fc = DenseLayer(288, 10)

    def forward(self, x):
        x = self.conv.forward(x)
        x = relu(x)
        x = self.flatten.forward(x)
        x = self.fc.forward(x)
        return softmax(x)

    def backward(self, x, y, learning_rate):
        grad_output = x - y
        grad_output = self.fc.backward(grad_output, learning_rate)
        grad_output = self.flatten.backward(grad_output)
        grad_output = relu_derivative(self.conv.outputs) * grad_output
        grad_output = self.conv.backward(grad_output, learning_rate)


def train_cnn():
    from sklearn.datasets import load_digits
    import matplotlib.pyplot as plt
    digits = load_digits()
    x_data = digits.images / 16.0
    y_data = np.eye(10)[digits.target]

    split_index = int(0.8 * len(x_data))
    x_train, x_test = x_data[:split_index], x_data[split_index:]
    y_train, y_test = y_data[:split_index], y_data[split_index:]

    cnn = SimpleCNN()
    epochs = 10
    learning_rate = 0.01

    for epoch in range(epochs):
        predictions = cnn.forward(x_train)
        loss = cross_entropy_loss(predictions, y_train)

        cnn.backward(predictions, y_train, learning_rate)

        accuracy = compute_accuracy(predictions, y_train)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    test_predictions = cnn.forward(x_test)
    test_accuracy = compute_accuracy(test_predictions, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")


train_cnn()