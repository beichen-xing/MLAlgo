import numpy as np


class LinearRegressionSGD:

    def __init__(self, learning_rate=0.01, iterations=1000, batch_size=1):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.batch_size = batch_size
        self.theta = None

    def fit(self, X, y):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        m, n = X.shape
        self.theta = np.zeros(n)

        for _ in range(self.iterations):

            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, m, self.batch_size):
                X_batch = X_shuffled[i: i + self.batch_size]
                y_batch = y_shuffled[i: i + self.batch_size]

                predictions = X_batch.dot(self.theta)
                gradients = (1 / len(X_batch)) * X_batch.T.dot(predictions - y_batch)
                self.theta -= self.learning_rate * gradients

    def predict(self, X):
        X = np.c_[np.one((X.shape[0], 1)), X]
        return X.dot(self.theta)

# closed form is possible, but reverse matrix may be hard to calculated
