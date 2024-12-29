import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        distance = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = np.argsort(distance)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


X_train = [[1, 2], [2, 3], [3, 4], [5, 6]]
y_train = [0, 0, 1, 1]

X_test = [[3, 3], [4, 5]]
# knn = KNN(k=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
print("Predictions:", predictions)