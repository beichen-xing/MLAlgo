import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier


def gini_impurity(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities ** 2)


def mse_error(y):
    mean_y = np.mean(y)
    return np.mean((y - mean_y) ** 2)


def split_dataset(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask
    return X[left_mask], X[right_mask], y[left_mask], y[right_mask]


def best_split(X, y):
    best_feature = None
    best_threshold = None
    best_impurity = float('inf')
    n_samples, n_features = X.shape

    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            _, _, y_left, y_right = split_dataset(X, y, feature_index, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            # impurity = (len(y_left) / n_samples) * gini_impurity(y_left) \
            #            + (len(y_right) / n_samples) * gini_impurity(y_right)

            impurity = (len(y_left) / n_samples) * mse_error(y_left) \
                       + (len(y_right) / n_samples) * mse_error(y_right)

            if impurity < best_impurity:
                best_feature = feature_index
                best_threshold = threshold
                best_impurity = impurity

    return best_feature, best_threshold


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        if depth == self.max_depth or num_labels == 1 or num_samples <= 1:
            most_common_label = Counter(y).most_common(1)[0][0]
            return {"type": "leaf", "label": most_common_label}

        feature_index, threshold = best_split(X, y)
        if feature_index is None:
            most_common_label = Counter(y).most_common(1)[0][0]
            return {"type": "leaf", "label": most_common_label}

        X_left, X_right, y_left, y_right = split_dataset(X, y, feature_index, threshold)
        left_subtree = self._grow_tree(X_left, y_left, depth + 1)
        right_subtree = self._grow_tree(X_right, y_right, depth + 1)

        return {
            "type": "node",
            "feature_index": feature_index,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree
        }

    def predict(self, X):
        return np.array([self._traverse_tree(sample, self.tree) for sample in X])

    def _traverse_tree(self, x, tree):
        if tree["type"] == "leaf":
            return tree["label"]

        feature_index = tree["feature_index"]
        threshold = tree["threshold"]

        if x[feature_index] <= threshold:
            return self._traverse_tree(x, tree["left"])
        else:
            return self._traverse_tree(x, tree["right"])


# random forest
class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=tree_predictions)


# adaboost
class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.models = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        weights = np.ones(n_samples) / n_samples
        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=1)
            tree.fit(X, y)
            predictions = tree.predict(X)

            err = np.sum(weights * (predictions != y)) / np.sum(weights)

            alpha = 0.5 * np.log((1 - err) / (err + 1e-10))
            self.alphas.append(alpha)
            self.models.append(tree)

            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)

    def predict(self, X):
        model_preds = np.array([alpha * model.predict(X) for alpha, model in zip(self.alphas, self.models)])
        return np.sign(np.sum(model_preds, axis=0))


# gradient boosting
class GradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.initial_prediction = None

    def fit(self, X, y):
        self.initial_prediction = np.mean(y)
        residual = y - self.initial_prediction

        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=3)
            tree.fit(X, residual)
            predictions = tree.predict(X)

            residual -= self.learning_rate * predictions
            self.models.append(tree)

    def predict(self, X):
        prediction = self.initial_prediction
        for tree in self.models:
            prediction += self.learning_rate * tree.predict(X)
        return prediction


# xgboost



# use sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor

import xgboost as xgb

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',
    'max_depth': 4,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}
model = xgb.train(params, dtrain, 100)

# model = RandomForest(n_estimators=10, max_depth=5)
# model = RandomForestClassifier(n_estimators=10, random_state=42)
# model = AdaBoost(n_estimators=10)
# model = AdaBoostClassifier(n_estimators=10, random_state=42)
# model = GradientBoosting(n_estimators=10, learning_rate=0.1)
# model = GradientBoostingRegressor(n_estimators=10, learning_rate=0.1, random_state=42)
# model.fit(X_train, y_train)


y_pred = model.predict(dtest)
mse_gb = mean_squared_error(y_test, y_pred)
print(f"Gradient Boosting MSE: {mse_gb:.2f}")
# accuracy_ab = accuracy_score(y_test, y_pred)
# print(f"AdaBoost Accuracy (sklearn): {accuracy_ab:.2f}")
