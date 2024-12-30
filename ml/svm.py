import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


# for non-linear SVMs, will apply kernal function to input X,
# to increase the dimension of it for a linear plane to divide it
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition > 1:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


iris = datasets.load_iris()
X = iris.data[:, :2]
y = (iris.target != 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# svm_model = SVC(kernel='linear', C=1.0)
svm_model = SVM()
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    w = model.coef_[0]
    b = model.intercept_[0]
    slope = -w[0] / w[1]

    xx_margin = np.linspace(x_min, x_max, 100)
    yy_decision = slope * xx_margin + (-b) / w[1]
    yy_margin1 = slope * xx_margin + (1 - b) / w[1]
    yy_margin2 = slope * xx_margin + (-1 - b) / w[1]

    plt.plot(xx_margin, yy_decision, 'k-', label='Decision Boundary')
    plt.plot(xx_margin, yy_margin1, 'k--', label='Margin 1')
    plt.plot(xx_margin, yy_margin2, 'k--', label='Margin 2')

    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')

    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.title('SVM Decision Boundary with Margins and Support Vectors')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='best')
    plt.show()

plot_decision_boundary(X_test, y_test, svm_model)
