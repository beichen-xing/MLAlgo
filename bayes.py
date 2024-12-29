from collections import defaultdict

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


class GaussianNBFromScratch:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.class_prior = {}

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0)
            self.class_prior[c] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        predictions = []
        for x in X:
            log_probs = {}
            for c in self.classes:
                log_prior = np.log(self.class_prior[c])
                log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.var[c])) - \
                                 0.5 * np.sum(((x - self.mean[c]) ** 2) / self.var[c])
                log_probs[c] = log_prior + log_likelihood
            predictions.append(max(log_probs, key=log_probs.get))

        return np.array(predictions)


class MultinomialNBFromScratch:
    def __init__(self):
        self.class_log_prior = {}
        self.feature_log_prob = {}

    def fit(self, X, y, vocab_size):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self.feature_count = defaultdict(lambda: np.zeros(n_features))
        self.class_count = defaultdict(int)

        for i in range(n_samples):
            self.class_count[y[i]] += 1
            self.feature_count[y[i]] += X[i]

        for c in self.classes:
            self.class_log_prior[c] = np.log(self.class_count[c] / n_samples)
            total_count = self.feature_count[c].sum() + vocab_size
            self.feature_log_prob[c] = np.log((self.feature_count[c] + 1) / total_count)

    def predict(self, X):
        predictions = []
        for x in X:
            log_probs = {c: self.class_log_prior[c] + (x * self.feature_log_prob[c].T) for c in self.classes}
            predictions.append(max(log_probs, key=log_probs.get))
        return np.array(predictions)


data = {
    'text': [
        'Free money now!!!',
        'Hey, how are you doing?',
        'Win a free trip to Paris',
        'Important meeting tomorrow',
        'Claim your free prize now',
        'Let’s catch up soon'
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}

# data = {
#     'feature1': [2.3, 1.9, 3.6, 2.7, 1.6, 3.1, 2.0, 1.8],
#     'feature2': [4.5, 3.2, 5.1, 4.8, 2.9, 5.3, 3.8, 3.0],
#     'label': [1, 0, 1, 1, 0, 1, 0, 0]
# }

df = pd.DataFrame(data)
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
# X = df[['feature1', 'feature2']].values.astype(float)
y = df['label'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# classifier = MultinomialNB()
# classifier.fit(X_train, y_train)
# classifier = GaussianNB()
# classifier = GaussianNBFromScratch()
classifier = MultinomialNBFromScratch()
classifier.fit(X_train, y_train, len(vectorizer.vocabulary_))

y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:", classification_report(y_test, y_pred))

new_emails = ["You won a lottery! Claim now!", "Let’s schedule a meeting."]
new_features = vectorizer.transform(new_emails)
predictions = classifier.predict(new_features)

for email, label in zip(new_emails, predictions):
    print(f"Email: '{email}' - {'Spam' if label == 1 else 'Ham'}")

# new_data = np.array([[3.0, 4.7], [1.5, 3.0]])
# predictions = classifier.predict(new_data)
#
# for features, label in zip(new_data, predictions):
#     print(f"Features: {features} - {'Positive' if label == 1 else 'Negative'}")
