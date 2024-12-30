import numpy as np
from collections import defaultdict


def stratified_k_fold_split(X, y, n_splits=5, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    class_indices = defaultdict(list)
    for idx, label in enumerate(y):
        class_indices[label].append(idx)

    for label in class_indices:
        np.random.shuffle(class_indices[label])

    folds = [[] for _ in range(n_splits)]
    for label, indices in class_indices.items():
        for i, idx in enumerate(indices):
            folds[i % n_splits].append(idx)

    stratified_folds = []
    for i in range(n_splits):
        test_indices = folds[i]
        train_indices = [idx for j in range(n_splits) if j != i for idx in folds[j]]
        stratified_folds.append((train_indices, test_indices))

    return stratified_folds


X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

folds = stratified_k_fold_split(X, y, n_splits=3, random_seed=42)
# print(folds)


from sklearn.model_selection import StratifiedKFold
n_splits = 3
stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
# for fold_idx, (train_idx, test_idx) in enumerate(stratified_kfold.split(X, y)):
#     print(f"Fold {fold_idx + 1}")
#     print(f"Train indices: {X[train_idx]}")
#     print(f"Test indices: {X[test_idx]}")


# cross-validation
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X, y = iris.data, iris.target
clf = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(clf, X, y, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())

cv_scores = cross_val_score(clf, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
print("Cross-validation scores with StratifiedKFold:", cv_scores)
print("Mean CV score with StratifiedKFold:", cv_scores.mean())


from sklearn.model_selection import StratifiedGroupKFold
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
groups = np.array([1, 1, 1, 0, 0, 0, 1, 2, 3, 4])  # Groups (e.g., patients)

# StratifiedGroupKFold
sgkf = StratifiedGroupKFold(n_splits=3)
for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups)):
    print(f"Fold {fold_idx + 1} (StratifiedGroupKFold)")
    print(f"Train indices: {train_idx}")
    print(f"Test indices: {test_idx}\n")
