import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve

##########################################

# True Positives (TP): Correctly predicted positive instances.
# True Negatives (TN): Correctly predicted negative instances.
# False Positives (FP): Incorrectly predicted positives (actual negatives classified as positive).
# False Negatives (FN): Incorrectly predicted negatives (actual positives classified as negative).

# [TN, FP,
#  FN, TP]

# Row 1: Actual positive instances distributed across predictions.
# Row 2: Actual negative instances distributed across predictions.
# Column 1: Predicted positives distributed across actual classes.
# Column 2: Predicted negatives distributed across actual classes.

# Precision: TP / (TP + FP)
# High precision means most of the positive predictions made by the model are correct.

# Recall: TP / (TP + FN) = TPR = TP / Total_Positives
# High recall means the model correctly identifies most of the actual positive cases.

# FPR: FP / (FP + TN)
# A low FPR indicates the model is conservative and avoids false alarms.

# Specificity: TN / (TN + FP) Proportion of true negatives among actual negatives

# ROC Curve: Plots the True Positive Rate (TPR) (also called Sensitivity or Recall)
# against the False Positive Rate (FPR) at various threshold settings -- same model, different threshold
# A perfect model will have a curve that passes through the top-left corner
# A random classifier (e.g., flipping a coin) will produce a diagonal line

# The Precision-Recall (PR) Curve is a graphical representation of a model's performance
# in terms of precision and recall across different classification thresholds.
# Good for imbalanced dataset and when recall are critical.

##################################################


y_true = np.array([0, 0, 1, 1])
y_pred = np.array([0.1, 0.4, 0.35, 0.8])

sorted_indices = np.argsort(y_pred)[::-1]
y_true_sorted = y_true[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]

n_positives = np.sum(y_true)
n_negatives = len(y_true) - n_positives

tpr = []
fpr = []
precision = []
recall = []


tp = 0
fp = 0

for i in range(len(y_true)):
    if y_true_sorted[i] == 1:
        tp += 1
    else:
        fp += 1

    tpr.append(tp / n_positives)
    fpr.append(fp / n_negatives)

    recall.append(tp / n_positives)
    precision.append(tp / (tp + fp))

roc_auc = 0
for i in range(1, len(tpr)):
    roc_auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2

pr_auc = 0
for i in range(1, len(precision)):
    pr_auc += (recall[i] - recall[i - 1]) * (precision[i] + precision[i - 1]) / 2
print(precision, recall)
print('auc', roc_auc, pr_auc, auc(recall, precision))
import matplotlib.pyplot as plt
plt.figure()
plt.plot(precision, recall, label=f"ROC Curve ")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

fpr_, tpr_, thresholds_ = roc_curve(y_true, y_pred)
auc_score = roc_auc_score(y_true, y_pred)
# auc_score_ = auc(fpr_, tpr_)

precision, recall, _ = precision_recall_curve(y_true, y_pred)
print(precision, recall)
pr_auc_score = auc(recall, precision)
print('auc', auc_score, pr_auc_score, thresholds_)

plt.figure()
plt.plot(precision, recall, label=f"ROC Curve (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


