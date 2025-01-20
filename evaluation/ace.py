import numpy as np


def cal_ace(prob, accuracy, bins=10):
    bin_boundaries = np.linspace(0, 1, bins + 1)
    bin_indices = np.digitize(prob, bin_boundaries) - 1
    ace = 0.0

    for i in range(bins):
        in_bin = bin_indices == i
        if np.sum(in_bin) > 0:
            avg_confidence = np.mean(prob[in_bin])
            avg_accuracy = np.mean(accuracy[in_bin])
            ace += abs(avg_confidence - avg_accuracy)

    return ace / bins

# ACE(Average Calibration Error) is used to measure error between confidence and accuracy
