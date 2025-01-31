import numpy as np
import random


cities = ["NY", "LA", "Chicago", "Houston", "Phoenix"]
populations = [800, 400, 250, 230, 150]

probabilities = np.array(populations) / sum(populations)

sampled_cities = np.random.choice(cities, size=10, p=probabilities)
sampled_cities_new = random.choices(cities, weights=populations, k=10)

# print(sampled_cities)
# print(sampled_cities_new)


# in-balanced data
from sklearn.utils import resample
import pandas as pd


data = pd.DataFrame({
    'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
    'population': [800, 400, 250, 230, 150]
})

# uniformly random
upsampled_data = resample(data, replace=True, n_samples=100, random_state=42)
# print(upsampled_data)

downsample_data = resample(upsampled_data, replace=True, n_samples=20, random_state=42)
# print(downsample_data)

# linear interpolation upsampling
x = np.array([1, 3, 7, 10])
y = np.array([2, 6, 14, 20])

x_new = np.linspace(x.min(), x.max(), num=10)
y_new = np.interp(x_new, x, y)

print(np.column_stack((x_new, y_new)))

# SMOTE (use k nearest neighbour for interpolation)
from imblearn.over_sampling import SMOTE

data = pd.DataFrame({
    'feature1': [1, 2, 5, 6, 7],
    'feature2': [3, 4, 8, 9, 10],
    'label': [1, 1, 1, 0, 0]
})

X = data[['feature1', 'feature2']]
y = data['label']

smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=1)
X_resampled, y_resampled = smote.fit_resample(X, y)

resampled_data = pd.DataFrame(X_resampled, columns=['feature1', 'feature2'])
resampled_data['label'] = y_resampled

print(resampled_data)