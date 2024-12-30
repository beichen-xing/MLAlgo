import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def kmeans(data, k, max_iters=100, tol=1e-4):
    n_samples, n_features = data.shape
    centroids = data[np.random.choice(n_samples, k, replace=False)]

    for iteration in range(max_iters):
        distance = np.linalg.norm(data[:, None] - centroids, axis=2)
        labels = np.argmin(distance, axis=1)

        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        if np.all(np.abs(new_centroids - centroids) < tol):
            break

        centroids = new_centroids

    return centroids, labels


np.random.seed(42)
cluster_1 = np.random.randn(50, 2) + [2, 2]
cluster_2 = np.random.randn(50, 2) + [7, 7]
cluster_3 = np.random.randn(50, 2) + [12, 2]
data = np.vstack((cluster_1, cluster_2, cluster_3))

k = 3
# centroids, labels = kmeans(data, k)
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(data)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.title("K-Means Clustering")
plt.legend()
plt.show()

