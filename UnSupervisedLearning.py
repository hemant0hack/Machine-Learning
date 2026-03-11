# Import libraries
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Sample dataset
X = np.array([
    [1,2], [1,4], [1,0],
    [10,2], [10,4], [10,0]
])

# Create KMeans model (2 clusters)
kmeans = KMeans(n_clusters=2)

# Train the model
kmeans.fit(X)

# Get cluster labels
labels = kmeans.labels_

print("Cluster Labels:", labels)

# Plot clusters
plt.scatter(X[:,0], X[:,1], c=labels)
plt.title("K-Means Clustering (Unsupervised Learning)")
plt.show()