import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X = np.array([
    [1,2], [1,4], [1,0],
    [10,2], [10,4], [10,0]
])

kmeans = KMeans(n_clusters=2)

kmeans.fit(X)

labels = kmeans.labels_

print("Cluster Labels:", labels)

plt.scatter(X[:,0], X[:,1], c=labels)
plt.title("K-Means Clustering (Unsupervised Learning)")
plt.show()
