import pandas as pd
from sklearn.cluster import KMeans
path = r"/content/Instagram visits clustering.csv"
df= pd.read_csv(path)
X= df[['Instagram visit score']]
model= KMeans(n_clusters=3,random_state=0)
model.fit(X)
df['bp_cluster']= model.labels_
pd.set_option('display.max_rows',None)
print(df[['Instagram visit score','bp_cluster']])
print("\nCluster Centers (BP values):")
print(model.cluster_centers_)