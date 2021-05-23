import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('/home/runner/kaggle/titanic/edited_train.csv')
keep_cols = ["Sex", "Pclass", "Fare", "Age", "SibSp"]
new_df = df[keep_cols]
result_df = df[keep_cols + ['Survived']]

for col in new_df:
    new_df[col] = (new_df[col]-new_df[col].min()) / (new_df[col].max()-new_df[col].min())

k_vals = [k for k in range(1,26)]
error = []

data = new_df.to_numpy()

for k in k_vals:
    kmeans = KMeans(n_clusters=k).fit(data)
    error.append(kmeans.inertia_)
plt.style.use('bmh')
plt.plot(k_vals, error)
plt.xticks(k_vals)
plt.ylabel('sum squared distance from cluster center')
plt.xlabel('k value')
plt.ylim(0, 425)
plt.savefig('titanic_model_kmeans_clustering_error.png')

# using k=4
kmeans = KMeans(n_clusters=4, random_state=0).fit(data)

def get_cluster_len(cluster_num):
    return list(kmeans.labels_).count(cluster_num)

result_df['cluster'] = list(kmeans.labels_)
result_df['count'] = result_df['cluster'].apply(get_cluster_len)
cluster_df = result_df.groupby(['cluster']).mean()
print(cluster_df)
