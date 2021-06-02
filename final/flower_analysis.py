import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as kneighbors

df = pd.read_csv('/home/runner/kaggle/final/flower_data.csv')

# flower_species = df.groupby('Species').mean()
# print(flower_species)
df = df.sample(frac=1).reset_index(drop=True)

keep_cols = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
dv_col = df['Species']
df = df[keep_cols]

for col in df.columns:
    df[col] = (df[col]-df[col].min()) / (df[col].max()-df[col].min())

x_train = df[:75]
x_test = df[75:]
y_train = dv_col[:75]
y_test = dv_col[75:]

def knn_accuracy(fitted_knn):
    x_test_list = x_test.to_numpy().tolist()
    y_test_list = y_test.to_numpy().tolist()

    correct = 0
    for i in range(len(x_test_list)):
        prediction = fitted_knn.predict([x_test_list[i]])
        if prediction[0] == y_test_list[i]:
            correct += 1
    return correct / len(x_test_list)
'''
k_vals = [k for k in range(1,21)]
for k in k_vals:
    knn = kneighbors(n_neighbors=k)
    knn.fit(x_train, y_train)
    print('{} nearest neighbors: {}% accuracy'.format(k, knn_accuracy(knn)*100))
'''
# best k value seems to be 6

knn = kneighbors(n_neighbors=6)
knn.fit(x_train, y_train)
print('{}% accuracy with k=6'.format(knn_accuracy(knn)*100))
