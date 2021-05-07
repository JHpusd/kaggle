import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as knearestclass
import matplotlib.pyplot as plt
import time

begin_time = time.time()

# getting and manipulating data
df = pd.read_csv('/home/runner/kaggle/assignment_89/data.csv')
features = ["Sex", "Pclass", "Fare", "Age","SibSp"]
x_df = df[features][:100]
y_df = df['Survived'][:100]
y_arr = y_df.to_numpy().tolist()

print("finished setting up data")
df_time = time.time()
print("time to set up dataframe: "+ str(df_time-begin_time)+'\n')

# helper functions
def leave_one_out_true_false(knn, x_arr, y_arr, row_index):
    x_copy = list(x_arr)
    y_copy = list(y_arr)
    classification = y_copy[row_index]
    values = x_copy[row_index]

    x_copy.pop(row_index)
    y_copy.pop(row_index)

    prediction = knn.fit(x_copy, y_copy).predict([values])
    if prediction == classification:
        return True
    return False

def leave_one_out_accuracy(knn, x_arr, y_arr):
    correct = 0
    for row_index in range(len(x_arr)):
        if leave_one_out_true_false(knn, x_arr, y_arr, row_index):
            correct += 1
    return correct / len(x_arr)

# starting the problem
k_vals = [i for i in range(100) if i%2==1]
unnormalized = []
simple_scaling = []
ss_df = x_df.copy()
min_max = []
mm_df = x_df.copy()
z_scoring = []
zs_df = x_df.copy()

for col in [col for col in x_df]:
    ss_df[col] = ss_df[col] / ss_df[col].max()
    mm_df[col] = (mm_df[col]-mm_df[col].min()) / (mm_df[col].max()-mm_df[col].min())
    zs_df[col] = (zs_df[col] - zs_df[col].mean()) / zs_df[col].std()

print('finished normalizing')
normalization_time = time.time()
print('time to normalize: '+str(normalization_time - df_time)+'\n')

unn_x = x_df.to_numpy().tolist()
ss_x = ss_df.to_numpy().tolist()
mm_x = mm_df.to_numpy().tolist()
zs_x = zs_df.to_numpy().tolist()

print('finished converting dataframes to lists')
conversion_time = time.time()
print('time to convert: '+str(conversion_time-normalization_time)+'\n')

for k in k_vals:
    knn = knearestclass(n_neighbors=k)
    unnormalized.append(leave_one_out_accuracy(knn, unn_x, y_arr))
    simple_scaling.append(leave_one_out_accuracy(knn, ss_x, y_arr))
    min_max.append(leave_one_out_accuracy(knn, mm_x, y_arr))
    z_scoring.append(leave_one_out_accuracy(knn, zs_x, y_arr))

print('finished adding accuracies')
accuracy_time = time.time()
print('time to add accuracies: '+str(accuracy_time - conversion_time)+'\n')

plt.style.use('bmh')
plt.plot(k_vals, unnormalized, label='unnormalized')
plt.plot(k_vals, simple_scaling, label='simple sclaing')
plt.plot(k_vals, min_max, label='min-max')
plt.plot(k_vals, z_scoring, label='z-scoring')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Leave-One-Out Accuracy for Various Normalizations')
plt.legend(loc='best')
plt.savefig('optimized_normalizaion_accuracies.png')

print('finished plotting')
plot_time = time.time()
print('time to plot: '+str(plot_time - accuracy_time)+'\n')

end_time = time.time()
total_time = end_time - begin_time
print('time taken:', total_time)
print('relative goal:', 1.958076667*45)
