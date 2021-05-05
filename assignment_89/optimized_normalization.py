import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as knearestclass
import matplotlib.pyplot as plt
import time

begin_time = time.time()

# getting and manipulating data
df = pd.read_csv('/home/runner/kaggle/assignment_89/data.csv')
df = df[["Survived", "Sex", "Pclass", "Fare", "Age","SibSp"]][:100]

print("finished setting up data")
df_time = time.time()
print("time to set up dataframe: "+ str(df_time-begin_time)+'\n')

# helper functions
def leave_one_out_true_false(knn, df, row_index, dv):
    x_df = df[[col for col in df.columns if col != dv]]
    y_df = df[dv]

    classification = df[dv].iloc[[row_index]].to_numpy().tolist()[0]
    values = x_df.iloc[[row_index]].to_numpy().tolist()[0]

    train_df = x_df.drop([row_index])
    train = train_df.reset_index(drop=True).to_numpy().tolist()
    test_df = y_df.drop([row_index])
    test = test_df.reset_index(drop=True).to_numpy().tolist()

    prediction = knn.fit(train, test).predict([values])
    if prediction == classification:
        return True
    return False

def leave_one_out_accuracy(knn, df, dv):
    df_arr = df.to_numpy().tolist()
    correct = 0
    for row_index in range(len(df_arr)):
        if leave_one_out_true_false(knn, df, row_index, dv):
            correct += 1
    return correct / len(df_arr)

# starting the problem
k_vals = [i for i in range(20) if i%2==1]
unnormalized = []
simple_scaling = []
ss_df = df.copy()
min_max = []
mm_df = df.copy()
z_scoring = []
zs_df = df.copy()

for col in [col for col in ss_df if col != 'Survived']:
    ss_df[col] = ss_df[col] / ss_df[col].max()
    mm_df[col] = (mm_df[col]-mm_df[col].min()) / (mm_df[col].max()-mm_df[col].min())
    zs_df[col] = (zs_df[col] - zs_df[col].mean()) / zs_df[col].std()

print('finished normalizing')
normalization_time = time.time()
print('time to normalize: '+str(normalization_time - df_time)+'\n')

for k in k_vals:
    knn = knearestclass(n_neighbors=k)
    unnormalized.append(leave_one_out_accuracy(knn, df, 'Survived'))
    simple_scaling.append(leave_one_out_accuracy(knn, ss_df, 'Survived'))
    min_max.append(leave_one_out_accuracy(knn, mm_df, 'Survived'))
    z_scoring.append(leave_one_out_accuracy(knn, zs_df, 'Survived'))

print('finished adding accuracies')
accuracy_time = time.time()
print('time to add accuracies: '+str(accuracy_time - normalization_time)+'\n')

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
print('time taken:', end_time - begin_time)
