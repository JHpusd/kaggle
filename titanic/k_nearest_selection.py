import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as knearestclass
import matplotlib.pyplot as plt

def leave_one_out_true_false(knn, df, row_index):
    x_df = df[[col for col in df.columns if col != 'Survived']]
    y_df = df['Survived']

    classification = df['Survived'].iloc[[row_index]].to_numpy().tolist()[0]
    values = x_df.iloc[[row_index]].to_numpy().tolist()[0]

    train_df = x_df.drop([row_index])
    train = train_df.reset_index(drop=True).to_numpy().tolist()
    test_df = y_df.drop([row_index])
    test = test_df.reset_index(drop=True).to_numpy().tolist()

    dummy_knn = knn.fit(train, test)
    result_classification = dummy_knn.predict([values])
    if result_classification == classification:
        return True
    return False

def leave_one_out_accuracy(knn, df):
    df_arr = df.to_numpy().tolist()
    correct = 0
    for row_index in range(len(df_arr)):
        if leave_one_out_true_false(knn, df, row_index):
            correct += 1
    return correct / len(df_arr)

df = pd.read_csv('/home/runner/kaggle/titanic/train.csv')
keep_cols = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']
df = df[keep_cols]

# Sex
def sex_to_int(entry):
    if entry == 'male':
        return 0
    elif entry == 'female':
        return 1
    else:
        print('sex to int error')
        return None
df['Sex'] = df['Sex'].apply(sex_to_int)

# Age
age_nan = df['Age'].apply(lambda x: np.isnan(x))
age_not_nan = df['Age'].apply(lambda x: not np.isnan(x))
mean_age = df['Age'][age_not_nan].mean()
df['Age'][age_nan] = mean_age

# SibSp
def greater_than_zero(entry):
    if entry > 0:
        return 1
    else:
        return 0
df['SibSp>0'] = df['SibSp'].apply(greater_than_zero)

# Parch
df['Parch>0'] = df['Parch'].apply(greater_than_zero)
del df['Parch']

# CabinType
def get_cabin_type(entry):
    if entry != 'None':
        return entry[0]
    else:
        return entry

df['Cabin'] = df['Cabin'].fillna('None')
df['CabinType'] = df['Cabin'].apply(get_cabin_type)
for cabin_type in df['CabinType'].unique():
    name = 'CabinType='+cabin_type
    values = df['CabinType'].apply(lambda x: int(x==cabin_type))
    df[name] = values

del df['CabinType']
del df['Cabin']

# Embarked
df['Embarked'] = df['Embarked'].fillna('None')
for embark in df['Embarked'].unique():
    name = 'Embarked='+embark
    values = df['Embarked'].apply(lambda x: int(x==embark))
    df[name] = values

del df['Embarked']

features_to_use = ['Sex','Pclass','Fare','Age','SibSp','SibSp>0','Parch>0','Embarked=C','Embarked=None','Embarked=Q','Embarked=S','CabinType=A','CabinType=B','CabinType=C','CabinType=D','CabinType=E','CabinType=F','CabinType=G','CabinType=None','CabinType=T']
columns = ['Survived'] + features_to_use
df = df[columns]
'''
# 87 part a
plt.style.use('bmh')
# unscaled
k_vals = [1,3,5,10,15,20,30,40,50,75,100,150,200,300,400,600,800]
accuracies = []
for k in k_vals:
    knn = knearestclass(n_neighbors=k)
    accuracies.append(leave_one_out_accuracy(knn, df))
plt.plot(k_vals, accuracies, label='unscaled')

# simple scaling
df_copy = df.copy()
accuracies = []
for col in df_copy.columns:
    df_copy[col] = df[col] / df[col].max()
for k in k_vals:
    knn = knearestclass(n_neighbors=k)
    accuracies.append(leave_one_out_accuracy(knn, df_copy))
plt.plot(k_vals, accuracies, label='simple scaling')

# min-max scaling
df_copy = df.copy()
accuracies = []
for col in df_copy.columns:
    df_copy[col] = (df_copy[col]-df_copy[col].min()) / (df[col].max()-df[col].min())
for k in k_vals:
    knn = knearestclass(n_neighbors=k)
    accuracies.append(leave_one_out_accuracy(knn, df_copy))
plt.plot(k_vals, accuracies, label='min-max scaling')

# z-scoring
df_copy = df.copy()
accuracies = []
for col in df_copy.columns:
    df_copy[col] = (df_copy[col]-df_copy[col].mean()) / df_copy[col].std()
for k in k_vals:
    knn = knearestclass(n_neighbors=k)
    accuracies.append(leave_one_out_accuracy(knn, df_copy))
plt.plot(k_vals, accuracies, label='z-scoring')

plt.xlabel('k')
plt.ylabel('accuracy')
plt.legend(loc='upper right')
plt.savefig('Titanic Survival Modeling with KNN')
'''
df_copy = df[:100]
used_cols = ["Survived","Sex","Pclass","Fare","Age","SibSp"]
df_copy = df_copy[used_cols]
k_vals = [1,3,5,10,15,20,30,40,50,75]
accuracies = []
for k in k_vals:
    knn = knearestclass(n_neighbors=k)
    accuracies.append(leave_one_out_accuracy(knn, df_copy))
plt.style.use('bmh')
plt.plot(k_vals, accuracies)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.savefig('Leave_one_out_cross_validation.png')
