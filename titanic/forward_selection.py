import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import sys

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

# interaction features
for i in range(len(features_to_use)):
    item_1 = features_to_use[i]
    for j in range(len(features_to_use)):
        item_2 = features_to_use[j]
        if j <= i or ('SibSp' in item_1 and 'SibSp' in item_2) or ('Embarked=' in item_1 and 'Embarked=' in item_2) or ('CabinType=' in item_1 and 'CabinType=' in item_2):
            continue
        name = item_1 + ' * ' + item_2
        df[name] = df[item_1] * df[item_2]

# training and testing sets
train_df = df[:500]
test_df = df[500:]

# setting up forward selection
def list_apply(input_list, function):
    return [function(x) for x in input_list]

def convert_prediction_to_survival(entry):
    if entry < 0.5:
        return 0
    else:
        return 1

def get_accuracy(predictions, original):
    result = 0
    assert len(predictions)==len(original), 'lists have different lengths'
    for i in range(len(predictions)):
        if predictions[i] == original[i]:
            result += 1
    return result/len(predictions)

def get_set_accuracy(df, used_features, train_or_test):
    if len(used_features) == 0:
        return 0
    cols = ['Survived'] + used_features
    train_arr = np.array(train_df[cols])
    test_arr = np.array(test_df[cols])
    x_train = train_arr[:, 1:]
    y_train = train_arr[:, 0]
    x_test = test_arr[:, 1:]
    y_test = test_arr[:, 0]

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(x_train, y_train)

    train_predictions = log_reg.predict(x_train)
    train_predictions = list_apply(train_predictions, convert_prediction_to_survival)
    test_predictions = log_reg.predict(x_test)
    test_predictions = list_apply(test_predictions, convert_prediction_to_survival)

    if train_or_test == 'train':
        return get_accuracy(train_predictions, y_train)
    return get_accuracy(test_predictions, y_test)

def get_next_feature(df, all_features, used_features):
    all_features_copy = list(all_features)
    used_features_copy = list(used_features)
    for item in used_features:
        all_features_copy.remove(item)
    
    best_col = all_features[0]
    best_test_acc = get_set_accuracy(df, used_features+[best_col], 'test')
    for col in all_features:
        test_acc = get_set_accuracy(df, used_features+[col], 'test')
        if test_acc > best_test_acc:
            best_col = col
            best_test_acc = test_acc

    original_test_acc = get_set_accuracy(df, used_features, 'test')
    if best_test_acc <= original_test_acc:
        return None

    return best_col

all_features = list(df.columns)
all_features = all_features[1:]
used_features = []

next_feature = get_next_feature(df, all_features, used_features)
while next_feature != None:
    used_features.append(next_feature)
    train_acc = get_set_accuracy(df, used_features, 'train')
    test_acc = get_set_accuracy(df, used_features, 'test')
    print(used_features)
    print('training accuracy:',train_acc)
    print('testing accuracy:',test_acc,'\n')
    next_feature = get_next_feature(df, all_features, used_features)

