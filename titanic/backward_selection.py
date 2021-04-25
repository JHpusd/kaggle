import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import sys
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

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

# setting up backwards selection
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
    train_df = df[:500]
    test_df = df[500:]
    if len(used_features) == 0:
        return 0
    cols = ['Survived'] + used_features
    train_arr = np.array(train_df[cols])
    test_arr = np.array(test_df[cols])
    x_train = train_arr[:, 1:]
    y_train = train_arr[:, 0]
    x_test = test_arr[:, 1:]
    y_test = test_arr[:, 0]

    log_reg = LogisticRegression(max_iter=100, random_state=0)
    log_reg.fit(x_train, y_train)

    train_predictions = log_reg.predict(x_train)
    train_predictions = list_apply(train_predictions, convert_prediction_to_survival)
    test_predictions = log_reg.predict(x_test)
    test_predictions = list_apply(test_predictions, convert_prediction_to_survival)

    if train_or_test == 'train':
        return get_accuracy(train_predictions, y_train)
    return get_accuracy(test_predictions, y_test)

def create_removed_df(df, remove_item):
    return df[[item for item in df.columns if item != remove_item]]

all_features = list(df.columns)
all_features = all_features[1:]
print("EVERYTHING:")
# print("all features", all_features)
removed_indicies = []
print("training:", get_set_accuracy(df, all_features, 'train'))
print("testing:", get_set_accuracy(df, all_features, 'test'))
print("removed indicies:", removed_indicies, "\n")

print("PRUNING:")
baseline_test_acc = get_set_accuracy(df, all_features, 'test')
for index, item in enumerate(all_features):
    # print("length of all features:",len(all_features))
    print("candidate for removal: "+str(item)+" (index "+str(index)+")")
    removed_df = create_removed_df(df, item)
    removed_list = list(all_features)
    removed_list.remove(item)
    # print("length of removed list:",len(removed_list))
    train_acc = get_set_accuracy(removed_df, removed_list, 'train')
    test_acc = get_set_accuracy(removed_df, removed_list, 'test')
    print("training:",train_acc)
    print("testing:",test_acc)

    if test_acc < baseline_test_acc:
        print("kept")
    else:
        print("removed")
        baseline_test_acc = test_acc
        removed_indicies.append(index)
        all_features = removed_list
        df = removed_df
    print("baseline testing accuracy:",baseline_test_acc)
    # print("length of all features:",len(all_features))
    print("removed indicies:",removed_indicies,"\n")
print("final training:",get_set_accuracy(df, all_features, 'train'))
print("final testing:",get_set_accuracy(df, all_features, 'test'))
