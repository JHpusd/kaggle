import pandas as pd
import numpy as np

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
def sibsp_greater_than_zero(entry):
    if entry > 0:
        return 1
    else:
        return 0
df['SibSp>0'] = df['SibSp'].apply(sibsp_greater_than_zero)

print(df['Age'])
