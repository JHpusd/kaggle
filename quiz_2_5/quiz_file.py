import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys

df = pd.read_csv('/home/runner/kaggle/quiz_2_5/StudentsPerformance.csv')
# print(df.columns)
# print(sum(list(df['math score'])) / len(list(df['math score'])))

def test_prep_to_int(entry):
    if entry == 'none':
        return 0
    elif entry == 'completed':
        return 1
# df['test preparation course'] = df['test preparation course'].apply(test_prep_to_int)

test_prep = 0
no_test_prep = 0
prep_counter = 0
no_prep_counter = 0
for i in range(len(df['test preparation course'])):
    prep = df['test preparation course'][i]
    if prep == 0:
        no_test_prep += df['math score'][i]
        no_prep_counter += 1
    elif prep == 1:
        test_prep += df['math score'][i]
        prep_counter += 1

parent_educ = []
for i in range(len(df['parental level of education'])):
    if df['parental level of education'][i] not in parent_educ:
        parent_educ.append(df['parental level of education'][i])

df = df[['math score', 'parental level of education', 'test preparation course']]
dummy = pd.get_dummies(df['parental level of education'])
df = pd.concat([df, dummy], axis=1)
del df['parental level of education']
dummy = pd.get_dummies(df['test preparation course'])
df = pd.concat([df, dummy], axis=1)
del df['test preparation course']
df = np.array(df)

data = np.array(df)
train_arr = data[:-3, :]
test_arr = data[-3:, :]

X_train = train_arr[:, 1:]
X_test = test_arr[:, 1:]

y_train = train_arr[:, 0]
y_test = test_arr[:, 0]

regressor = LinearRegression()
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
print(predictions)

