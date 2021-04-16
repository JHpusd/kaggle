import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
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

dummy = pd.get_dummies(df['parental level of education'])

train = df[:997]
test = df[997:]
