import pandas as pd
import numpy as np

df = pd.read_csv('/home/runner/kaggle/quiz_2_6/data.csv')
# columns: ['enrollee_id', 'city', 'city_development_index', 'gender', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job', 'training_hours', 'target']

# print(sum(df['training_hours']) / len(df['training_hours']))

# print(sum(df['target']) / len(df['target']))

df_copy = df.groupby(['city']).count()
students = df_copy['enrollee_id']
# print(students['city'])
# max_id = students.idxmax()
# print(max_id)
# print(students[max_id])

cities = df['city'].to_numpy().tolist()
cities = [int(i.split('city_')[1]) for i in cities]
# print(max(cities))

df_copy = df.groupby(['company_size']).count()
print(df_copy)
