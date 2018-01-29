import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer

df_train = pd.read_csv(
    '/Users/antonpiskunov/Programming/yandex-machine-learning/week-4/salary-train.csv')
df_test = pd.read_csv(
    '/Users/antonpiskunov/Programming/yandex-machine-learning/week-4/salary-train.csv')

df_train['LocationNormalized'].fillna('nan', inplace=True)
df_train['ContractTime'].fillna('nan', inplace=True)
df_test['LocationNormalized'].fillna('nan', inplace=True)
df_test['ContractTime'].fillna('nan', inplace=True)

enc = DictVectorizer()
X_train_categ = enc.fit_transform(df_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(df_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

print(df_train.head())
print(df_test.head())
print(X_train_categ)
print(X_test_categ)