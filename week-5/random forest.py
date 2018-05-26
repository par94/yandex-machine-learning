import pandas as pd
import numpy as np
import math
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

df = pd.read_csv(os.path.join(__location__,'abalone.csv'))
df['Sex'] = df['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
X = df.iloc[:, 0:8]
print(X.head())
y = df['Rings']

k_fold = KFold(n_splits=5, shuffle=True, random_state=1)
list_cv_score = []

for k in range(1, 51):
    clf = RandomForestRegressor(n_estimators=k,random_state=1)
    cv_score = cross_val_score(clf, X, y=y, cv=k_fold, scoring='r2')
    avg_cv_score = np.mean(cv_score)
    dict_cv_score = {}
    dict_cv_score.update({'k':k, 'score':avg_cv_score})
    list_cv_score.append(dict_cv_score)
df_cv_score = pd.DataFrame(list_cv_score)
print(df_cv_score)
print(df_cv_score.idxmax())
print(df_cv_score.max())