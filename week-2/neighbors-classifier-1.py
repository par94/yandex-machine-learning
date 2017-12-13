import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold, cross_val_score

df_source = pd.read_csv('/Users/antonpiskunov/Programming/yandex-machine-learning/week-2/wine.data', sep=',', header=None)
y = df_source[0]
x = df_source.loc[:, 1:13]
x_array = scale(x)
x_scaled = pd.DataFrame(data=x_array)
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
list_cv_score = []
for k in range(1, 51):
    clf = KNeighborsClassifier(n_neighbors=k)
    cv_score = cross_val_score(clf, x_scaled, y=y, cv=k_fold)
    avg_cv_score = np.mean(cv_score)
    dict_cv_score = {}
    dict_cv_score.update({'k':k, 'score':avg_cv_score})
    list_cv_score.append(dict_cv_score)
df_cv_score = pd.DataFrame(list_cv_score)
print(df_cv_score.idxmax())
print(df_cv_score.max())
