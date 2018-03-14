import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score

ds = datasets.load_boston(return_X_y=True)
X = pd.DataFrame(data = ds[0])
Y = pd.DataFrame(data = ds[1])
X_array = scale(X)
X_scaled = pd.DataFrame(data=X_array)
p_array = np.linspace(1, 10, num=200, endpoint=True)
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
list_cv_score = []
for p in p_array:
    neigh = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=p)
    cv_score = cross_val_score(neigh, X_scaled, y=Y, cv=k_fold, scoring='neg_mean_squared_error')
    avg_cv_score = np.mean(cv_score)
    dict_cv_score = {}
    dict_cv_score.update({'p':p, 'score':avg_cv_score})
    list_cv_score.append(dict_cv_score)
df_cv_score = pd.DataFrame(list_cv_score)
print(df_cv_score.idxmax())
print(df_cv_score.max())
print(df_cv_score)