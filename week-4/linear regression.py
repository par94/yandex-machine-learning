import pandas as pd
import numpy as np
import os
from sklearn import datasets
from scipy.sparse import coo_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

df_train = pd.read_csv(os.path.join(__location__, 'salary-train.csv'))
df_train['FullDescription'] = df_train['FullDescription'].str.lower()
df_train['FullDescription'] = df_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)

vec = TfidfVectorizer(min_df=5)
X_train_text = vec.fit_transform(df_train['FullDescription'])

df_train['LocationNormalized'].fillna('nan', inplace=True)
df_train['ContractTime'].fillna('nan', inplace=True)

enc = DictVectorizer()
X_train_categ = enc.fit_transform(df_train[['LocationNormalized', 'ContractTime']].to_dict('records'))

X_train = hstack([X_train_text,X_train_categ])
y_train = df_train['SalaryNormalized']

linreg = Ridge(alpha=1.0)
linreg.fit(X_train,y_train)

df_test = pd.read_csv(os.path.join(__location__, 'salary-test-mini.csv'))

df_test['FullDescription'] = df_train['FullDescription'].str.lower()
df_test['FullDescription'] = df_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
X_test_text = vec.transform(df_test['FullDescription'])

X_test_categ = enc.transform(df_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test = hstack([X_test_text,X_test_categ])

y_test = linreg.predict(X_test)




#print(len(y_test))

#print(X_train)
#print(X_test)

print(y_test)