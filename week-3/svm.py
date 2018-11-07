import pandas as pd
import numpy as np
from sklearn.svm import SVC

df_source = pd.read_csv(
    '/Users/antonpiskunov/Programming/yandex-machine-learning/week-3/svm-data.csv', header=None)
x = df_source.loc[:, 1:2]
y = df_source[0]

clf = SVC(C = 100000, kernel='linear', random_state=241)
clf.fit(x, y)
print(df_source)
print(clf.support_vectors_)
