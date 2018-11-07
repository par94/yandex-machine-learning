import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df_train_source = pd.read_csv(
    '/Users/antonpiskunov/Programming/yandex-machine-learning/week-2/perceptron-train.csv', header=None)
df_test_source = pd.read_csv(
    '/Users/antonpiskunov/Programming/yandex-machine-learning/week-2/perceptron-test.csv', header=None)
x_train = df_train_source.loc[:, 1:2]
y_train = df_train_source[0]
x_test = df_test_source.loc[:, 1:2]
y_test = df_test_source[0]

percp = Perceptron(random_state=241)
percp.fit(x_train, y_train)
y_pred = percp.predict(x_test)
unscaled_score = accuracy_score(y_test,y_pred)
print(unscaled_score)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
percp.fit(x_train_scaled, y_train)
y_pred = percp.predict(x_test_scaled)
scaled_score = accuracy_score(y_test,y_pred)
print(scaled_score)
print(scaled_score-unscaled_score)

