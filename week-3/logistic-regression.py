import pandas as pd
import numpy as np
import os
import math
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

df_source = pd.read_csv(
    os.path.join(__location__, 'data-logistic.csv'), header=None)
x = df_source.loc[:, 1:2]
y = df_source[0]

def fw1(w1, w2, x, y, c, k):
    l = len(y)
    S = 0
    for i in range(0,l):
        S += y[i] * x[1][i] * (1 - 1 / (1 + math.exp(-y[i] * (w1 * x[1][i] + w2 * x[2][i]))))
    w1_upd = w1 + S * k / l - k * c * w1
    return w1_upd

def fw2(w1, w2, x, y, c, k):
    l = len(y)
    S = 0
    for i in range(0,l):
        S += y[i] * x[2][i] * (1 - 1 / (1 + math.exp(-y[i] * (w1 * x[1][i] + w2 * x[2][i]))))
    w2_upd = w2 + S * k / l - k * c * w2
    return w2_upd

def grad_desc(x, y, c=0.0, w1=0.0, w2=0.0, k=0.1, error=1e-5, n_max = 10000):
    i = 0
    test_error = error + 1
    while (error < test_error) and (i < n_max):
        w1_upd, w2_upd = fw1(w1, w2, x, y, c, k), fw2(w1, w2, x, y, c, k)
        test_error = math.sqrt((w1_upd - w1) ** 2 + (w2_upd - w2) ** 2)
        w1, w2 = w1_upd, w2_upd
    return [w1, w2]

w1, w2 = grad_desc(x, y)
rw1, rw2 = grad_desc(x, y, c=10)

def a(x, w1, w2):
    return 1 / (1 + math.exp(-w1 * x[1] - w2 * x[2]))

y_score = x.apply(lambda x: a(x, w1, w2), axis=1)
y_rscore = x.apply(lambda x: a(x, rw1, rw2), axis=1)
auc = roc_auc_score(y, y_score)
rauc = roc_auc_score(y, y_rscore)

print(auc, rauc)