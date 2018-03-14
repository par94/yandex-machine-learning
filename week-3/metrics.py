import pandas as pd
import numpy as np
import math
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve

df_source = pd.read_csv(
    '/Users/antonpiskunov/Programming/yandex-machine-learning/week-3/classification.csv')
tp, fp, tn, fn = 0, 0, 0, 0

for index, row in df_source.iterrows():
    if row['true'] == row['pred']:
        if row['true'] == 1:
            tp += 1
        else:
            tn += 1
    else:
        if row['true'] == 1:
            fn += 1
        else:
            fp += 1
print(tp, fp, fn, tn)
print("%.2f" % (accuracy_score(df_source['true'], df_source['pred'])))
print("%.2f" % (precision_score(df_source['true'], df_source['pred'])))
print("%.2f" % (recall_score(df_source['true'], df_source['pred'])))
print("%.2f" % (f1_score(df_source['true'], df_source['pred'])))

df_source = pd.read_csv(
    '/Users/antonpiskunov/Programming/yandex-machine-learning/week-3/scores.csv')
print("%.2f" % (roc_auc_score(df_source['true'], df_source['score_logreg'])))
print("%.2f" % (roc_auc_score(df_source['true'], df_source['score_svm'])))
print("%.2f" % (roc_auc_score(df_source['true'], df_source['score_knn'])))
print("%.2f" % (roc_auc_score(df_source['true'], df_source['score_tree'])))

precision, recall, thresholds = precision_recall_curve(df_source['true'], df_source['score_logreg'])
precision_max_logreg = 0
for i in range(0, len(precision)):
    if recall[i] >= 0.7:
        if precision[i] > precision_max_logreg:
            precision_max_logreg = precision[i]

precision, recall, thresholds = precision_recall_curve(df_source['true'], df_source['score_svm'])
precision_max_svm = 0
for i in range(0, len(precision)):
    if recall[i] >= 0.7:
        if precision[i] > precision_max_svm:
            precision_max_svm = precision[i]

precision, recall, thresholds = precision_recall_curve(df_source['true'], df_source['score_knn'])
precision_max_knn = 0
for i in range(0, len(precision)):
    if recall[i] >= 0.7:
        if precision[i] > precision_max_knn:
            precision_max_knn = precision[i]

precision, recall, thresholds = precision_recall_curve(df_source['true'], df_source['score_tree'])
precision_max_tree = 0
for i in range(0, len(precision)):
    if recall[i] >= 0.7:
        if precision[i] > precision_max_tree:
            precision_max_tree = precision[i]

print("%.2f" % precision_max_logreg)
print("%.2f" % precision_max_svm)
print("%.2f" % precision_max_knn)
print("%.2f" % precision_max_tree)
