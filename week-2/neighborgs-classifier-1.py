from __future__ import print_function
import pandas as pd
import numpy as np
import graphviz
import subprocess
import os
from sklearn.tree import DecisionTreeClassifier, export_graphviz

df_source = pd.read_csv('wine.data', sep=',', header=None)
print(df_source.head())