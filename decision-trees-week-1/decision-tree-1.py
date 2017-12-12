from __future__ import print_function
import pandas as pd
import numpy as np
import graphviz
import subprocess
import os
from sklearn.tree import DecisionTreeClassifier, export_graphviz

def visualize_tree(tree, feature_names): #doesn't work for some reason
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

def get_code(tree, feature_names, target_names,
             spacer_base="    "):
    """Produce psuedo-code for decision tree.

    Args
    ----
    tree -- scikit-leant DescisionTree.
    feature_names -- list of feature names.
    target_names -- list of target (class) names.
    spacer_base -- used for spacing code (default: "    ").

    Notes
    -----
    based on http://stackoverflow.com/a/30104792.
    """
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    def recurse(left, right, threshold, features, node, depth):
        spacer = spacer_base * depth
        if (threshold[node] != -2):
            print(spacer + "if ( " + features[node] + " <= " + \
                  str(threshold[node]) + " ) {")
            if left[node] != -1:
                    recurse(left, right, threshold, features,
                            left[node], depth+1)
            print(spacer + "}\n" + spacer +"else {")
            if right[node] != -1:
                    recurse(left, right, threshold, features,
                            right[node], depth+1)
            print(spacer + "}")
        else:
            target = value[node]
            for i, v in zip(np.nonzero(target)[1],
                            target[np.nonzero(target)]):
                target_name = target_names[i]
                target_count = int(v)
                print(spacer + "return " + str(target_name) + \
                      " ( " + str(target_count) + " examples )")

    recurse(left, right, threshold, features, 0, 0)

df_source = pd.read_csv('/Users/antonpiskunov/Programming/yandex-machine-learning/decision-trees-week-1/titanic-dataset.csv')
df = df_source[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]
df['Sex'] = df['Sex'].str.replace("female", "0")
df['Sex'] = df['Sex'].str.replace("male", "1")
df.infer_objects()
df['Sex'] = df['Sex'].astype(int)
df_clean = df.dropna(axis=0, how='any')

features = list(df_clean.columns[:4])
X = df_clean[features]
y = df_clean['Survived']
clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)
importances = clf.feature_importances_
print(features)
print(importances)
visualize_tree(clf,features)
tree.export_graphviz(clf,
     out_file='tree.dot') 

#get_code(clf, features, y)
