import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

newsgroups = datasets.fetch_20newsgroups(
    subset='all', 
    categories=['alt.atheism', 'sci.space']
)
x = newsgroups.data
y = newsgroups.target

vectorizer = TfidfVectorizer()
x_transformed =  vectorizer.fit_transform(x)

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf_test = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf_test, grid, scoring='accuracy', cv=cv)
gs.fit(x_transformed, y)
c_max = 0
score_max = 0
for a in gs.grid_scores_:
    if (score_max < a.mean_validation_score):
        c_max = a.parameters['C']
        score_max = max(score_max,a.mean_validation_score)

clf = SVC(kernel='linear', random_state=241, C=c_max)
clf.fit(x_transformed, y)

df = pd.DataFrame(clf.coef_.toarray())
df = df.T
df = df.abs()

top10 = df.nlargest(n=10, columns = 0, keep = 'first')
output = []

feature_mapping = vectorizer.get_feature_names()
for feature in top10.index.values:
    output.append(feature_mapping[feature])
output.sort()
print(output)
