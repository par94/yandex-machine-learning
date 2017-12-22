import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

newsgroups = datasets.fetch_20newsgroups(
    subset='all', 
    categories=['alt.atheism', 'sci.space']
)
x = newsgroups.data
y = newsgroups.target

vectorizer = TfidfVectorizer()

feature_mapping = vectorizer.get_feature_names()
print (feature_mapping)