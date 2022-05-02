# coding: utf-8

import os
import shutil
import re

import numpy as np 

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from load_data import preprocess, load_datasets

X_train_data, y_train, X_test_data, y_test = load_datasets()

stopwords = open('dict/stop_words.txt', encoding='utf-8').read().split()

# TF-IDF feature extraction
tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_data)
words = tfidf_vectorizer.get_feature_names()

# naive bayes
print("start naive bayes")
# Pipeline
text_clf = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', MultinomialNB()),
])

text_clf.fit(X_train_data, y_train)

predicted = text_clf.predict(X_test_data)
print(classification_report(predicted, y_test))
# # confusion_matrix(predicted, y_test)

# # Logistic Regression
text_clf_lr = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', LogisticRegression()),
])
text_clf_lr.fit(X_train_data, y_train)
predicted_lr = text_clf_lr.predict(X_test_data)
print(classification_report(predicted_lr, y_test))
# confusion_matrix(predicted_lr, y_test)

# SVM
print("start svm")

#for x in [1.1, 1.2, 1.25, 1.3]:
# print(x)
x=1.2
text_clf_svm = Pipeline([
        ('vect', TfidfVectorizer()),
        ("linear svc", SVC(C=x, kernel="linear"))
    ])
text_clf_svm.fit(X_train_data, y_train)
# text_clf_svm.predict(X_new_data)
predicted_svm = text_clf_svm.predict(X_test_data)
print(classification_report(predicted_svm, y_test))
# confusion_matrix(predicted_svm, y_test)



print("start training random forest")
test_clf_random_forest = Pipeline([
            ('vect', TfidfVectorizer()),
            ("random forest", RandomForestClassifier(n_estimators=1000))
        ])
test_clf_random_forest.fit(X_train_data, y_train)
predicted_random_forest = test_clf_random_forest.predict(X_test_data)
print(classification_report(predicted_random_forest, y_test))