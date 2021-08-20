#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords, wordnet
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize


# # Preprocessing


# Loading the data
data_df = pd.read_csv('train (1).csv', header=None)
data_df.columns = ['text', 'class']
data_df

# Defining a function that will clean, tokenize and lemmatize the data the data:

sw = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


def preprocess(series):
    
    """ This function takes as argument a pd Series of email texts and performs the following operations:
    - cleans the texts
    - pos tags
    - lemmatizes"""
    
    start = time.time()
    
    def clean_df(col):
    
        """ This function takes as input a df column of dtype string, removes stopwords, punctuation, 
        numeric values and emails, filters out words of len(1) and returns the text tokens"""
    
        col = word_tokenize(col.lower())
        col = [word for word in col if word not in sw]
        col = [word for word in col if word not in punctuation]
        col = [word.strip(punctuation) for word in col]
        col = " ".join([word for word in col if not word.isdigit()])
        col = re.sub(r'http\S*|www\S*', '', col)
        col = re.findall(r'[a-z]+', col)
        col = [word for word in col if len(word)>1]
        return col
    
    # Applying clean_df():
    start_time = time.time()
    series = series.map(clean_df)
    print("Cleaning time: ---%s seconds---" % (time.time() - start_time))
    
    # Applying pos_tag:
    start_time = time.time()
    series = series.apply(pos_tag)
    print("Pos tagging time: ---%s seconds---" % (time.time() - start_time))
    
    def pos_tagger(tag_lst):
        
        """ This function takes as input the list of (word,tag) tuples generated by the pos_tag module, 
        translates the tags into wordnet objects and returns a new list of tuples with wordnet tags to be fed
        into the lemmatizer."""
        wordnet_lst = []
        for word, nltk_tag in tag_lst:
            if nltk_tag.startswith('J'):
                wordnet_lst.append((word, wordnet.ADJ))
            elif nltk_tag.startswith('V'):
                wordnet_lst.append((word, wordnet.VERB))
            elif nltk_tag.startswith('N'):
                wordnet_lst.append((word, wordnet.NOUN))
            elif nltk_tag.startswith('R'):
                wordnet_lst.append((word, wordnet.ADV))
            else:          
                wordnet_lst.append((word, None))
        return wordnet_lst
    
    # Applying pos_tagger():
    start_time = time.time()
    series = series.map(pos_tagger)
    print("Translating tags for lemmatizer: ---%s seconds---" % (time.time() - start_time))
    
    #Lemmatizing:
    def lemmata(doc):
        
        """ This function takes as input a list of (word, tag) tuples and returns the lemmatize function
        with pos tags. None tags are ignored"""
        
        lem = []
        for word, tag in doc:
            if tag == None:
                lem.append(word)
            else:
                lem.append(lemmatizer.lemmatize(word, tag))
        return " ".join(lem)
        
    start_time = time.time()
    
    series = series.map(lemmata)
    print("Lemmatizing time: ---%s seconds---" % (time.time() - start_time))
    print("Total preprocessing time: ---%s seconds---" % (time.time() - start))
    
    return series


data_df.text = preprocess(data_df.text)
data_df.drop_duplicates(inplace=True)
clean_data = data_df


# # Training

from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier


x_train, x_test, y_train, y_test = train_test_split(clean_data['text'],
                                                    clean_data['class'],
                                                    test_size=0.05,
                                                    random_state=3
                                                    )
# Vectorize and normalize the data
tf = TfidfVectorizer(min_df=3, ngram_range=(1,3))
x_train = tf.fit_transform(x_train)
x_test = tf.transform(x_test)

x_train = normalize(x_train)
x_test = normalize(x_test)

# Encode the labels
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)


models = [KNeighborsClassifier(),
          SVC(),
          LinearSVC()]


params_set = [{"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"], 
               "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]},
              {"kernel":["linear", "rbf"], "C":[0.0, 1.0, 10.0], "class_weight": [None, "balanced"], 
               "break_ties":[True, False], 'gamma': ['scale', 'auto']},              
              {"loss": ["squared_hinge", "hinge"], "C": [0.1, 1.0, 10.0], "fit_intercept": [True, False], 
              "intercept_scaling": [1, 5, 10], "class_weight": [None, "balanced"]},
              ]

d = dict(zip(models, params_set))

# **Run time approx 21 minutes!
start_time = time.time()
for model in d.keys():
    cv = GridSearchCV(model, d[model], n_jobs=-1)
    cv.fit(x_train, y_train)
    print(model)
    print(cv.best_params_) 
    print(cv.best_score_)

print("Proprocessing time: %s seconds" % (time.time() - start_time))


# ### Using the best params, returned from GridSearch, we measure the Weighted F1 Score of each model.

start = time.time()
knn = KNeighborsClassifier(n_neighbors=6, weights='distance')
knn.fit(x_train, y_train)
p = knn.predict(x_test)
print(knn, f1_score(y_test, p, average='weighted'))
print("%s seconds" % (time.time() - start))


start = time.time()
svc = SVC(kernel='linear', C=10.0, break_ties=True, gamma='scale', class_weight='balanced')
svc.fit(x_train, y_train)
p = svc.predict(x_test)
print(svc, f1_score(y_test, p, average='weighted'))
print("%s seconds" % (time.time() - start))


start = time.time()
lin_svc = LinearSVC(loss='hinge', C=10.0, class_weight=None, fit_intercept=False, intercept_scaling=1)
lin_svc.fit(x_train, y_train)
p = lin_svc.predict(x_test)
print(lin_svc, f1_score(y_test, p, average='weighted'))
print("%s seconds" % (time.time() - start))


# Therefore, we conclude that SVC has the highest score!

# _______________________________________________________________________________________________________________________________________________________________________________________
