#!/usr/bin/python3
import json
import re
import string
from collections import Counter
from operator import itemgetter

import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

## Helper functions for Information Retrieval Natural Language Processing on datasets
tfidf = TfidfVectorizer()

nltk.download('stopwords')
def tokenize(text):
    stemmer = PorterStemmer()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    return " ".join([stemmer.stem(word.lower()) for word in tokens])

def tokenize_dataframes(dataframe, columns,key = None, filter=slice(None)):
    documents = {}
    if key == None:
        key = dataframe.index
    for index, element in dataframe[filter].itterows():
        document = ''.join([tokenize(str(dataframe[column])) for column in columns])
        dataframe[key] = document

def search_vec(query, features, threshold=0.1):
    new_features = tfidf.transform([query])
    cosine_similarities = linear_kernel(new_features, features).flatten()
    related_docs_indices, cos_sim_sorted = zip(*sorted(enumerate(cosine_similarities), key=itemgetter(1), 
                                                       reverse=True))
    doc_ids = []
    for i, cos_sim in enumerate(cos_sim_sorted):
        if cos_sim < threshold:
            break
        doc_ids.append(related_docs_indices[i])
    return doc_ids

def test():
    print("blub!")

