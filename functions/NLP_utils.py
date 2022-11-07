#!/usr/bin/python3
import json
import re
import string
from collections import Counter
from operator import itemgetter
import numpy as np
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
def vector_space_retrieval(queries,dataframe,k=1):
    document_dict = tokenize_dataframe(dataframe)
    doc_vectors = tfidf.fit_transform(document_dict.values())
    doc_ids = []
    similarity_coefs = []
    for query in queries.values():
        vector_queries = tfidf.transform([query])
        cosine_similarities = linear_kernel(vector_queries, doc_vectors).flatten()
        related_docs_indices, cos_sim_sorted = zip(*sorted(enumerate(cosine_similarities), key=itemgetter(1), 
                                                        reverse=True))
        for i, cos_sim in enumerate(cos_sim_sorted):
            if i >= k:
                break
            doc_ids.append(related_docs_indices[i])
            similarity_coefs.append(cos_sim)
    new_df = dataframe.iloc[doc_ids].copy()
    new_df["similarity"] = np.array(similarity_coefs)
    return new_df

def tokenize(text):
    stemmer = PorterStemmer()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    return " ".join([stemmer.stem(word.lower()) for word in tokens])

def tokenize_dataframe(dataframe):
    documents = {}
    for index, beer in dataframe.iterrows():
        documents[beer["beer_id"]] = (tokenize(beer["beer_name"]) + " " + tokenize(beer['brewery_name']) + " " + tokenize(str(beer["abv"])))
    print("Documents tokenized")
    return documents

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

def color(df,similarity,threshold=0.6):
    good = f"background-color:#C6F8D3" 
    bad = f"background-color:#F8C6C6" 
    color_df = pd.DataFrame('',index=df.index,columns=df.columns)
    BA_good_fits = similarity[similarity["BA_similarity"] > threshold ].index
    RB_good_fits = similarity[similarity["RB_similarity"] > threshold ].index
    BA_bad_fits = similarity[similarity["BA_similarity"] <= threshold ].index
    RB_bad_fits = similarity[similarity["RB_similarity"] <= threshold ].index
    color_df.loc[BA_good_fits,'BA_beer_name'] = good
    color_df.loc[RB_good_fits,'RB_beer_name'] = good
    color_df.loc[BA_bad_fits,'BA_beer_name'] = bad
    color_df.loc[RB_bad_fits,'RB_beer_name'] = bad
    return color_df

def querify(query_df):
    sat_queries = {}
    for index, biere in query_df.iterrows():
        sat_queries[index] = (tokenize(biere['nom']) +" "  + tokenize(biere['brasseur']) + " " + tokenize(str(biere['alcool'])))
    return sat_queries

def display_results_df(base_df,results_df,reference_column,results):
    new_df = pd.DataFrame()
    new_df[reference_column] = base_df[reference_column]
    new_df[results] = results_df[results].copy()
    return new_df