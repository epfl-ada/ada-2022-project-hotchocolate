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
import fasttext
import warnings
#Do not want warnings about nltk libraries
warnings.filterwarnings("ignore")
## Helper functions for Information Retrieval Natural Language Processing on datasets
tfidf = TfidfVectorizer()

nltk.download('stopwords')
def vector_space_retrieval(queries,dataframe,k=5):
    """
    implementation of Vector Space retrieval with cosine similarities
     Parameters
    ----------
    queries   (pandas.Series)    : dataframe with concatenated tokens used as a 'query' for information retrieval
    dataframe (pandas.DataFrame) : corpus considered for retrieval
    k          (int)             : number of matches to be found


    Returns
    -------
    (pandas.DataFrame) DataFrame with k top matches for 'queries'. Has a column for the cosine similarity value calculated during retrieval
    '''
    """
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
    """Transforms a 'text' into a collection of tokens. Implemented with lowercase, stemming and punctuation removal
    
    Parameters
    ----------
    text         (string)  : text to be tokenizec

    Returns
    -------
    (string) tokenized text
    '''"""
    stemmer = PorterStemmer()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    return " ".join([stemmer.stem(word.lower()) for word in tokens])

def tokenize_dataframe(dataframe):
    """tokenizes a specific combination (abv, beer_name and brewery_name) of columns of a beer dataset. Used for preparing queries and documents information retrieval.
    
    Parameters
    ----------
    dataframe    (string)  : dataframe with columns to be tokenized and used to form documents

    Returns
    -------
    (list(string)) list of documents 
    """
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

def querify(query_df):
    """ produces queries for the SAT beer dataset
    
    Parameters
    ----------
    dataframe    (string)  : dataframe of SAT beers. Must have columns 'nom', 'alcool' and 'brasseur'.

    Returns
    -------
    (list(string)) list of queries
    """
    sat_queries = {}
    for index, biere in query_df.iterrows():
        sat_queries[index] = (tokenize(biere['nom']) +" "  + tokenize(biere['brasseur']) + " " + tokenize(str(biere['alcool'])))
    return sat_queries

def display_results_df(base_df,results_df,reference_column,results):
    new_df = pd.DataFrame()
    new_df[reference_column] = base_df[reference_column]
    new_df[results] = results_df[results].copy()
    return new_df

### Helper functions for word count, language identification and date 
def fetch_ratings(dataset):
    """fetches ratings of beers for plotting and exploratory data analysis

    Parameters
    ----------
    dataset    (string)  : name of the dataset. Either 'BeerAdvocate' or 'RateBeer'

    Returns
    -------
    (pandas.Series) pandas.Series with all the ratings of the dataset
    """
    first = pd.read_csv(f"data/{dataset}_ratings_part_0.csv",low_memory=False)
    if dataset == "BeerAdvocate" :
        first = first[first['rating'] != ' nan']
    if dataset == "RateBeer" :
        first = first[first['rating'] != 'NaN']    
    ratings = first.rating.astype(float)
    if dataset == "BeerAdvocate" :
        csv_count = 17
    else :
        csv_count = 15

    for index in range(1,csv_count):   
        temp = pd.read_csv(f"data/{dataset}_ratings_part_{index}.csv",low_memory=False)
        if dataset == "BeerAdvocate" :
            first = temp[temp['rating'] != 'nan']
        if dataset == "RateBeer" :
            first = temp[temp['rating'] != 'NaN']        
        
        rating = temp.rating.astype(float)
        ratings = pd.concat([ratings, rating])
    return ratings

def summary_analysis(dataset):
    """produces a dataframe of word counts by rating and of datetimes of rating creationf for exploratory data analysis purposes.
    
    Parameters
    ----------
    dataset     (string)  : name of the dataset. Either 'RateBeer' or 'BeerAdvocate
    Returns
    -------
    (series) series of wordcounts for each rating
    (series) series of correctly encoded datetime.Datetimes for the dates of creationof reviews
    '''
    """
    first = pd.read_csv(f"data/{dataset}_ratings_part_0.csv",low_memory=False)
    
    #NaNs between datasets are not standardized in the txt file
    if dataset == "BeerAdvocate" :
        first = first[first['rating'] != ' nan']
    if dataset == "RateBeer" :
        first = first[first['rating'] != 'NaN']
    print("Started counting words and binning dates...")
    first["word_count"] = first.text.apply(lambda x: len(str(x).split()))
    
    #We filter reviews which have only one word (NaNs in BeerAdvocate mostly)
    first = first[first['word_count'] > 1]
    
    counts = first.word_count
    dates = first.date
    #We iterate over the csvs used to keep the data, in order to not load the full dataset in memory
    if dataset == "BeerAdvocate" :
        csv_count = 17
    else :
        csv_count = 15
    for index in range(1,csv_count):
        temp = pd.read_csv(f"data/{dataset}_ratings_part_{index}.csv",low_memory=False)
        temp["word_count"] = temp.text.apply(lambda x: len(str(x).split()))
        #We filter reviews which have only one word (NaNs in BeerAdvocate mostly)
        temp = temp[temp['word_count'] > 1]
        count = temp.word_count
        date = temp.date
        counts = pd.concat([counts, count])
        dates = pd.concat([dates, date])
    print("Done")
    return counts, dates