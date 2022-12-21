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



def debiasing(website,beer_df,unique_user):
    """corrects for the bias of a given dataset. The correction heuristic is inspired from https://krisjensen.github.io/files/bias_blog.pdf/.
    Correction is implemented with clipping (such that all ratings are between 0 and 5) and attenuation (such that users with only 1 rating are not corrected and that the correction increases with number of ratings)
    
    Parameters
    ----------                                  
    website         (string)  :  name of the website/dataset considered. Either "RateBeer" or "BeerAdvocate"
    beer_df         (pandas.DataFrame) : dataframe with beer ratings that will be corrected
    unique_user     (pandas.DataFrame) : dataframe of beer reviewers (without duplicates) used as a basis to determine systematic reviewer bias 
    
    Returns
    -------
    None
    
    """
    
    def attenuating(row,max_rating):
        """attenuates the bias correction of a specific user. 
        Attenuation is an affine function of the number of ratings of a given user.
        
        Parameters
        -----------
        row: row of the given dataframe
        max_rating: maximum rating found in the dataframe
        
        Returns
        -----------
        attenuation coefficient 
        """
        if row.nbr_ratings==1:
            attenuation_coeff=0 #We cancel the bias if the user only rated once.
        if row.nbr_ratings==max_rating:
            attenuation_coeff=1 #If the user has the rated the most, we do not attenuate their bias.
        else:
            attenuation_coeff=1/(max_rating-1)*row.nbr_ratings-1/(max_rating-1) #for the other users, the bias is attenuated with a coefficient between 0 and 1 affine function of the number of ratings
        return attenuation_coeff
    
    def clip (dataframe):
        """clips a debiased rating such that all ratings are in [0,5] range
        Parameters
        ----------
        dataframe: dataframe on which the ratings will be debiased
        
        Returns
        ----------
        debiased_rating: debiased rating in [0,5] range
        """
        debiased_rating=dataframe['rating']-dataframe['bias'] #We compute the debiased rating as the bias of the user substracted to the initial rating of the user
        if debiased_rating<0: 
            debiased_rating=0 #if the new rating is inferior to 0, we clip it to 0
        if debiased_rating>5: 
            debiased_rating=5 #if the new rating is superior to 5, we clip it to 5
        return debiased_rating
    
    # We define local variables depending on the website
    if website == "RateBeer":
        acronym = "RB"
        NUM_CSV = 15
        average_rating = beer_df.avg.mean()
    if website == "BeerAdvocate":
        acronym = "BA"
        NUM_CSV = 17
        average_rating = beer_df.avg.mean()
    #We iterate over the ratings dataset and group ratings by user.
    grouped_by_users = pd.DataFrame([])
    for i in range(1,NUM_CSV):
        temp = pd.read_csv(f'DATA/{website}_ratings_part_{i}.csv')
        df_partial_grouped_by_users_ratings=temp.groupby(["user_id"]).rating.sum().to_frame()#We group the ratings by users and return the sum of the ratings of each user 
        grouped_by_users = pd.concat([grouped_by_users,df_partial_grouped_by_users_ratings]).groupby(["user_id"]).sum()# We compile the results from  all our csvs    
        del temp
        
        
    grouped_by_users = grouped_by_users.reset_index()
    user_to_nbr_ratings=dict(zip(unique_user.user_id,unique_user.nbr_ratings)) #create a dictionary which keys are user_ids and values are the number of ratings of said users
    grouped_by_users["nbr_ratings"]=grouped_by_users.user_id.map(user_to_nbr_ratings) #we map the number of ratings to the corresponding user_ids
    #We calculate the attenuation coefficient and bias of each user
    maximum_rating=max(grouped_by_users["nbr_ratings"])
    grouped_by_users["attenuation coeff"]=grouped_by_users.apply(lambda row: attenuating(row,maximum_rating),axis=1)
    #Since we computed the sum of ratings of each users when we iterated over the csvs, we need to divide this value by the number of ratings of the user to get the average. The bias is the average rating of the user - average rating of all beers
    grouped_by_users["bias"]=(grouped_by_users["rating"]/grouped_by_users["nbr_ratings"]-average_rating)*grouped_by_users["attenuation coeff"] #this bias must be multiplied by the attenuation coefficient
    #We apply the correction to each rating of the dataframe and clip the rating so its always between [0,5]
    user_to_bias=dict(zip(grouped_by_users.user_id,grouped_by_users.bias)) #We create a dictionary which keys are the user ids and and values are the user biases
    grouped_by_beer = pd.DataFrame([])
    for i in range(0,NUM_CSV): #We iterate over all the csvs
        temp = pd.read_csv(f'DATA/{website}_ratings_part_{i}.csv')
        temp["bias"]=temp.user_id.map(user_to_bias) #we map the bias to its user
        temp["debiased_rating"]=temp["rating"]-temp["bias"] #We substract the bias from the initial rating to get the debiased rating
        temp["debiased_rating"]=temp.apply(clip,axis=1) #Clipping of the new rating
        #Needed to save work, only need to be done once
        #temp.to_csv(f"DATA/{website}_ratings_part_{i}_corrected_w_attenuation.csv")
        partial_grouped_by_beer = temp.groupby(["beer_id"]).debiased_rating.sum().to_frame() #We group the debiased ratings by beer ids and return the sum of ratings for each beers
        grouped_by_beer = pd.concat([grouped_by_beer,partial_grouped_by_beer]).groupby(["beer_id"]).sum() #We compile the results from each csv


    grouped_by_beer=grouped_by_beer.reset_index()
    beer_to_debiased_rating=dict(zip(grouped_by_beer.beer_id,grouped_by_beer.debiased_rating)) #we create a dictionary which keys are the beer ids and values are the sums of debiased ratings
    beer_to_nbr_ratings=dict(zip(beer_df.beer_id,beer_df.nbr_ratings)) #we create a dictionary which keys are the beer ids and the keys are the number of ratings for those beers
    #We compute the debiased average of all beers as the sum of debiased ratings divided by the number of ratings
    beer_df["debiased_avg"]=beer_df.beer_id.map(beer_to_debiased_rating)/beer_df.beer_id.map(beer_to_nbr_ratings) 
    beer_df.to_csv(f"DATA/{website}_beers_corrected_avg.csv",index=False)


Anglo_American_Ales=['Altbier', 'Barley Wine',"Bitter",'Premium Bitter/ESB',"Golden Ale/Blond Ale","Brown Ale", "California Common","Cream Ale","Black IPA","India Pale Ale (IPA)","Imperial IPA","Session IPA","Kölsch","American Pale Ale","Irish Ale","English Strong Ale", "American Strong Ale","Mild Ale","Amber Ale","English Pale Ale","Traditional ALe","Scotch Ale","Old Ale","Scottish Ale"]
Beligan_Style_Ales=["Belgian Ale","Belgian Strong Ale","Bière de Garde","Abbey Dubbel",'Abt/Quadrupel',"Saison","Abbey Tripel"]
Lagers=["Pale Lager","Premium Lager","Imperial Pils/Strong Pale Lager","India Style Lager","Amber Lager/Vienna",'Czech Pilsner (Světlý)',"Pilsener","Heller Bock","Doppelbock","Dumbler Bock","Weizen Bock","Esibock","Malt Liquor","Oktoberfest/Märzen","Radler/Shandy","Zwickel/Keller/Landbier","Dortmunder/Helles",'Dunkel/Tmavý','Schwarzbier','Polotmavý']
Stout_and_Porter=["Stout","Imperial Stout","Foreign Stout","Sweet Stout","Dry Stout","Porter","Baltic Porter","Imperial Porter"]
Wheat_beer=["Wheat Ale","Witbier",'German Hefeweizen','Dunkelweizen','German Kristallweizen']
Sour_beer=["Berliner Weisse","Sour/Wild Ale","Sour Red/Brown",'Grodziskie/Gose/Lichtenhainer','Lambic Style - Gueuze', 'Lambic Style - Unblended','Lambic Style - Faro','Lambic Style - Fruit',"Grodziskie/Gose/Lichtenhainer"]
Other_styles=["Spice/Herb/Vegetable","Smoked",'Fruit Beer',"Sahti/Gotlandsdricke/Koduõlu",'Low Alcohol','Specialty Grain']
Cider_Mead_Saké=['Cider','Mead','Saké - Daiginjo', 'Saké - Namasaké','Saké - Ginjo', 'Saké - Infused', 'Saké - Tokubetsu','Saké - Junmai', 'Saké - Nigori', 'Saké - Koshu', 'Saké - Taru','Saké - Honjozo', 'Saké - Genshu', 'Saké - Futsu-shu','Perry']
beer_style_dict={key: "Anglo American Ales" for key in Anglo_American_Ales}|{key: "Belgian Style Ales" for key in Beligan_Style_Ales}|{key:"Lagers" for key in Lagers}|{key:"Stout and Porter" for key in Stout_and_Porter}|{key:"Wheat beer" for key in Wheat_beer}|{key:"Sour beer" for key in Sour_beer}|{key:"Other styles" for key in Other_styles}|{key:"Cider, Mead and Saké" for key in Cider_Mead_Saké}
