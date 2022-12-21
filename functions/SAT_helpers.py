#!/usr/bin/python3
from functions import read_data
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


RB_style_dict = {
            "IPA" : 'India Pale Ale (IPA)',
            "Blanche" : "Belgian Ale",
            "White IPA": "India Pale Ale (IPA)",
            "Sour": 'Sour/Wild Ale',
            "Blonde" : 'Golden Ale/Blond Ale',
            "New England IPA" : "American Pale Ale",
            "Imperial Stout" : "Imperial Stout",
            "Berliner Weisse" : 'Berliner Weisse',
            "Ambrée": 'Amber Ale',
            "Pale Ale" : 'English Pale Ale'
        }

RB_countries_dict = {
            'Royaume-Uni' : 'England',
            "Suisse" : "Switzerland",
            "Norvège": "Norway",
            "Allemagne": 'Germany',
            "Pays-Bas" : "Netherlands",
            "Pologne" : "Poland",
            "Espagne" : 'Spain',
            "France" : "France"
        }

BA_style_dict = {
            "IPA" : 'English India Pale Ale (IPA)',
            "Blanche" : 'Belgian IPA',
            "White IPA": 'American Pale Wheat Ale',
            "Blonde" : 'Belgian IPA',
            "Lambic" : 'Lambic - Fruit',
            "Sour": 'Extra Special / Strong Bitter (ESB)',
            "New England IPA" : 'American IPA',
            "Imperial Stout" : 'American Double / Imperial Stout',
            "Berliner Weisse" : 'Berliner Weissbier',
            "Ambrée": 'American Amber / Red Ale',
            "Pale Ale" :  'American Pale Ale (APA)',
            "Imperial IPA" :  'American Double / Imperial IPA' 
        }

BA_countries_dict = {
        'Royaume-Uni' : 'England',
        "Suisse" : "Switzerland",
        "Norvège": "Norway",
        "Allemagne": 'Germany',
        "Pays-Bas" : "Netherlands",
        "Pologne" : "Poland",
        "Espagne" : 'Spain',
        "France" : "France",
        "Belgique" : "Belgium"}

def generate_automatic_beer_matches(website,matched_dataset):
    """Generates a dataframe with the items/beers of a given 'website' dataset that have more than 0.8 cosine similarity with SAT beers.
    
    Parameters
    ----------
    website         (string)     :  Name of the dataset. Either 'RateBeer' or 'BeerAdvocate'.
    matched_dataset (dataframe)  :  Dataframe with all matches between SAT beers and dataset
    Returns
    -------
    (dataframe) : dataframe of all SAT beers that found a reasonable match in the 'website' dataframe
    (dataframe) : dataframe of all SAT beers that have not found a reasonable match in the 'website' dataframe
    (dataframe) : dataframe with top 5 retrievals of SAT beers without match. Used for manual matching in following step of the pipeline
    """
    if website == "RateBeer":
        acronym = "RB"
    if website == "BeerAdvocate":
        acronym = "BA"
    SAT_beers = read_data.fetch_satellite_df()
    SAT_match_candidates = matched_dataset
    beers = pd.read_csv(f"DATA/{website}_beers_corrected_avg.csv")
    beers =  beers[beers["nbr_ratings"] != 0].copy()
    mask = ((SAT_match_candidates["alcool"] == SAT_match_candidates[f"{acronym}_abv"]) & (SAT_match_candidates[f"{acronym}_similarity"] > 0.8))
    automatic_matches = SAT_match_candidates[mask][["nom",f"{acronym}_beer_name",f"{acronym}_avg",f"{acronym}_abv",f"{acronym}_similarity",f"{acronym}_brewery_name",f"{acronym}_style",f"{acronym}_beer_id"]].drop_duplicates(subset="nom", keep='first', inplace=False, ignore_index=False)
    
    not_matched =  SAT_match_candidates[~mask][["nom",f"{acronym}_beer_name",f"{acronym}_avg",f"{acronym}_abv",f"{acronym}_similarity",f"{acronym}_brewery_name",f"{acronym}_style",f"{acronym}_beer_id"]].drop_duplicates(subset="nom", keep='first', inplace=False, ignore_index=False)
    not_matched =  SAT_beers[~SAT_beers["nom"].isin(automatic_matches["nom"].unique())]
    top5_for_manual_matching = SAT_match_candidates[~SAT_match_candidates["nom"].isin(automatic_matches["nom"].unique())]
    return automatic_matches,not_matched, top5_for_manual_matching



def prepare_features(website,matched_dataset):
    """ Constructs feature vectors for beers. 
        These features are used for training a model on rating estimation and/or to estimate ratings
        . Feature vectors consists of:
        - Alcohol content (float), 
        - dummy variables for country of origin of the brewery (int)
        - dummy variables for the beer style
    
        When given the name of the dataset ('website') and the dataset subset ('matched_dataset')
    consisting only of beers sold at SAT
    
    Parameters
    ----------
    website         (string)     :  Name of the dataset. Either 'RateBeer' or 'BeerAdvocate' 
                                 
    matched_dataset (dataframe)  : dataframe of all the beers sold on SAT that were found 
                on the dataset corresponding with 'website'

    Returns
    -------
    (dataframe) : dataframe without the rating, but with abv (alcohol content) value and 
                dummy variables for all considered features. Used for estimating ratings of SAT beers
    (dataframe) : dataframe with the rating given in the 'website' dataset. Used to train the model
    
    """
    if website == "RateBeer":
        acronym = "RB"
        style_dict = {
            "IPA" : 'India Pale Ale (IPA)',
            "Blanche" : "Belgian Ale",
            "White IPA": "India Pale Ale (IPA)",
            "Sour": 'Sour/Wild Ale',
            "Blonde" : 'Golden Ale/Blond Ale',
            "New England IPA" : "American Pale Ale",
            "Imperial Stout" : "Imperial Stout",
            "Berliner Weisse" : 'Berliner Weisse',
            "Ambrée": 'Amber Ale',
            "Pale Ale" : 'English Pale Ale'
        }
        countries_dict = {
            'Royaume-Uni' : 'England',
            "Suisse" : "Switzerland",
            "Norvège": "Norway",
            "Allemagne": 'Germany',
            "Pays-Bas" : "Netherlands",
            "Pologne" : "Poland",
            "Espagne" : 'Spain',
            "France" : "France"
        }
    if website == "BeerAdvocate":
        acronym = "BA"
        style_dict = {
            "IPA" : 'English India Pale Ale (IPA)',
            "Blanche" : 'Belgian IPA',
            "White IPA": 'American Pale Wheat Ale',
            "Blonde" : 'Belgian IPA',
            "Lambic" : 'Lambic - Fruit',
            "Sour": 'Extra Special / Strong Bitter (ESB)',
            "New England IPA" : 'American IPA',
            "Imperial Stout" : 'American Double / Imperial Stout',
            "Berliner Weisse" : 'Berliner Weissbier',
            "Ambrée": 'American Amber / Red Ale',
            "Pale Ale" :  'American Pale Ale (APA)',
            "Imperial IPA" :  'American Double / Imperial IPA' 
        }
        countries_dict = {
        'Royaume-Uni' : 'England',
        "Suisse" : "Switzerland",
        "Norvège": "Norway",
        "Allemagne": 'Germany',
        "Pays-Bas" : "Netherlands",
        "Pologne" : "Poland",
        "Espagne" : 'Spain',
        "France" : "France",
        "Belgique" : "Belgium"}
    SAT_beers = read_data.fetch_satellite_df()

    beers_to_predict = SAT_beers.loc[~SAT_beers["nom"].isin(matched_dataset["nom"])]
    beers_to_predict["type"] = beers_to_predict["type"].apply(lambda x : style_dict[x])
    beers_to_predict["from"] = beers_to_predict["from"].apply(lambda x : countries_dict[x])
    beers = pd.read_csv(f"DATA/{website}_beers_corrected_avg.csv")
    features_for_traning = beers[["abv","location","style","avg"]]
    features_for_traning.dropna(subset="avg",axis='index',inplace=True)
    features_for_traning.fillna(0,inplace=True)
    SAT_features = beers_to_predict[["alcool","from","type"]]
    SAT_features.columns = ["abv","location","style"]
    sat_beers_to_rate = pd.concat([features_for_traning[["abv","location","style"]],SAT_features],axis=0)
    sat_beers_to_rate_with_dummies =pd.get_dummies(sat_beers_to_rate,columns=["style","location"])
    features = pd.get_dummies(features_for_traning,columns=["style","location"])
    return sat_beers_to_rate_with_dummies.tail(len(beers_to_predict)), features,beers_to_predict, features_for_traning


def randomforest_sat_beers_ratings(features_to_train,features_to_estimate):
    """ Trains a RandomForestRegressor with 'features_to_train' in order to estimate ratings of beers corresponding to 'features_to_estimate'
    
    Parameters
    ----------
    features_to_train    (dataframe)  :  dataframe of features used to train the dataset. Column '1' should be the labels. 
                                 
    features_to_estimate (dataframe)  : dataframe of features used to estimate beer ratings of beers without a match.

    Returns
    -------
    (float) : r2 score of the regression performed
    (np.array) : array with predicted ratings
    
    """
    
    X = pd.concat([features_to_train.iloc[:,0],features_to_train.iloc[:,2:len(features_to_train.columns)]],axis=1)
    y = features_to_train["avg"]
    clf = RandomForestRegressor()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 )
    clf.fit(X_train,y_train)
    y_fitted = clf.predict(X_test)
    r2_score_result = r2_score(y_test,y_fitted)
    predictions = clf.predict(features_to_estimate)
    return r2_score_result, predictions  


def save_and_display_sat_ratings(website,predictions,matching_results,beers_to_predict,training_set):
    """
    saves and displays rating estimation results to a csv and display the full set of ratings of sat beers. 
    Ratings correspond to : 
    - Ratings that were automatically matched with generate_automatic_beer_matches()
    - Ratings that were manually matched
    - Ratings that were estimated with a regressor model with randomforest_sat_beers_ratings()
    
     Parameters
    ----------
    website          (string)    : Name of the dataset. Either 'RateBeer' or 'BeerAdvocate'. 
                                 
    predictions      (np.array)  : Array of predicted ratings for beers without a match in the 'website' dataframe.

    matching_results (dataframe) : Ratings for beers that found a match in 'website' dataframe.
    
    training_set     (dataframe) : dataframe of features used to train the dataset. Column '1' should be the labels. 

    
    """
    if website == "BeerAdvocate":
        acronym = "BA"
    if website == "RateBeer":
        acronym = "RB"
    SAT_beers = read_data.fetch_satellite_df()
    beers_to_predict["predictions"] = predictions
    naive_average = training_set.groupby(by=["abv","location"]).agg({"mean"})
    SAT_ratings = beers_to_predict.merge(naive_average,how="left",left_on=["alcool","from"],right_on=["abv","location"])
    SAT_beers = SAT_beers.merge(matching_results[["nom",f"{acronym}_avg",f"{acronym}_beer_id"]],how="left",on="nom")
    SAT_results = SAT_beers.merge(SAT_ratings[["nom","predictions",('avg', 'mean')]],how="left",left_on="nom",right_on="nom")
    #III.5. As a sanity check and to have an alternative for our model, we naively compute averages 
    #of beers that come from the same country and have the same ABV. We ignore information about beer type
    #in order to have averages over bigger sets (otherwise, some combinations of (origin,abv,type) would have a single element)
    SAT_results[f'{acronym}_avg'].fillna(SAT_results['predictions'],inplace=True)
    SAT_results.rename({f'{acronym}_avg' : 'avg'},inplace=True,axis=1)
    SAT_results.drop_duplicates(subset="nom", keep='first', inplace=True, ignore_index=False)
    SAT_results.sort_values(by="avg",ascending=False,inplace=True)
    SAT_results.to_csv(f"data/predicted_SAT_{acronym}_sorted.csv",index=True)
    display(SAT_results[["nom","type","brasseur","avg"]])

