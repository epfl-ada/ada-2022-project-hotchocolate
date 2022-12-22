#!/usr/bin/python3

import pandas as pd
import tarfile
import gzip
import re
import requests
import urllib.request
from bs4 import BeautifulSoup
import json
MAX_CSV_SIZE = 1000000
CHUNK_SIZE = 200
def fetch_satellite_data():
    return 0
COLUMNS_NAMES = ["beer_name","beer_id","brewery_name","brewery_id","style","abv","date","username","user_id","appearance","aroma","palate","taste","overall","rating","text"]

def fetch_csv(dataset_path, name):
    with tarfile.open(dataset_path) as tar:
        dataframe = pd.DataFrame()
        for filename in tar.getnames():
            if name in filename:   
                with tar.extractfile(filename) as file:
                    dataframe = pd.read_csv(file)
    return dataframe


def fetch_reviews(dataset_path, max_csv_size = MAX_CSV_SIZE,early_stop = 0):
    """dumps ratings and/or reviews that are in a large text file to multiple csv files of max_csv_size length.
        
        Parameters
        ----------                                  
        dataset_path  (dataframe) :  path to the tar file containing the dataset
        max_csv_size  (int)       : max size of the csvs created by the function
        early_stop    (int)       : for debugging purposes. Stops function at early_stop csvs created. Default is 0 and creates as many csvs as needed
        
        
        Returns
        -------
        location of user with doubled user_ids

   
    """

    tarfile_name = re.search("[ \w-]+?(?=\.)",dataset_path)[0]
    with tarfile.open(dataset_path) as tar:
        datadumps = [filename for filename in tar.getnames() if "txt.gz" in filename]
        print(datadumps)
        filename = input("Please choose a file from the list above to open: ")
        with tar.extractfile(filename) as file:
            with gzip.open(file,'rt') as f:
                review = []
                review_dict = {}
                row_count = 0
                csv_count = 0
                for line in f:
                    if len(line) < 2:
                        review_dict[row_count] = review
                        row_count += 1
                        review = []
                        if row_count % max_csv_size == 0:
                            df = pd.DataFrame.from_dict(review_dict, orient="index")
                            df.columns = COLUMNS_NAMES
                            df.to_csv(f"DATA/{tarfile_name.replace('.tar','')}_{filename.replace('.txt.gz','')}_part_{csv_count}.csv")
                            del review_dict #Just to not kill my memory please disregard :)
                            review_dict = {}
                            print(f"Dumping data to csv number {csv_count}...")
                            csv_count += 1
                            if early_stop and csv_count == early_stop : 
                                break
                    else:
                        (key,value) = line.split(": ", 1)
                        #BeerAdvocate has one column called "review" that is useless and makes everything harder
                        if key != "review":
                            review.append(value.rstrip())
    
    df = pd.DataFrame.from_dict(review_dict, orient="index")
    df.columns=COLUMNS_NAMES
    df.to_csv(f"DATA/{tarfile_name.replace('.tar','')}_{filename.replace('.txt.gz','')}_part_{csv_count}.csv")
    del review_dict #Just to not kill my memory please disregard :)
    print(f"Dumping data to csv number {csv_count}...")
    print("Success!")
    if early_stop :
        return 1
    else : 
        return df
#    except: 
#        print("Euh, no file was found in this path")

def fetch_satellite_df():

    url = "https://satellite.bar/bar/"
    soup = BeautifulSoup(requests.get(url).content, 'html.parser')
    beer = re.findall('var djangoBoissons = (.*?);\s*$', soup.prettify(), re.M)
    beerJson = beer[0]
    satellite_dict = json.loads(beerJson)
    satellite_df = pd.DataFrame(satellite_dict)

    return satellite_df



def find_favourite_beers(website,threshold=10):
    """
        Calculates the beers/beer styles with most votes and with biggest ratings (given there are at least 'threshold ratings')
        
        Parameters
    ----------                                  
    website         (string)  :  name of the website/dataset considered
    threshold       (int)     :  The minimum number of ratings a beer/style need to be considered a valid best beer/style
    
    Returns
    -------
    most_reviewed_beer    (dataframe)
    favorite_beer         (dataframe)
    most_reviewed_style   (dataframe)
    favorite_style        (dataframe)
        
    """
    def correct_double_user_id(users,row):
        """ Routine that corrects the location of users with multiple user_id (case of Ratebeer dataset, when user_id is doubled, one of the entried has no location)
            To be used as an argument of pandas.DataFrame.apply()
            Parameters
        ----------                                  
        users         (dataframe) :  dataframe with user information
        row           (array)     :  row in dataframe corresponding to user with two user_ids 

        
        Returns
        -------
        location of user with doubled user_ids

        """
        name = row["username"].strip()
        return users[users["user_name"] == name]['location'].values[0]
    users = fetch_csv(f"DATA/{website}.tar","users")
    #Unknown location for some users, some users have user_id duplicates either in the users.csv (in this case, no location is available) or in ratings.txt:
    users["location"] = users["location"].fillna("Unknown")
    beer_most_drinked_by_country =pd.DataFrame([])
    style_most_drinked_by_country =pd.DataFrame([])

    if website == "RateBeer":
        TOTAL_CSV = 15
        acronym = "RB"
    if website == "BeerAdvocate":
        TOTAL_CSV = 17
        acronym = "BA"
    print(f"Task : find favourite beers/styles of users. Processing {website} dataset: {TOTAL_CSV} csv files in total.")
    for number in range(0,TOTAL_CSV):
        temp = pd.read_csv(f"DATA/{website}_ratings_part_{number}_corrected_w_attenuation.csv")
        
        temp= temp.merge(users[["user_id","location"]], how="left",left_on= "user_id",right_on="user_id")
        temp["user_id"] = temp["user_id"].apply(lambda x : str(x).strip()) 
        double_id_rows = temp[temp["location"].isna()]
        if len(double_id_rows) != 0:
            temp.loc[temp["location"].isna(),"location"] = temp.loc[temp["location"].isna()].apply(lambda row : correct_double_user_id(users,row),axis=1).apply(str)
    
        temp_grouped_on_beer = temp[["location","beer_name","beer_id","style","brewery_name","user_id","rating"]].groupby(by=["location","beer_name","beer_id","style","brewery_name"]).agg({"user_id":"count","rating":"sum"}).reset_index()
        temp_grouped_on_style = temp[["location","style","user_id","rating"]].groupby(by=["location","style"]).agg({"user_id":"count","rating":"sum"}).reset_index()
        beer_most_drinked_by_country = pd.concat([beer_most_drinked_by_country, temp_grouped_on_beer]).groupby(['location', 'beer_name',"beer_id","style","brewery_name"]).sum().reset_index()
        style_most_drinked_by_country =  pd.concat([style_most_drinked_by_country, temp_grouped_on_style]).groupby(['location', 'style']).sum().reset_index()
    
    print("Calculating most rated and best rated beers and styles...")
    #We renormalize the averages that were aggregated from all parts of the dataset
    beer_most_drinked_by_country["normalized_rating"] = beer_most_drinked_by_country.apply(lambda row: row["rating"]/row["user_id"],axis=1)
    style_most_drinked_by_country["normalized_rating"] = style_most_drinked_by_country.apply(lambda row: row["rating"]/row["user_id"],axis=1)

    #Standardize column names before saving work
    BEER_COLUMN_NAMES = ["location","beer_name","beer_id","style","brewery_name","count","cumulated_rating","normalized_rating"]
    STYLE_COLUMN_NAMES = ["location","style","count","cumulated_rating","normalized_rating"]
    beer_most_drinked_by_country.columns = BEER_COLUMN_NAMES
    style_most_drinked_by_country.columns = STYLE_COLUMN_NAMES

    def most_rated(df):
        df = df.reset_index()
        row = df.iloc[df['count'].idxmax()].copy()
        return row
    def best_rating(df,threshold,metric):
        df = df.reset_index()
        row = df.loc[[0]].copy()
        df = df[df["count"] > threshold].copy()
        if len(df) == 0:
            row[metric] = "No beer with enough votes"
            if metric == "beer_name":
                row["beer_id"] = -1
                row["style"] = "-"
                row["brewery_name"] = "-"
                row["normalized_rating"] = "0"
            return row
        else : 
            row = df.loc[[df['normalized_rating'].idxmax()]].copy()
            return row
    #Create a count of user in each countries     
    count_of_user = pd.DataFrame (users['location'].value_counts())
    count_of_user.rename(columns = {'location':'count_user'}, inplace = True)
    count_of_user['location']= count_test.index
    count_of_user.reset_index(inplace = True, drop = True)
    
    def merge_count_user_data(dataset):
        '''Merge count_of_user with a dataset '''
        dataset = dataset.merge(count_of_user[["location","count_user"]],
                                               how="outer",left_on="location",right_on="location")
        return dataset

    most_reviewed_beer = merge_count_user_data(beer_most_drinked_by_country.groupby(by="location").apply(lambda df : most_rated(df)).reset_index(drop=True))
    
    favorite_beer = merge_count_user_data(beer_most_drinked_by_country.groupby(by="location").apply(lambda df : best_rating(df,threshold,"beer_name")).reset_index(drop=True))

    most_reviewed_style = merge_count_user_data(style_most_drinked_by_country.groupby(by="location").apply(lambda df : most_rated(df)).reset_index(drop=True))
    
    favorite_style = merge_count_user_data(style_most_drinked_by_country.groupby(by="location").apply(lambda df : best_rating(df,threshold,"style")).reset_index(drop=True))
    
    print("Success!")
    return most_reviewed_beer,favorite_beer,most_reviewed_style,favorite_style